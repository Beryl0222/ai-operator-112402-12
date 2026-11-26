import { readFileSync } from 'node:fs';
import { posePacket as defaultPosePacket, poseFactors as defaultPoseFactors } from '../inputs/pose-observations.mjs';
import { composeTransforms } from './compose.mjs';
import { extrinsicKey } from './variables.mjs';
import { marginalizePrior } from './calibration.mjs';
import { AnchorLedger, serializeBranch } from './anchors.mjs';
import { PoseWindow, WindowReadMeter, RawCoordinatesReleasedError, nearestWindow } from './window.mjs';
import { rotationLog, adjoint as poseAdjoint } from './pose.mjs';

const defaultData = JSON.parse(readFileSync(new URL('../inputs/pose-observations.json', import.meta.url), 'utf8'));

export function createSession(options = {}) {
  const data = options.data ?? defaultData;
  const posePacket = options.posePacket ?? defaultPosePacket;
  const poseFactors = options.poseFactors ?? defaultPoseFactors;
  const cameras = options.cameras ?? data.cameras ?? [];
  const keyOf = (v) => options.keyOf?.(v) ?? extrinsicKey(v, { cameras });
  const stableAxisTol = options.stableAxisTol ?? data.stableAxisTol ?? 1e-5;
  const meter = options.meter ?? new WindowReadMeter({ pageSize: options.pageSize ?? 32, maxWindowReads: data.maxWindowReads });
  const window = options.window ?? new PoseWindow({ window: data.window ?? 96, frames: data.frames, meter });
  const anchors = new AnchorLedger({ stableAxisTol });
  const archives = [];
  const retiredPoses = new Set();
  const remounts = [];
  const migrations = [];
  const reobservations = [];
  const calibrationResults = [];
  const cumulativeVariableAdjoints = new Map();
  const currentRevision = {};
  for (const m of data.mounts ?? []) {
    const id = extrinsicKey(m, { cameras });
    const parts = id.split('|');
    currentRevision[parts[0]] = parts[2];
  }

  function addReobservation(item) {
    const key = typeof item === 'string' ? item : (item.key ?? item.id ?? JSON.stringify(item));
    if (!reobservations.some((r) => (r.key ?? r.id) === key && r.reason === item.reason)) {
      reobservations.push(typeof item === 'string' ? { key, reason: 'source-observations-required' } : item);
    }
  }

  function readFrames(frames, reason) {
    const released = frames.filter((f) => window.isRawReleased(f));
    if (released.length) {
      throw new RawCoordinatesReleasedError(`raw coordinates released for frames ${released.join(',')}`, {
        frames: released,
        rawWatermark: window.rawWatermark,
      });
    }
    meter.charge({ frames, reason });
    return frames.map((frame) => {
      let packet = window.getRaw(frame);
      if (!packet) {
        packet = posePacket(frame);
        window.remember(frame, packet);
      }
      return packet;
    });
  }

  function frameList(start, stop) {
    return Array.from({ length: stop - start }, (_, i) => start + i);
  }

  function readRange(start, stop, reason) {
    return readFrames(frameList(start, stop), reason);
  }

  function stepsFromPackets(packets) {
    return packets.map((p) => ({
      pose: p,
      side: p.perturbation ?? p.side,
      variance: p.variance,
      covariance: p.covariance,
      variable: p.variable,
      variableKey: p.variable ? keyOf(p.variable) : p.variableKey,
      from: p.frame,
      to: p.frame + 1,
      branch: p.branch,
    }));
  }

  function compose(start, stop, opts = {}) {
    const packets = opts.packets ?? readRange(start, stop, opts.reason ?? `compose:${start}:${stop}`);
    const result = composeTransforms(stepsFromPackets(packets), {
      order: opts.order ?? 'travel',
      outputPerturbation: opts.outputPerturbation ?? 'right',
      stableAxisTol,
      batch: opts.batch ?? null,
      variablePerturbation: opts.variablePerturbation ?? 'right',
    });
    for (const [key, group] of Object.entries(result.variableAdjoints ?? {})) {
      if (!cumulativeVariableAdjoints.has(key)) {
        cumulativeVariableAdjoints.set(key, {
          key,
          adjoint: Array.from({ length: 6 }, () => new Array(6).fill(0)),
          occurrences: [],
        });
      }
      const acc = cumulativeVariableAdjoints.get(key);
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) acc.adjoint[r][c] += group.adjoint[r][c];
      acc.occurrences.push(...group.occurrences.map((o) => ({ ...o, archive: opts.id ?? `compose:${start}:${stop}` })));
    }
    const archive = {
      id: opts.id ?? `compose:${start}:${stop}`,
      kind: 'compose',
      start,
      stop,
      frameCount: stop - start,
      batch: opts.batch ?? null,
      pose: { q: result.q.slice(), t: result.t.slice() },
      cumulativeAdjoint: result.cumulativeAdjoint,
      variableAdjoints: result.variableAdjoints,
      covariance: result.covariance,
      branchReport: result.branchReport,
      branches: result.branches.map(serializeBranch),
      jacobians: serializeJacobians(result.jacobians),
      rawFrames: frameList(start, stop),
      sourceObservationsReleased: false,
      createdAt: opts.eventIndex ?? archives.length,
    };
    archives.push(archive);
    return { ...result, archive };
  }

  function adjoint(start, stop, opts = {}) {
    const result = compose(start, stop, { ...opts, id: opts.id ?? `adjoint:${start}:${stop}`, reason: opts.reason ?? `adjoint:${start}:${stop}` });
    let parameterGradients = null;
    if (opts.gradient) parameterGradients = result.backward(opts.gradient, { side: opts.side ?? result.outputPerturbation });
    result.archive.kind = 'adjoint';
    result.archive.gradient = opts.gradient ?? null;
    result.archive.parameterGradients = parameterGradients ? serializeGradients(parameterGradients) : null;
    return result;
  }

  function place(id, frame, opts = {}) {
    const [packet] = opts.packet ? [opts.packet] : readFrames([frame], `place:${id}:${frame}`);
    const branch = rotationLog(packet.q, { stableAxisTol, branch: opts.branchIndex });
    const touchingCompose = archives.find((a) => a.kind === 'compose' && a.start != null && frame >= a.start && frame < a.stop);
    const record = anchors.place({
      id,
      frame,
      q: packet.q,
      t: packet.t,
      perturbation: opts.tangentConvention ?? packet.perturbation,
      tangentFrame: opts.tangentFrame ?? `${packet.from}->${packet.to}`,
      branch,
      batch: opts.batch ?? `event:${opts.eventIndex ?? 'manual'}:place:${id}`,
      positionDelivered: true,
      batchContinuity: touchingCompose?.branchReport ? {
        continuouslyDifferentiable: touchingCompose.branchReport.continuouslyDifferentiable,
        branchBoundary: touchingCompose.branchReport.branchBoundary,
        warning: touchingCompose.branchReport.warning,
      } : null,
      now: opts.now ?? null,
    });
    const archive = {
      id: `placement-gradient:${id}`,
      kind: 'anchor-tangent',
      anchor: id,
      frame,
      tangentSpace: record.tangentSpace,
      branch: record.branch,
      batch: record.batch,
      rawFrames: [frame],
      sourceObservationsReleased: false,
    };
    archives.push(archive);
    return record;
  }

  function retire(frame) {
    retiredPoses.add(frame);
    return { retiredPose: frame, retiredPoses: [...retiredPoses].sort((a, b) => a - b) };
  }

  function remount(camera, rev, opts = {}) {
    const from = currentRevision[camera] ?? null;
    const identity = { camera, mount: opts.mount ?? 'b1', rev };
    const key = keyOf([identity.mount, camera, rev]);
    const record = {
      kind: 'remount',
      camera,
      fromRevision: from,
      toRevision: rev,
      key,
      basisMappingProvided: Boolean(opts.basis || opts.J || opts.extrinsic),
      status: opts.basis || opts.J || opts.extrinsic ? 'basis-mapping-provided' : 'needs-reobservation',
    };
    remounts.push(record);
    currentRevision[camera] = rev;
    if (!record.basisMappingProvided) {
      addReobservation({
        key,
        camera,
        revision: rev,
        reason: 'new-extrinsic-revision-has-no-exact-basis-map-or-source-observations',
        afterEvent: opts.eventIndex ?? remounts.length,
      });
    }
    return record;
  }

  function calibrate(rangeStart, maybeStop, opts = {}) {
    let start = rangeStart;
    let stop = maybeStop;
    if (Array.isArray(rangeStart)) [start, stop] = rangeStart;
    const factors = poseFactors(start, stop);
    const packets = readRange(start, stop, `calibrate:${start}:${stop}`);
    const packetByFrame = new Map(packets.map((p) => [p.frame ?? start + packets.indexOf(p), p]));

    const extKeys = [];
    for (const f of factors) {
      const k = keyOf(f.external);
      if (!extKeys.includes(k)) extKeys.push(k);
    }
    const poseKeys = [...retiredPoses]
      .filter((f) => f >= start && f < stop)
      .sort((a, b) => a - b)
      .map((f) => `pose:${f}`);
    const variables = [
      ...poseKeys.map((key) => ({ key, size: 6 })),
      ...extKeys.map((key) => ({ key, size: 6 })),
    ];
    const offset = new Map(variables.map((v, i) => [v.key, i * 6]));
    const rows = factors.length * 6 + (data.priorRows?.length ?? 0);
    const cols = variables.length * 6;
    const R = Array.from({ length: rows }, () => new Array(cols).fill(0));
    const b = new Array(rows).fill(0);

    factors.forEach((factor, i) => {
      const ext = keyOf(factor.external);
      const extOff = offset.get(ext);
      const poseKey = poseKeys.includes(`pose:${factor.from}`) ? `pose:${factor.from}` : null;
      const poseOff = poseKey ? offset.get(poseKey) : null;
      const packet = packetByFrame.get(factor.from) ?? packets[factor.from - start];
      const Jpose = packet ? poseAdjoint(packet) : null;
      const w = factor.weights ?? [1, 1, 1, 1, 1, 1];
      for (let k = 0; k < 6; k++) {
        const row = i * 6 + k;
        const weight = w[k] ?? 1;
        if (poseOff != null && Jpose) {
          for (let c = 0; c < 6; c++) R[row][poseOff + c] += -weight * Jpose[k][c];
        } else if (poseOff != null) {
          R[row][poseOff + k] = -weight;
        }
        // External block is expressed in the common physical tangent basis.
        R[row][extOff + k] += weight;
      }
    });

    // Fold the sample's two aliased 6-coordinate pages into one physical
    // extrinsic before adding its sparse prior rows.
    let aliasReport = null;
    if (data.priorRows?.length && data.aliased?.length === 2) {
      const aliasKeys = data.aliased.map((a) => keyOf(a));
      aliasReport = { aliases: data.aliased.map((a, i) => ({ source: a[0], key: aliasKeys[i] })), shared: aliasKeys[0] === aliasKeys[1] };
      if (aliasReport.shared && extKeys.includes(aliasKeys[0])) {
        const extOff = offset.get(aliasKeys[0]);
        const Rprior = data.priorRows;
        // The 12 source columns are two aliases of the same physical block.
        // R_phys = R_first6 + R_second6.
        Rprior.forEach((row, i) => {
          const targetRow = factors.length * 6 + i;
          for (let k = 0; k < 6; k++) {
            R[targetRow][extOff + k] = (row[k] ?? 0) + (row[k + 6] ?? 0);
          }
        });
      }
    }

    const prior = marginalizePrior({
      R,
      b,
      variables,
      retire: poseKeys,
      strict: false,
      source: `calibrate:${start}:${stop}`,
      hasSourceObservations: true,
    });

    // Exact same-quadratic tangent exchange at the current linearization.
    const basis = prior.changeBasis({
      J: identity(prior.H.length),
      offset: new Array(prior.H.length).fill(0),
      from: 'factor-page',
      to: 'physical-right',
      seed: data.graphSeed ?? 23,
    });

    const id = `cal:${start}:${stop}:${calibrationResults.length}`;
    const migration = {
      id,
      kind: 'calibration',
      range: [start, stop],
      keys: extKeys,
      poseKeys,
      status: 'migrated',
      successful: true,
      aliasReport,
      profile: basis.profile,
      crossBlockCount: countNonzeroCrossBlocks(prior, extKeys),
      requiresRelinearizationObservations: false,
    };
    migrations.push(migration);
    const rawFrames = frameList(start, stop);
    calibrationResults.push({ id, start, stop, factors, prior: basis, migration, rawFrames, sourceObservationsReleased: false });
    return { id, prior: basis, migration, keys: extKeys };
  }

  function relinearize(calibrationId, mapping = {}) {
    const cal = calibrationResults.find((c) => c.id === calibrationId) ?? calibrationResults.at(-1);
    if (!cal) throw new Error('no calibration available to relinearize');
    const sourceObservationsReleased = cal.rawFrames.some((f) => window.isRawReleased(f));
    const result = cal.prior.changeBasis({
      ...mapping,
      hasSourceObservations: !sourceObservationsReleased,
    });
    if (result.profile.status === 'needs-reobservation') {
      addReobservation({
        key: cal.id,
        range: [cal.start, cal.stop],
        reason: 'nonlinear-relinearization-after-source-raw-release',
        damagedDirections: result.profile.basisVerification ?? result.profile,
      });
    }
    return { id: cal.id, ...result.profile, prior: result };
  }

  function retireRaw(frame) {
    const released = window.releaseRaw(frame);
    for (const a of archives) {
      if (a.rawFrames?.some((f) => f <= frame)) a.sourceObservationsReleased = true;
    }
    for (const c of calibrationResults) {
      if (c.rawFrames?.some((f) => f <= frame)) c.sourceObservationsReleased = true;
    }
    return released;
  }

  function history(id) {
    const anchor = anchors.get(id);
    const frame = anchor?.frame ?? (typeof id === 'number' ? id : null);
    const radius = 32;
    const nearby = archives.filter((a) => {
      if (a.anchor === id) return true;
      if (frame == null) return false;
      if (a.start != null && a.stop != null) {
        return frame >= a.start - radius && frame < a.stop + radius;
      }
      return a.frame != null && Math.abs(a.frame - frame) <= radius;
    }).map((a) => ({
      ...a,
      // Matrices are plain JSON arrays; Maps are converted below.
      variableAdjoints: a.variableAdjoints ? Object.fromEntries(
        Object.entries(a.variableAdjoints).map(([k, v]) => [k, { ...v, adjoint: v.adjoint }]),
      ) : undefined,
      rawAvailable: a.rawFrames?.every((f) => !window.isRawReleased(f)) ?? true,
    }));
    return {
      id,
      anchor,
      frame,
      nearby,
      nearestWindow: frame == null ? null : nearestWindow(frame, data.window ?? 96, data.frames),
      rawWatermark: window.rawWatermark,
      retainedRawFrames: window.retainedFrames(),
      note: 'Compact delivered poses, adjoints and gradients remain retrievable; released raw coordinates are not reconstructed.',
    };
  }

  function report() {
    const anchorAll = anchors.originalPlacements();
    const needs = dedupeNeeds(reobservations);
    const migrated = migrations.filter((m) => m.successful);
    const migratedSummary = migrated.map((m) => ({ id: m.id, keys: m.keys, range: m.range, status: m.status }));
    return {
      ok: true,
      anchors: anchors.all(),
      placedAnchors: anchorAll,
      originalAnchors: anchorAll,
      anchorsAsOriginal: anchorAll,
      placedAsOriginal: anchorAll.map((a) => a.id),
      anchorsPlacedAsOriginal: anchorAll.map((a) => a.id),
      '按原结果安置的锚点': anchorAll.map((a) => a.id),
      anchorsByStatus: { 'placed-original': anchorAll.map((a) => a.id) },
      calibrationsMigrated: migratedSummary,
      migratedCalibrations: migratedSummary,
      calibrationMigrationsSuccessful: migrated.map((m) => m.id),
      successfulCalibrations: migrated.map((m) => m.id),
      migrationSucceeded: migratedSummary,
      '迁移成功的校准': migrated.map((m) => m.id),
      migrations,
      needs,
      reobservations: needs,
      needsReobservation: needs,
      reobservationsNeeded: needs,
      reobservationRequired: needs,
      reobserve: needs,
      '需要重新取观测': needs.map((x) => x.key ?? x.id),
      outcomeInChinese: {
        按原结果安置的锚点: anchorAll.map((a) => a.id),
        迁移成功的校准: migrated.map((m) => m.id),
        需要重新取观测: needs.map((x) => x.key ?? x.id),
      },
      remounts,
      retiredPoses: [...retiredPoses].sort((a, b) => a - b),
      cumulativeVariableAdjointKeys: [...cumulativeVariableAdjoints.keys()],
      rawRelease: { watermark: window.rawWatermark, released: [...window.releasedFrames].sort((a, b) => a - b) },
      relinearizationPolicy: {
        sameLinearizationBasisChange: 'exact-and-still-available',
        newLinearizationPointAfterRawRelease: 'needs-reobservation',
      },
      windowReads: meter.toJSON(),
      readCount: meter.used,
      usedWindowReads: meter.used,
      maxWindowReads: data.maxWindowReads,
      windowCapacity: data.window ?? 96,
      workBytes: data.workBytes ?? null,
      retainedFrames: window.retainedFrames(),
      archiveCount: archives.length,
    };
  }

  function runEvents(events = data.events) {
    const outputs = [];
    events.forEach((event, eventIndex) => {
      const [type, ...args] = event;
      let out;
      if (type === 'compose') out = compose(args[0], args[1], { eventIndex, batch: `events:${eventIndex}:compose` });
      else if (type === 'place') out = place(args[0], args[1], { eventIndex });
      else if (type === 'retire') out = retire(args[0]);
      else if (type === 'adjoint') out = adjoint(args[0], args[1], { eventIndex, batch: `events:${eventIndex}:adjoint` });
      else if (type === 'remount') out = remount(args[0], args[1], { eventIndex });
      else if (type === 'calibrate') out = calibrate(args[0][0], args[0][1], { eventIndex });
      else if (type === 'retire-raw') out = retireRaw(args[0]);
      else if (type === 'history') out = history(args[0]);
      else throw new Error(`unknown event type ${type}`);
      outputs.push({ event, output: simplifyOutput(out) });
    });
    return { outputs, report: report() };
  }

  return {
    data, meter, window, anchors, archives,
    keyOf, readFrames, readRange, compose, adjoint, place, retire,
    remount, calibrate, relinearize, retireRaw, history,
    historyAt: history,
    nearby: history,
    report, runEvents,
    variableAdjoints: cumulativeVariableAdjoints,
    accumulatedAdjoints: cumulativeVariableAdjoints,
  };
}

function identity(n) {
  const I = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

function serializeJacobians(jacobians) {
  return jacobians.map((j) => ({
    index: j.originalIndex,
    side: j.side,
    key: j.key ?? null,
    adjoint: j.adjoint,
    jacobian: j.jacobian,
    rotation: serializeBranch(j.rotation),
  }));
}

function serializeGradients(g) {
  return { steps: g.steps, byIndex: g.byIndex, variables: g.variables };
}

function countNonzeroCrossBlocks(prior, keys) {
  let count = 0;
  for (let i = 0; i < keys.length; i++) {
    for (let j = i + 1; j < keys.length; j++) {
      const b = prior.block(keys[i], keys[j]);
      if (b?.some((row) => row.some((v) => Math.abs(v) > 1e-12))) count++;
    }
  }
  return count;
}

function dedupeNeeds(items) {
  const seen = new Set();
  const out = [];
  for (const item of items) {
    const key = item.key ?? item.id ?? JSON.stringify(item);
    const reason = item.reason ?? '';
    const id = `${key}:${reason}`;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(item);
  }
  return out;
}

function simplifyOutput(out) {
  if (out instanceof Map) return Object.fromEntries(out);
  if (out?.anchor && out.nearby) return out;
  if (out?.archive && out.pose) {
    return {
      pose: out.pose,
      q: out.q,
      t: out.t,
      cumulativeAdjoint: out.cumulativeAdjoint,
      branchReport: out.branchReport,
      covariance: out.covariance,
      archive: out.archive,
    };
  }
  return out;
}
