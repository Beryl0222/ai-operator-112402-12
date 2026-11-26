import {
  createMatrix,
  eye,
  matmul,
  matvec,
  transpose,
  pseudoinverseSym,
  inverseMatrix,
  maxAbsDifference,
  seededNormal,
  norm,
} from './linalg.mjs';

export class MarginalizationError extends Error {
  constructor(message, profile) {
    super(`[MARGINALIZATION_PROFILE] ${message}`);
    this.name = 'MarginalizationError';
    this.code = 'MARGINALIZATION_PROFILE';
    this.profile = profile;
  }
}

export function residualToInformation(R, b, weights, { weightKind = 'sqrt-information' } = {}) {
  let W = null;
  if (weights) {
    if (Array.isArray(weights) && Array.isArray(weights[0])) W = weights;
    else {
      W = createMatrix(weights.length, weights.length);
      weights.forEach((w, i) => {
        W[i][i] = weightKind === 'precision' || weightKind === 'information' ? w : w * w;
      });
    }
  }
  const WR = W ? matmul(W, R) : R;
  const H = matmul(transpose(R), WR);
  let rhs = null;
  if (b) {
    const Wb = W ? matvec(W, b) : b;
    rhs = matvec(transpose(R), Wb);
  }
  return { H, rhs, information: H };
}

function asKey(v) {
  return typeof v === 'string' ? v : (v?.key ?? v?.name ?? null);
}

export function makeLayout({ dimension, variables, blockSize = 6 }) {
  if (!variables?.length) return [{ key: null, start: 0, size: dimension }];
  let offset = 0;
  return variables.map((v) => {
    if (typeof v === 'string') {
      const item = { key: v, start: offset, size: blockSize };
      offset += blockSize;
      return item;
    }
    const size = v.size ?? v.dimension ?? blockSize;
    const start = v.start ?? v.offset ?? offset;
    if (v.start == null) offset += size;
    else offset = start + size;
    return { key: asKey(v), start, size, ...v };
  });
}

export function resolveCoordinates(selection, layout, dimension) {
  if (selection == null) return [];
  const items = Array.isArray(selection) ? selection : [selection];
  const coords = [];
  for (const item of items) {
    if (typeof item === 'number' && Number.isInteger(item)) {
      coords.push(item);
    } else if (Array.isArray(item) && item.length === 2 && item.every(Number.isInteger)) {
      for (let i = item[0]; i < item[1]; i++) coords.push(i);
    } else if (typeof item === 'string' || item?.key != null) {
      const key = asKey(item);
      const block = layout.find((b) => b.key === key);
      if (!block) throw new MarginalizationError(`unknown variable ${key}`, { key, layout });
      for (let i = block.start; i < block.start + block.size; i++) coords.push(i);
    } else if (item?.start != null && item?.size != null) {
      for (let i = item.start; i < item.start + item.size; i++) coords.push(i);
    } else {
      throw new MarginalizationError('cannot resolve marginalization coordinate selection', { item });
    }
  }
  return [...new Set(coords)].sort((a, b) => a - b);
}

function extract(A, rows, cols) {
  return rows.map((r) => cols.map((c) => A[r][c]));
}

function compactVariables(layout, keepSet, keepCoords) {
  // Preserve whole variable blocks when possible.
  const whole = layout.filter((b) => {
    for (let i = b.start; i < b.start + b.size; i++) if (!keepSet.has(i)) return false;
    return true;
  });
  let offset = 0;
  return whole.map((b) => {
    const out = { ...b, start: offset, oldStart: b.start };
    offset += b.size;
    return out;
  }).filter((b) => keepCoords.some((c) => c >= b.oldStart && c < b.oldStart + b.size));
}

export function marginalizePrior(input = {}) {
  let H = input.H ?? input.information;
  let rhs = input.rhs ?? input.r ?? input.vector ?? null;
  const b = input.b ?? null;
  const R = input.R ?? null;
  if ((!H || !rhs) && R) {
    const info = residualToInformation(R, b ?? new Array(R.length).fill(0), input.weights, { weightKind: input.weightKind });
    H = H ?? info.H;
    rhs = rhs ?? info.rhs;
  }
  if (!H) throw new MarginalizationError('marginalizePrior requires H/information or R');

  const dimension = H.length;
  rhs = rhs ? rhs.slice() : new Array(dimension).fill(0);
  if (input.g && !input.rhs) rhs = input.g.map((v) => -v);

  const layout = input.layout ?? makeLayout({ dimension, variables: input.variables, blockSize: input.blockSize ?? 6 });
  const retireCoords = resolveCoordinates(input.retire ?? input.marginalize ?? input.poseVariables, layout, dimension);
  const retireSet = new Set(retireCoords);
  let keepCoords = input.keep ? resolveCoordinates(input.keep, layout, dimension) : [];
  if (!input.keep) {
    keepCoords = Array.from({ length: dimension }, (_, i) => i).filter((i) => !retireSet.has(i));
  }
  const keepSet = new Set(keepCoords);

  if (retireCoords.length === 0) {
    return {
      H, information: H, rhs, g: rhs.map((v) => -v), layout,
      variables: layout, retire: [], keep: keepCoords,
      profile: { exact: true, rank: dimension, nullity: 0, damagedDirections: [] },
      damageProfile: { exact: true, rank: dimension, nullity: 0, damagedDirections: [] },
    };
  }

  const Hrr = extract(H, retireCoords, retireCoords);
  const Hrk = extract(H, retireCoords, keepCoords);
  const Hkr = transpose(Hrk);
  const Hkk = extract(H, keepCoords, keepCoords);
  const rhsR = retireCoords.map((i) => rhs[i]);
  const rhsK = keepCoords.map((i) => rhs[i]);

  const pin = pseudoinverseSym(Hrr, { rankTolerance: input.rankTolerance });
  const correctionH = matmul(matmul(Hkr, pin.value), Hrk);
  const correctionRhs = matvec(matmul(Hkr, pin.value), rhsR);
  const Hkeep = Hkk.map((row, i) => row.map((v, j) => v - correctionH[i][j]));
  const rhsKeep = rhsK.map((v, i) => v - correctionRhs[i]);

  const U0rows = pin.nullBasis; // each entry is a column eigenvector of length r
  const nullity = U0rows.length;
  const leak = nullity ? createMatrix(nullity, keepCoords.length) : [];
  const rhsNull = new Array(nullity).fill(0);
  for (let a = 0; a < nullity; a++) {
    const u = U0rows[a];
    rhsNull[a] = u.reduce((s, v, i) => s + v * rhsR[i], 0);
    for (let j = 0; j < keepCoords.length; j++) {
      let s = 0;
      for (let i = 0; i < retireCoords.length; i++) s += u[i] * Hrk[i][j];
      leak[a][j] = s;
    }
  }

  const damageTol = input.damageTolerance ?? 1e-8 * Math.max(1, ...pin.eigenvalues.map(Math.abs));
  const damagedDirections = [];
  let maxLeak = 0;
  for (let a = 0; a < nullity; a++) {
    const rowNorm = norm(leak[a]);
    maxLeak = Math.max(maxLeak, rowNorm);
    if (rowNorm > damageTol) {
      damagedDirections.push({
        retiredCoordinates: retireCoords,
        retiredDirection: U0rows[a].slice(),
        coupling: leak[a].slice(),
        rhsComponent: rhsNull[a],
        norm: rowNorm,
        coupledKeepCoordinates: keepCoords.filter((_, j) => Math.abs(leak[a][j]) > damageTol),
      });
    }
  }

  const exact = damagedDirections.length === 0;
  const profile = {
    kind: 'schur-marginalization',
    exact,
    rank: pin.rank,
    retiredDimension: retireCoords.length,
    nullity,
    eigenvalues: pin.eigenvalues,
    rankTolerance: pin.tolerance,
    damageTolerance: damageTol,
    maxLeak,
    maxRhsNull: Math.max(0, ...rhsNull.map(Math.abs)),
    damagedDirections,
    retainedCrossBlocksPreserved: true,
    note: exact
      ? 'Retired variables minimized; all off-diagonal blocks among retained parameters were preserved.'
      : 'Rank-deficient retired block has coupling in a null direction; direction is profiled rather than hidden by deleting cross-blocks.',
  };

  if (input.strict && !exact) throw new MarginalizationError(profile.note, profile);

  const nextLayout = compactVariables(layout, keepSet, keepCoords);
  const result = new CalibrationPrior({
    H: Hkeep,
    rhs: rhsKeep,
    R: input.R && keepCoords.length === dimension - retireCoords.length ? undefined : undefined,
    layout: nextLayout.length ? nextLayout : [{ key: null, start: 0, size: keepCoords.length }],
    oldLayout: layout,
    keep: keepCoords,
    retire: retireCoords,
    profile,
    source: input.source ?? null,
    relinearized: false,
    hasSourceObservations: input.hasSourceObservations ?? true,
  });
  return result;
}

export class CalibrationPrior {
  constructor(input) {
    this.H = input.H ?? input.information;
    this.information = this.H;
    this.rhs = input.rhs ?? new Array(this.H.length).fill(0);
    this.g = this.rhs.map((v) => -v);
    this.constant = input.constant ?? 0;
    this.layout = input.layout ?? makeLayout({ dimension: this.H.length, variables: input.variables });
    this.variables = this.layout;
    this.profile = input.profile ?? { exact: true };
    this.damageProfile = this.profile;
    this.errorProfile = this.profile;
    this.source = input.source ?? null;
    this.relinearized = input.relinearized ?? false;
    this.hasSourceObservations = input.hasSourceObservations ?? true;
    this.basisHistory = input.basisHistory ?? [];
  }

  block(keyA, keyB = keyA) {
    const a = this.layout.find((b) => b.key === keyA);
    const b = this.layout.find((b) => b.key === keyB);
    if (!a || !b) return null;
    return Array.from({ length: a.size }, (_, i) =>
      Array.from({ length: b.size }, (_, j) => this.H[a.start + i][b.start + j]));
  }

  quadraticValue(x) {
    let hx = 0;
    for (let i = 0; i < this.H.length; i++) {
      for (let j = 0; j < this.H.length; j++) hx += x[i] * this.H[i][j] * x[j];
    }
    return 0.5 * hx - this.rhs.reduce((s, r, i) => s + r * x[i], 0) + this.constant;
  }

  changeBasis(mapping = {}) {
    return changePriorBasis(this, mapping);
  }

  marginalize(selection, options = {}) {
    return marginalizePrior({
      H: this.H,
      rhs: this.rhs,
      layout: this.layout,
      retire: selection,
      ...options,
    });
  }

  toJSON() {
    return {
      H: this.H,
      information: this.H,
      rhs: this.rhs,
      g: this.g,
      constant: this.constant,
      layout: this.layout,
      variables: this.variables,
      profile: this.profile,
      source: this.source,
      relinearized: this.relinearized,
      hasSourceObservations: this.hasSourceObservations,
      basisHistory: this.basisHistory,
    };
  }
}

export function changePriorBasis(input, maybeMapping) {
  let priorInput = input;
  let mapping = maybeMapping ?? {};
  // Convenience overload: changePriorBasis({H, rhs, J, offset}).
  if (arguments.length === 1 && input && (input.J || input.oldFromNew || input.newFromOld || input.offset)) {
    mapping = input;
    priorInput = input.prior ?? input.old ?? input;
  }
  const old = priorInput instanceof CalibrationPrior ? priorInput : new CalibrationPrior(priorInput);
  const oldDim = old.H.length;
  const newDim = mapping.newDimension ?? mapping.dimension ?? (mapping.J?.[0]?.length ?? oldDim);
  const J = mapping.J ?? eye(oldDim).map((row) => row.slice(0, newDim));
  if (J.length !== oldDim || J[0].length !== newDim) {
    throw new MarginalizationError('basis J must be oldDimension x newDimension', { got: [J.length, J[0]?.length], oldDim, newDim });
  }
  const offset = mapping.offset ?? mapping.delta ?? new Array(oldDim).fill(0);
  const Hnew = matmul(transpose(J), matmul(old.H, J));
  const Hd = matvec(old.H, offset);
  const rhsNew = matvec(transpose(J), old.rhs.map((r, i) => r - Hd[i]));
  const constantNew = old.constant
    + 0.5 * offset.reduce((s, v, i) => s + v * Hd[i], 0)
    - offset.reduce((s, v, i) => s + v * old.rhs[i], 0);

  const migrated = new CalibrationPrior({
    H: Hnew,
    rhs: rhsNew,
    constant: constantNew,
    layout: mapping.layout ?? old.layout.map((b, i) => ({ ...b, start: i * (b.size ?? 6) })),
    profile: {
      kind: 'basis-change',
      exactAtLinearization: true,
      requiresSourceObservationsForRelinearization: mapping.newLinearizationPoint != null,
      warning: mapping.newLinearizationPoint != null
        ? 'Affine tangent exchange is exact for the old quadratic; nonlinear relinearization at a new point requires source observations.'
        : null,
    },
    source: old.source,
    relinearized: false,
    hasSourceObservations: old.hasSourceObservations,
    basisHistory: [...old.basisHistory, { from: mapping.from ?? 'old', to: mapping.to ?? 'new', at: mapping.at ?? null }],
  });

  const verification = verifyBasisChange(old, migrated, J, offset, {
    samples: mapping.samples ?? 8,
    seed: mapping.seed ?? 23,
    tolerance: mapping.verificationTolerance ?? 1e-7,
  });
  migrated.profile.basisVerification = verification;
  if (mapping.strict && !verification.ok) {
    throw new MarginalizationError('basis change failed same-quadratic-form verification', verification);
  }
  const sourceObservationsAvailable = mapping.hasSourceObservations ?? old.hasSourceObservations;
  if (mapping.newLinearizationPoint != null && !sourceObservationsAvailable && mapping.allowApproximate !== true) {
    migrated.profile.relinearizationBlocked = true;
    migrated.profile.status = 'needs-reobservation';
  } else {
    migrated.profile.status = 'migrated';
  }
  return migrated;
}

export function verifyBasisChange(oldPrior, newPrior, J, offset, { samples = 8, seed = 23, tolerance = 1e-7 } = {}) {
  const random = seededNormal(seed);
  let maxDifference = 0;
  const checks = [];
  for (let s = 0; s < samples; s++) {
    const y = Array.from({ length: newPrior.H.length }, () => random() * (s === 0 ? 0 : 0.3));
    const x = matvec(J, y).map((v, i) => v + offset[i]);
    const oldValue = oldPrior.quadraticValue(x);
    const newValue = newPrior.quadraticValue(y);
    const diff = Math.abs(oldValue - newValue);
    maxDifference = Math.max(maxDifference, diff);
    checks.push({ y, oldValue, newValue, diff });
  }
  // Also check zero/new origin explicitly.
  {
    const y = new Array(newPrior.H.length).fill(0);
    const x = offset.slice();
    const diff = Math.abs(oldPrior.quadraticValue(x) - newPrior.quadraticValue(y));
    maxDifference = Math.max(maxDifference, diff);
  }
  const scale = Math.max(1, ...checks.map((c) => Math.abs(c.oldValue)), ...checks.map((c) => Math.abs(c.newValue)));
  return {
    ok: maxDifference <= tolerance * scale + tolerance,
    maxDifference,
    scaledTolerance: tolerance * scale + tolerance,
    samples: samples + 1,
  };
}

export function assertSameQuadraticForm(a, b, options = {}) {
  const result = verifyBasisChange(a, b, options.J ?? eye(a.H.length), options.offset ?? new Array(a.H.length).fill(0), options);
  if (!result.ok) throw new MarginalizationError('quadratic forms differ after basis exchange', result);
  return result;
}

// Marginal covariance of retained coordinates.  Unlike information-form
// Schur complement, this simply selects the retained principal block; all
// retained/retained cross-covariances stay intact.
export function marginalizeCovariance(P, options = {}) {
  const dimension = P.length;
  const layout = options.layout ?? makeLayout({ dimension, variables: options.variables, blockSize: options.blockSize ?? 6 });
  const keep = options.keep ? resolveCoordinates(options.keep, layout, dimension) : Array.from({ length: dimension }, (_, i) => i);
  const cov = extract(P, keep, keep);
  return { covariance: cov, keep, layout: compactVariables(layout, new Set(keep), keep) };
}

export function changeResidualBasis(input = {}) {
  const R = input.R;
  const b = input.b ?? new Array(R.length).fill(0);
  const J = input.J ?? input.oldFromNew ?? eye(R[0].length);
  const offset = input.offset ?? input.delta ?? new Array(R[0].length).fill(0);
  // R (J y + offset) - b = (R J)y - (b - R offset).
  const Rnew = matmul(R, J);
  const Roffset = matvec(R, offset);
  const bnew = b.map((bi, i) => bi - Roffset[i]);
  return { R: Rnew, b: bnew };
}

// Coordinate convention:
//   old = J * new + offset  (default when J/oldFromNew supplied)
// Therefore C_old = J C_new J^T and C_new = J^{-1} C_old J^{-T}.
// If newFromOld=B is supplied, C_new = B C_old B^T.
export function changeCovarianceBasis(cov, mapping = {}) {
  let B = mapping.newFromOld ?? mapping.B ?? null;
  let J = mapping.oldFromNew ?? mapping.J ?? null;
  if (!B) {
    if (!J) J = eye(cov.length);
    const Jinv = inverseMatrix(J);
    B = Jinv;
  }
  const next = matmul(matmul(B, cov), transpose(B));
  const checkJ = J ?? inverseMatrix(B);
  const reconstructed = matmul(matmul(checkJ, next), transpose(checkJ));
  const maxDifference = maxAbsDifference(reconstructed, cov);
  return {
    covariance: next,
    J: checkJ,
    newFromOld: B,
    verification: { ok: maxDifference <= (mapping.tolerance ?? 1e-7) * Math.max(1, maxAbsDifference(cov, eye(cov.length))), maxDifference },
  };
}

// Accumulate adjoint/Jacobian calibration measurements into information form.
// A measurement may contain several variable blocks; all J_i^T W J_j cross
// terms are retained instead of reducing the problem to diagonal blocks.
export function buildInformationFromAdjoints(measurements = [], options = {}) {
  const keys = options.variables ?? [...new Set(measurements.flatMap((m) => Object.keys(m.jacobians ?? { [m.key]: true })).filter((k) => typeof k === 'string'))];
  const layout = options.layout ?? makeLayout({ variables: keys, blockSize: options.blockSize ?? 6 });
  const n = layout.reduce((max, b) => Math.max(max, b.start + b.size), 0);
  const H = createMatrix(n, n);
  const rhs = new Array(n).fill(0);

  for (const m of measurements) {
    const blocks = m.jacobians ?? { [m.key]: m.jacobian ?? m.adjoint };
    let W = m.information ?? m.weightMatrix ?? null;
    if (!W && m.covariance) W = pseudoinverseSym(m.covariance, { rankTolerance: m.rankTolerance }).value;
    if (!W && Array.isArray(m.weights)) {
      W = createMatrix(6, 6);
      m.weights.forEach((w, i) => { W[i][i] = m.weightKind === 'sqrt-information' ? w * w : w; });
    }
    if (!W) W = eye(6);
    const b = m.b ?? null;
    for (const [keyI, Ji] of Object.entries(blocks)) {
      const bi = layout.find((x) => x.key === keyI);
      if (!bi || !Ji) continue;
      const JtW = matmul(transpose(Ji), W);
      if (b) {
        const update = matvec(JtW, b);
        for (let r = 0; r < 6; r++) rhs[bi.start + r] += update[r];
      }
      for (const [keyJ, Jj] of Object.entries(blocks)) {
        const bj = layout.find((x) => x.key === keyJ);
        if (!bj || !Jj) continue;
        const term = matmul(JtW, Jj);
        for (let r = 0; r < bi.size; r++) {
          for (let c = 0; c < bj.size; c++) H[bi.start + r][bj.start + c] += term[r][c];
        }
      }
    }
  }
  return { H, information: H, rhs, layout };
}

export const marginalize = marginalizePrior;
export const marginalizePose = marginalizePrior;
export const marginalizePoses = marginalizePrior;
export const schurMarginalize = marginalizePrior;
export const changeBasis = changePriorBasis;
