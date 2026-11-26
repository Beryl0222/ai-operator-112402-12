import test from 'node:test';
import assert from 'node:assert/strict';
import {
  posePacket,
} from '../inputs/pose-observations.mjs';
import {
  rotationLog,
  relativeTransformSequence,
  composeRotationVectors,
  composeTransforms,
  se3Exp,
  multiplyPoses,
  invertPose,
  se3Log,
  adjoint,
  marginalizePrior,
  changePriorBasis,
  extrinsicKey,
  pagesForRange,
  WindowReadMeter,
  createSession,
  trackTransforms,
} from '../index.mjs';

test('half-turn packets keep principal branch and expose pi ambiguity', () => {
  const tol = 1e-5;
  const l177 = rotationLog(posePacket(177).q, { stableAxisTol: tol });
  const l178 = rotationLog(posePacket(178).q, { stableAxisTol: tol });
  const l179 = rotationLog(posePacket(179).q, { stableAxisTol: tol });

  assert.ok(l177.angle < Math.PI);
  assert.equal(l178.ambiguous, true);
  assert.equal(l179.axis[2] < 0, true);
  assert.equal(l177.axis[2] > 0, true);

  const rel = relativeTransformSequence([178, 179].map(posePacket), { stableAxisTol: tol });
  assert.ok(Math.abs(rel[0].log.vector[2] - 1e-4) < 1e-8);

  const twoThirdsPi = (2 * Math.PI) / 3;
  const composedRotations = composeRotationVectors(
    [[0, 0, twoThirdsPi], [0, 0, twoThirdsPi]],
    { stableAxisTol: tol },
  );
  assert.ok(composedRotations.angle > 2 * Math.PI / 3);
  assert.ok(composedRotations.axis[2] < 0);
  assert.ok(Math.abs(composedRotations.angle - 2 * Math.PI / 3) < 1e-9);
});

test('mixed left/right composition Jacobians match finite differences', () => {
  let seed = 23;
  const rand = () => {
    seed = (1664525 * seed + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const delta = () => Array.from({ length: 6 }, () => (rand() - 0.5) * 0.4);
  const steps = ['left', 'right', 'left'].map((side) => ({ pose: se3Exp(delta()), side }));
  const composed = composeTransforms(steps, { outputPerturbation: 'right' });

  for (let i = 0; i < steps.length; i++) {
    for (let c = 0; c < 6; c++) {
      const eps = 1e-6;
      const perturb = (sign) => {
        const e = new Array(6).fill(0);
        e[c] = sign * eps;
        const d = se3Exp(e);
        const pose = steps[i].side === 'left' ? multiplyPoses(d, steps[i].pose) : multiplyPoses(steps[i].pose, d);
        return steps.map((s, j) => (j === i ? { ...s, pose } : s));
      };
      const plus = composeTransforms(perturb(1), { outputPerturbation: 'right' });
      const minus = composeTransforms(perturb(-1), { outputPerturbation: 'right' });
      const fd = se3Log(multiplyPoses(invertPose(minus.pose), plus.pose)).vector.map((v) => v / (2 * eps));
      const analytic = composed.jacobians[i].jacobian.right.map((row) => row[c]);
      for (let r = 0; r < 6; r++) assert.ok(Math.abs(fd[r] - analytic[r]) < 2e-5);
    }
  }

  const grad = composed.backward([1, 2, 3, 4, 5, 6]);
  assert.equal(grad.steps.length, 3);
  assert.ok(grad.steps.every((g) => g.length === 6));
});

test('shared extrinsic aliases accumulate one physical adjoint block', () => {
  const E = se3Exp([0.1, -0.2, 0.05, 0.3, 0.1, -0.2]);
  const result = composeTransforms([
    { pose: E, side: 'right', variableKey: 'zoom|b1|v4' },
    { pose: se3Exp([0, 0, 0, 0.2, 0, 0]), side: 'left' },
    { pose: E, side: 'left', variableKey: 'zoom|b1|v4' },
  ], { outputPerturbation: 'right', variablePerturbation: 'right' });

  assert.ok(result.variableAdjoints['zoom|b1|v4']);
  assert.equal(result.variableAdjoints['zoom|b1|v4'].occurrences.length, 2);
  assert.equal(extrinsicKey(['a183', 'b1', 'zoom', 'v4'], { cameras: ['zoom'] }), 'zoom|b1|v4');
  assert.equal(extrinsicKey(['b97', 'b1', 'zoom', 'v4'], { cameras: ['zoom'] }), 'zoom|b1|v4');
});

test('ordered track validates start-to-end chaining', () => {
  const a = se3Exp([0.1, 0, 0, 0, 0, 0]);
  const b = se3Exp([0, 0.2, 0, 0, 0, 0]);
  const out = trackTransforms({
    start: 'f0', end: 'f2',
    steps: [
      { from: 'f0', to: 'f1', pose: a },
      { from: 'f1', to: 'f2', pose: b },
    ],
  });
  assert.equal(out.from, 'f0');
  assert.equal(out.to, 'f2');
  assert.throws(() => trackTransforms({
    steps: [
      { from: 'f0', to: 'f1', pose: a },
      { from: 'f2', to: 'f3', pose: b },
    ],
  }), /do not chain/);
});

test('Schur marginalization retains retained-variable cross blocks', () => {
  const H = [
    [4, 1, 0.7, 0.2, 0.1, 0],
    [1, 3, 0.3, 0.8, 0, 0.1],
    [0.7, 0.3, 5, 2, 0.6, 0.4],
    [0.2, 0.8, 2, 4, 0.2, 0.9],
    [0.1, 0, 0.6, 0.2, 3, 1],
    [0, 0.1, 0.4, 0.9, 1, 3],
  ];
  const rhs = [1, 2, 3, 4, 5, 6];
  const m = marginalizePrior({
    H, rhs,
    variables: [{ key: 'p', size: 2 }, { key: 'a', size: 2 }, { key: 'b', size: 2 }],
    retire: ['p'],
  });
  assert.equal(m.profile.exact, true);
  const ab = m.block('a', 'b');
  assert.ok(Math.abs(ab[0][0] - 0.5836363636363636) < 1e-9);
  assert.ok(ab.some((row) => row.some((v) => Math.abs(v) > 0.1)));
});

test('rank deficient leaking direction is profiled, not hidden by dropping blocks', () => {
  const H = [
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
  ];
  const m = marginalizePrior({
    H,
    rhs: [1, 1, 2, 1],
    variables: [{ key: 'p', size: 2 }, { key: 'e', size: 2 }],
    retire: ['p'],
    rankTolerance: 1e-9,
    damageTolerance: 1e-8,
  });
  assert.equal(m.profile.exact, false);
  assert.equal(m.profile.damagedDirections.length, 1);
  assert.ok(m.profile.maxLeak > 0.9);
});

test('basis exchange verifies the same quadratic form', () => {
  const H = [
    [3, .2, .1],
    [.2, 2, .3],
    [.1, .3, 1.5],
  ];
  const rhs = [.4, -.2, .1];
  const prior = marginalizePrior({ H, rhs, variables: [{ key: 'e', start: 0, size: 3 }], retire: [] });
  // A non-identity invertible tangent-coordinate exchange.
  const J = [
    [1, .1, 0],
    [0, 1, .05],
    [.02, 0, 1],
  ];
  const offset = [.01, -.02, .005];
  const next = changePriorBasis(prior, { J, offset, samples: 20 });
  assert.equal(next.profile.basisVerification.ok, true);
});

test('page meter uses 32-frame pages and counts every page in a batch', () => {
  assert.deepEqual(pagesForRange(175, 180, 32), [5]);
  assert.deepEqual(pagesForRange(160, 193, 32), [5, 6]);
  const meter = new WindowReadMeter({ pageSize: 32, maxWindowReads: 2 });
  meter.charge({ start: 175, stop: 180 });
  assert.throws(() => meter.charge({ start: 160, stop: 193 }), /maxWindowReads/);
});

test('sample event run preserves anchor, migrates calibration and reports reobservation', () => {
  const session = createSession();
  const run = session.runEvents();
  const report = run.report;
  assert.deepEqual(report.anchorsPlacedAsOriginal, ['sub3']);
  assert.equal(report.anchors[0].tangentSpace.convention, 'left');
  assert.ok(report.anchors[0].batch);
  assert.equal(report.calibrationMigrationsSuccessful.length, 1);
  assert.ok(report.needsReobservation.some((x) => x.key === 'zoom|b1|v5'));
  assert.ok(report.windowReads.used <= report.windowReads.maxWindowReads);

  const historyEvent = run.outputs.find((o) => o.event[0] === 'history');
  assert.ok(historyEvent.output.nearby.some((a) => a.kind === 'adjoint'));
  assert.ok(historyEvent.output.nearby.some((a) => a.cumulativeAdjoint));

  assert.throws(() => session.readFrames([175], 'after-release'), /RAW_COORDINATES_RELEASED/);
  const history = session.history('sub3');
  assert.equal(history.anchor.status, 'placed-original');
  assert.ok(history.nearby.length >= 2);

  const relinearized = session.relinearize(undefined, { newLinearizationPoint: new Array(12).fill(0.01) });
  assert.equal(relinearized.status, 'needs-reobservation');
  assert.ok(session.report().needsReobservation.some((x) => x.key.startsWith('cal:')));
});
