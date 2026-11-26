import {
  adjoint,
  invertPose,
  multiplyPoses,
  toPose,
  normalizeSide,
  rotationLog,
  covarianceFromVariance,
  covarianceFromWeights,
  solveIncrement,
  PERTURBATION_ORDER,
} from './pose.mjs';
import { createMatrix, matmul, transpose } from './linalg.mjs';

let BATCH_SEQUENCE = 0;

export function nextBatchId(prefix = 'batch') {
  BATCH_SEQUENCE += 1;
  return `${prefix}-${Date.now?.() ?? 'node'}-${BATCH_SEQUENCE}`;
}

export function resetBatchSequence() {
  BATCH_SEQUENCE = 0;
}

function stepCovariance(step) {
  if (step.covariance) return covarianceFromVariance(step.covariance);
  if (step.variance) return covarianceFromVariance(step.variance);
  if (step.weights) return covarianceFromWeights(step.weights, { kind: step.weightKind });
  return null;
}

function identity6() {
  const I = createMatrix(6, 6);
  for (let i = 0; i < 6; i++) I[i][i] = 1;
  return I;
}

// Compose ordered SE(3) transforms and retain the exact tangent Jacobians.
//
// order:
//  - 'multiplication' (default): list is M1*M2*...*Mn
//  - 'travel'/'ordered': list follows frames from start to end, so the matrix
//    product is Mn*...*M1.
export function composeTransforms(transforms = [], options = {}) {
  const requestedOrder = options.order ?? 'multiplication';
  const travel = requestedOrder === 'travel' || requestedOrder === 'ordered' || options.ordered === true;
  const original = transforms.map((item, originalIndex) => normalizeStep(item, originalIndex, options));
  const mathSteps = travel ? original.map((_, i) => original[original.length - 1 - i]) : original.slice();

  const n = mathSteps.length;
  const prefix = new Array(n + 1);
  const suffix = new Array(n);
  prefix[0] = toPose({ q: [1, 0, 0, 0], t: [0, 0, 0] });
  for (let i = 0; i < n; i++) prefix[i + 1] = multiplyPoses(prefix[i], mathSteps[i].pose);
  const total = prefix[n];

  suffix[n - 1] = toPose({ q: [1, 0, 0, 0], t: [0, 0, 0] });
  for (let i = n - 2; i >= 0; i--) suffix[i] = multiplyPoses(mathSteps[i + 1].pose, suffix[i + 1]);

  const stableAxisTol = options.stableAxisTol ?? 1e-5;
  const mathJacobians = mathSteps.map((step, i) => {
    const A = prefix[i];
    const AB = prefix[i + 1];
    const B = suffix[i];
    const left = step.side === 'left' ? adjoint(A) : adjoint(AB);
    const right = step.side === 'right' ? adjoint(invertPose(B)) : adjoint(multiplyPoses(invertPose(B), invertPose(step.pose)));
    const rotation = rotationLog(step.pose.q, { stableAxisTol, branch: step.branch });
    return {
      originalIndex: step.originalIndex,
      index: step.originalIndex,
      key: step.key,
      variable: step.variable,
      variables: step.variables,
      side: step.side,
      pose: step.pose,
      rotation,
      jacobian: { left, right },
      adjoint: step.side === 'left' ? left : right,
      covariance: stepCovariance(step),
      raw: step.raw,
    };
  });

  const jacobians = original.map((_, idx) => mathJacobians.find((j) => j.originalIndex === idx));
  const outputSide = normalizeSide(options.outputPerturbation ?? options.outputSide ?? 'right');
  const covariance = {
    left: propagateCovariance(mathJacobians, 'left'),
    right: propagateCovariance(mathJacobians, 'right'),
  };

  const branchReport = analyzeBranches(jacobians.map((j) => j.rotation), options);
  const variableAdjoints = accumulateVariableAdjoints(jacobians, total, {
    physicalSide: options.variablePerturbation ?? options.physicalSide ?? outputSide,
    outputSide,
  });

  const result = {
    pose: total,
    q: total.q,
    t: total.t,
    R: total.R,
    T: total.T,
    adjoint: adjoint(total),
    cumulativeAdjoint: adjoint(total),
    startFrame: options.startFrame ?? original[0]?.from ?? options.from ?? null,
    endFrame: options.endFrame ?? original.at(-1)?.to ?? options.to ?? null,
    startCoordinateSystem: options.startCoordinateSystem ?? options.start ?? options.startFrame ?? original[0]?.from ?? options.from ?? null,
    endCoordinateSystem: options.endCoordinateSystem ?? options.end ?? options.endFrame ?? original.at(-1)?.to ?? options.to ?? null,
    from: options.from ?? original[0]?.from ?? null,
    to: options.to ?? original.at(-1)?.to ?? null,
    order: travel ? 'travel' : 'multiplication',
    outputPerturbation: outputSide,
    perturbationOrder: PERTURBATION_ORDER,
    jacobians,
    variableAdjoints,
    covariance: covariance[outputSide],
    covariances: covariance,
    branchReport,
    branches: jacobians.map((j) => j.rotation),
    batch: options.batch ?? null,
    backward(gradient, backwardOptions = {}) {
      const side = normalizeSide(backwardOptions.side ?? outputSide);
      const g = gradient.slice ? gradient.slice() : Array.from(gradient);
      const byIndex = new Array(transforms.length);
      const variableGradientMap = new Map();
      const variables = {};
      for (const j of jacobians) {
        const Jt = transpose(j.jacobian[side]);
        const value = Array.from({ length: 6 }, (_, r) => {
          let s = 0;
          for (let k = 0; k < 6; k++) s += Jt[r][k] * g[k];
          return s;
        });
        byIndex[j.originalIndex] = value;

        for (const v of enumerateVariables(j)) {
          const basis = variableBasis(j.pose, j.side, v.physicalSide ?? options.variablePerturbation ?? outputSide, v.basis);
          const basisT = transpose(basis);
          const key = v.key;
          const physical = Array.from({ length: 6 }, (_, r) => {
            let sum = 0;
            for (let k = 0; k < 6; k++) sum += basisT[r][k] * value[k];
            return sum;
          });
          if (!variableGradientMap.has(key)) {
            const first = physical.slice();
            variableGradientMap.set(key, first);
            variables[key] = first;
          } else {
            const cur = variableGradientMap.get(key);
            physical.forEach((x, i) => { cur[i] += x; });
            variables[key] = cur;
          }
        }
      }
      return { steps: byIndex, byIndex, variables, variableMap: variableGradientMap };
    },
  };
  result.modelInverse = (known = {}) => solveIncrement(total, known);
  return result;
}

function normalizeStep(item, originalIndex, options) {
  if (item && item.__normalizedStep) return item;
  const raw = item;
  const poseLike = item?.pose ?? item?.transform ?? item?.T ?? item?.matrix ?? item;
  const pose = toPose(poseLike);
  const side = normalizeSide(
    item?.perturbation ?? item?.side ?? options.defaultPerturbation ?? options.defaultSide ?? 'right',
  );
  let variables = item.variables;
  if (!variables && (item.variable || item.variableKey || item.key)) {
    variables = [{
      key: item.variableKey ?? item.key ?? (typeof item.variable === 'string' ? item.variable : undefined),
      descriptor: item.variable,
      physicalSide: item.variablePerturbation,
      basis: item.variableBasis,
    }];
  }
  return {
    __normalizedStep: true,
    originalIndex,
    pose,
    side,
    from: item.from ?? null,
    to: item.to ?? null,
    key: item.key ?? null,
    variable: item.variable ?? null,
    variables,
    branch: item.branch,
    covariance: item.covariance,
    variance: item.variance,
    weights: item.weights,
    weightKind: item.weightKind,
    raw,
  };
}

function propagateCovariance(jacobians, side) {
  const out = createMatrix(6, 6);
  for (const j of jacobians) {
    if (!j.covariance) continue;
    const J = j.jacobian[side];
    const term = matmul(matmul(J, j.covariance), transpose(J));
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) out[r][c] += term[r][c];
  }
  // Guard against -0 leaking into snapshots/tests.
  return out.map((row) => row.map((v) => (Math.abs(v) < 1e-15 ? 0 : v)));
}

function enumerateVariables(j) {
  if (j.variables?.length) {
    return j.variables.map((v) => (typeof v === 'string' ? { key: v } : v));
  }
  if (j.key) return [{ key: j.key }];
  return [];
}

function variableBasis(pose, occurrenceSide, physicalSide, explicitBasis) {
  if (explicitBasis) return explicitBasis;
  const phys = normalizeSide(physicalSide ?? 'right');
  if (occurrenceSide === phys) return identity6();
  if (occurrenceSide === 'left' && phys === 'right') return adjoint(pose);
  // occurrence right, physical left
  return adjoint(invertPose(pose));
}

export function accumulateVariableAdjoints(jacobians, totalPose, options = {}) {
  const physicalSide = normalizeSide(options.physicalSide ?? 'right');
  const outputSide = normalizeSide(options.outputSide ?? physicalSide ?? 'right');
  const groups = new Map();
  for (const j of jacobians) {
    for (const v of enumerateVariables(j)) {
      if (!v.key) continue;
      const basis = variableBasis(j.pose, j.side, v.physicalSide ?? physicalSide, v.basis);
      const J = matmul(j.jacobian[outputSide], basis);
      if (!groups.has(v.key)) {
        const zero = createMatrix(6, 6);
        groups.set(v.key, { key: v.key, adjoint: zero, cumulativeAdjoint: zero, jacobian: zero, occurrences: [], physicalSide, outputSide });
      }
      const group = groups.get(v.key);
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
        group.adjoint[r][c] += J[r][c];
        group.cumulativeAdjoint[r][c] += J[r][c];
        group.jacobian[r][c] += J[r][c];
      }
      group.occurrences.push({ index: j.originalIndex, side: j.side, jacobian: J });
    }
  }
  for (const group of groups.values()) {
    group.blocks = group.occurrences.map((o) => o.jacobian);
    group.stackedAdjoint = group.blocks.flat();
    group.summedAdjoint = group.adjoint;
    group.cumulativeAdjoint = group.adjoint;
  }
  return Object.fromEntries(groups);
}

export function analyzeBranches(logs, options = {}) {
  const crossings = [];
  for (let i = 1; i < logs.length; i++) {
    const a = logs[i - 1];
    const b = logs[i];
    const axisDot = a.axis[0] * b.axis[0] + a.axis[1] * b.axis[1] + a.axis[2] * b.axis[2];
    const nearBoundary = a.nearPi || b.nearPi || Math.abs(Math.PI - a.angle) < (options.stableAxisTol ?? 1e-5) * 10
      || Math.abs(Math.PI - b.angle) < (options.stableAxisTol ?? 1e-5) * 10;
    if (nearBoundary && axisDot < -0.5) {
      crossings.push({
        between: [i - 1, i],
        kind: 'principal-branch-sign-change',
        axisDot,
        previous: a,
        current: b,
        continuouslyDifferentiable: false,
      });
    }
  }
  return {
    crossings,
    branchBoundary: logs.some((l) => l.nearPi) || crossings.length > 0,
    continuouslyDifferentiable: crossings.length === 0,
    warning: crossings.length ? 'SO(3) log branch boundary; retain principal branches and do not unwrap adjacent axes as smooth derivatives.' : null,
  };
}

// Convenience for absolute packet sequences.  It computes true relative
// rotations by quaternion product instead of element-wise log addition.
export function relativeTransformSequence(poses, options = {}) {
  const ps = poses.map(toPose);
  return ps.slice(1).map((next, i) => ({
    from: i,
    to: i + 1,
    pose: multiplyPoses(invertPose(ps[i]), next),
    log: rotationLog(multiplyPoses(invertPose(ps[i]), next).q, options),
  }));
}

export const composeOrdered = (steps, options = {}) => composeTransforms(steps, { ...options, order: 'travel' });
export const composeTravel = composeOrdered;
export const chainTransforms = composeOrdered;
export const compose = composeTransforms;
export const poseComposition = composeTransforms;
export const cumulativeAdjointForVariables = accumulateVariableAdjoints;
export const cumulativeAdjoints = accumulateVariableAdjoints;
export const adjointCumulative = accumulateVariableAdjoints;
export const accumulateAdjoints = accumulateVariableAdjoints;
export const backwardCompose = (transforms, gradient, options = {}) =>
  composeTransforms(transforms, options).backward(gradient, options);
