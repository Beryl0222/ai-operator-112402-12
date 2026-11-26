import { quaternionExp, rotationLog, se3Exp, toPose, multiplyPoses, invertPose, solveIncrement } from './pose.mjs';
import { composeTransforms } from './compose.mjs';

// Convert the end-side model's compact delta into an SE(3) pose.  Supported:
//  - [wx,wy,wz] rotation vector
//  - [wx,wy,wz,vx,vy,vz] in the documented tangent order
//  - {omega|rotationVector|rvec, v|translation|t}
//  - normal pose objects {q,t} / matrix forms
export function deltaToPose(delta) {
  if (delta == null) return toPose({ q: [1, 0, 0, 0], t: [0, 0, 0] });
  if (Array.isArray(delta)) {
    if (delta.length === 3) return { ...se3Exp([...delta, 0, 0, 0]) };
    if (delta.length === 6) return { ...se3Exp(delta) };
    return toPose(delta);
  }
  if (delta.omega || delta.rotationVector || delta.rvec || delta.rotation) {
    const omega = delta.omega ?? delta.rotationVector ?? delta.rvec ?? delta.rotation;
    const translation = delta.v ?? delta.translation ?? delta.t ?? [0, 0, 0];
    return se3Exp([...omega.slice(0, 3), ...translation.slice(0, 3)]);
  }
  return toPose(delta.pose ?? delta.transform ?? delta);
}

// Correct replacement for element-wise addition of returned rotation vectors.
// It multiplies Exp(r_i) on the unit quaternion manifold and returns the
// principal log plus branch diagnostics.
export function composeRotationVectors(vectors, options = {}) {
  const quaternions = vectors.map((v) => quaternionExp(v));
  const q = quaternions.reduce((acc, qi) => multiplyPoses({ q: acc, t: [0, 0, 0] }, { q: qi, t: [0, 0, 0] }).q, [1, 0, 0, 0]);
  const log = rotationLog(q, options);
  return { ...log, q, quaternions };
}

export function composeDeltas(deltas = [], options = {}) {
  const steps = deltas.map((d, index) => {
    if (typeof d !== 'object' || Array.isArray(d)) {
      return { pose: deltaToPose(d), side: options.perturbation ?? options.side ?? 'right' };
    }
    return {
      ...d,
      pose: deltaToPose(d),
      side: d.perturbation ?? d.side ?? options.perturbation ?? options.side ?? 'right',
      originalIndex: index,
    };
  });
  return composeTransforms(steps, options);
}

// Reverse/model inverse: given the composed output and one known side, solve
// for the missing increment.
export function inverseModel(output, known = {}) {
  return solveIncrement(output, known);
}

export function modelBackward(deltas, gradient, options = {}) {
  return composeDeltas(deltas, options).backward(gradient, options);
}

export { solveIncrement, invertPose };

export const reverseModel = modelBackward;
export const reversePoseModel = modelBackward;
export const poseReverse = solveIncrement;
export const composePoseDeltas = composeDeltas;
export const composeIncrements = composeDeltas;
export const composePoseIncrements = composeDeltas;
export const invertPoseIncrement = inverseModel;
