import {
  eye,
  skew,
  matmul,
  matvec,
  createMatrix,
  transpose,
} from './linalg.mjs';

export const PERTURBATION_ORDER = Object.freeze(['wx', 'wy', 'wz', 'vx', 'vy', 'vz']);

export class PoseError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = 'PoseError';
    Object.assign(this, details);
  }
}

export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

export function normalizeQuaternion(q) {
  if (!Array.isArray(q) || q.length !== 4) {
    throw new PoseError('quaternion must be [w,x,y,z]', { quaternion: q });
  }
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!Number.isFinite(n) || n === 0) throw new PoseError('invalid quaternion norm', { quaternion: q });
  return q.map((v) => v / n);
}

export function quaternionMultiply(a, b) {
  const [aw, ax, ay, az] = normalizeQuaternion(a);
  const [bw, bx, by, bz] = normalizeQuaternion(b);
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

export function quaternionConjugate(q) {
  const [w, x, y, z] = normalizeQuaternion(q);
  return [w, -x, -y, -z];
}

export function quaternionInverse(q) {
  return quaternionConjugate(q); // normalizeQuaternion already divides by norm^2 effectively for unit q.
}

export function rotateVector(q, v) {
  const r = quaternionToMatrix(q);
  return matvec(r, v);
}

export function quaternionToMatrix(input) {
  const [w, x, y, z] = normalizeQuaternion(input);
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return [
    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
  ];
}

export function matrixToQuaternion(r) {
  const tr = r[0][0] + r[1][1] + r[2][2];
  let q;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [(0.25 * s), (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s];
  } else if (r[0][0] > r[1][1] && r[0][0] > r[2][2]) {
    const s = Math.sqrt(1 + r[0][0] - r[1][1] - r[2][2]) * 2;
    q = [(r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s];
  } else if (r[1][1] > r[2][2]) {
    const s = Math.sqrt(1 + r[1][1] - r[0][0] - r[2][2]) * 2;
    q = [(r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s];
  } else {
    const s = Math.sqrt(1 + r[2][2] - r[0][0] - r[1][1]) * 2;
    q = [(r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s];
  }
  return normalizeQuaternion(q);
}

export function so3Exp(omega) {
  const w = omega.slice(0, 3);
  const theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  const K = skew(w);
  if (theta2 < 1e-16) {
    return [
      [1 + K[0][0] + 0.5 * (matmul(K, K))[0][0], K[0][1] + 0.5 * matmul(K, K)[0][1], K[0][2] + 0.5 * matmul(K, K)[0][2]],
      [K[1][0] + 0.5 * matmul(K, K)[1][0], 1 + K[1][1] + 0.5 * matmul(K, K)[1][1], K[1][2] + 0.5 * matmul(K, K)[1][2]],
      [K[2][0] + 0.5 * matmul(K, K)[2][0], K[2][1] + 0.5 * matmul(K, K)[2][1], 1 + K[2][2] + 0.5 * matmul(K, K)[2][2]],
    ];
  }
  const theta = Math.sqrt(theta2);
  const k = scale3(K, 1 / theta);
  const k2 = matmul(k, k);
  const s = Math.sin(theta) / theta;
  const c = (1 - Math.cos(theta)) / theta2;
  // R = I + sin(theta)/theta K + (1-cos)/theta² K²
  return add3(eye(3), add3(scale3(K, Math.sin(theta) / theta), scale3(matmul(K, K), c)));
}

function scale3(a, s) {
  return a.map((row) => row.map((v) => v * s));
}
function add3(a, b) {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

export function quaternionExp(omega) {
  const [x, y, z] = omega;
  const theta = Math.hypot(x, y, z);
  if (theta < 1e-12) {
    const half = 0.5;
    return normalizeQuaternion([1 - theta * theta / 8, half * x, half * y, half * z]);
  }
  const half = theta / 2;
  const k = Math.sin(half) / theta;
  return [Math.cos(half), k * x, k * y, k * z];
}

// Principal SO(3) logarithm.  The returned angle is always in [0, pi].
// Near pi the axis is numerically/topologically ambiguous and the returned
// profile says so; callers must not silently unwrap neighbouring branches.
export function rotationLog(input, { stableAxisTol = 1e-5, branch = 0 } = {}) {
  let q;
  let r;
  if (Array.isArray(input) && input.length === 4 && typeof input[0] === 'number') {
    q = normalizeQuaternion(input);
    r = quaternionToMatrix(q);
  } else if (isMatrix3(input)) {
    r = input;
    q = matrixToQuaternion(r);
  } else if (input?.q) {
    return rotationLog(input.q, { stableAxisTol, branch: input.branch ?? branch });
  } else {
    throw new PoseError('rotationLog expects quaternion [w,x,y,z] or 3x3 matrix');
  }

  // Canonical quaternion hemisphere makes the principal angle unambiguous.
  let canonical = q;
  let sign = 1;
  if (canonical[0] < 0) {
    canonical = canonical.map((v) => -v);
    sign = -1;
  }
  const [w, x, y, z] = canonical;
  const vnorm = Math.hypot(x, y, z);
  const cosAngleHalf = clamp(w, -1, 1);
  let theta = 2 * Math.atan2(vnorm, Math.max(0, cosAngleHalf));
  if (!Number.isFinite(theta)) theta = Math.acos(clamp((r[0][0] + r[1][1] + r[2][2] - 1) / 2, -1, 1));

  let axis;
  if (vnorm > 1e-12) axis = [x / vnorm, y / vnorm, z / vnorm];
  else {
    // Small-angle matrix log is more stable than dividing by quaternion vector.
    axis = [0.5 * (r[2][1] - r[1][2]), 0.5 * (r[0][2] - r[2][0]), 0.5 * (r[1][0] - r[0][1])];
    const n = Math.hypot(...axis);
    axis = n > 0 ? axis.map((v) => v / n) : [1, 0, 0];
  }
  if (theta < 1e-12) {
    const omega = [0.5 * (r[2][1] - r[1][2]), 0.5 * (r[0][2] - r[2][0]), 0.5 * (r[1][0] - r[0][1])];
    return {
      vector: omega,
      angle: Math.hypot(...omega),
      axis: [1, 0, 0],
      branch,
      principalBranch: 0,
      nearPi: false,
      ambiguous: false,
      stable: true,
      stableAxisTol,
      canonicalSign: sign,
      candidates: [omega],
    };
  }

  const vector = axis.map((v) => theta * v);
  const piDistance = Math.abs(Math.PI - theta);
  const nearPi = piDistance <= stableAxisTol || Math.abs(w) <= stableAxisTol;
  return {
    vector,
    angle: theta,
    axis,
    branch,
    principalBranch: 0,
    nearPi,
    ambiguous: nearPi,
    stable: !nearPi,
    piDistance,
    stableAxisTol,
    canonicalSign: sign,
    candidates: nearPi ? [vector, vector.map((v) => -v)] : [vector],
  };
}

function isMatrix3(a) {
  return Array.isArray(a) && a.length === 3 && a.every((r) => Array.isArray(r) && r.length === 3);
}

export function se3Exp(delta) {
  const d = delta.slice(0, 6);
  const omega = d.slice(0, 3);
  const v = d.slice(3, 6);
  const theta = Math.hypot(...omega);
  const R = so3Exp(omega);
  let V;
  if (theta < 1e-10) {
    const K = skew(omega);
    V = add3(eye(3), scale3(K, 0.5));
  } else {
    const K = skew(omega);
    const k2 = matmul(K, K);
    V = add3(eye(3), add3(scale3(K, (1 - Math.cos(theta)) / (theta * theta)), scale3(k2, (theta - Math.sin(theta)) / (theta * theta * theta))));
  }
  return { q: matrixToQuaternion(R), t: matvec(V, v), R, T: poseMatrixFromRt(R, matvec(V, v)) };
}

export function se3Log(pose, options = {}) {
  const p = toPose(pose);
  const log = rotationLog(p.q, options);
  const omega = log.vector;
  const theta = log.angle;
  let Vinv;
  if (theta < 1e-10) {
    Vinv = eye(3);
  } else {
    const K = skew(omega);
    const half = theta / 2;
    // V^{-1} = I - 1/2 K + (1/theta - 1/2 cot(theta/2))/theta K²
    const coef = (1 / theta - 0.5 / Math.tan(half)) / theta;
    Vinv = add3(eye(3), add3(scale3(K, -0.5), scale3(matmul(K, K), coef)));
  }
  const v = matvec(Vinv, p.t);
  return { vector: [...omega, ...v], rotation: log };
}

export function poseMatrixFromRt(R, t) {
  return [
    [R[0][0], R[0][1], R[0][2], t[0]],
    [R[1][0], R[1][1], R[1][2], t[1]],
    [R[2][0], R[2][1], R[2][2], t[2]],
    [0, 0, 0, 1],
  ];
}

export function poseMatrix(input) {
  if (Array.isArray(input) && input.length === 4 && Array.isArray(input[0]) && input[0].length === 4) return input;
  return toPose(input).T;
}

export function toPose(input) {
  if (!input) throw new PoseError('missing pose');
  if (input.__pose) return input;
  let q, t;
  if (Array.isArray(input) && input.length === 4 && Array.isArray(input[0])) {
    const T = input;
    const R = T.slice(0, 3).map((row) => row.slice(0, 3));
    q = matrixToQuaternion(R);
    t = [T[0][3], T[1][3], T[2][3]];
  } else if (Array.isArray(input) && input.length === 7 && typeof input[0] === 'number') {
    q = input.slice(0, 4);
    t = input.slice(4, 7);
  } else if (input.q || input.quaternion) {
    q = input.q ?? input.quaternion;
    t = input.t ?? input.translation ?? input.position ?? [0, 0, 0];
  } else if (input.R || input.T || input.matrix) {
    const T = input.T ?? input.matrix ?? poseMatrixFromRt(input.R, input.t ?? [0, 0, 0]);
    const R = T.slice(0, 3).map((row) => row.slice(0, 3));
    q = matrixToQuaternion(R);
    t = [T[0][3], T[1][3], T[2][3]];
  } else if (Array.isArray(input) && input.length === 6) {
    const e = se3Exp(input);
    q = e.q; t = e.t;
  } else {
    throw new PoseError('unsupported pose representation', { input });
  }
  q = normalizeQuaternion(q);
  t = t.slice(0, 3).map(Number);
  const R = quaternionToMatrix(q);
  const T = poseMatrixFromRt(R, t);
  return { __pose: true, q, t, R, T };
}

// Adjoint for tangent order [omega_x, omega_y, omega_z, vx, vy, vz].
// Adj_T = [[R, 0], [t_hat R, R]].
export function adjoint(input) {
  const { R, t } = toPose(input);
  const tx = skew(t);
  const txR = matmul(tx, R);
  return [
    [...R[0], 0, 0, 0],
    [...R[1], 0, 0, 0],
    [...R[2], 0, 0, 0],
    [txR[0][0], txR[0][1], txR[0][2], ...R[0]],
    [txR[1][0], txR[1][1], txR[1][2], ...R[1]],
    [txR[2][0], txR[2][1], txR[2][2], ...R[2]],
  ];
}

export function inverseAdjoint(input) {
  return adjoint(invertPose(input));
}

export function invertPose(input) {
  const p = toPose(input);
  const q = quaternionInverse(p.q);
  const Rinv = transpose(p.R);
  const tInv = matvec(Rinv, p.t.map((v) => -v));
  return { q, t: tInv, R: Rinv, T: poseMatrixFromRt(Rinv, tInv) };
}

// Mathematical SE(3) product A * B (rightmost transform applied first).
export function multiplyPoses(a, b) {
  const A = toPose(a);
  const B = toPose(b);
  const q = quaternionMultiply(A.q, B.q);
  const t = A.t.map((v, i) => v + matvec(A.R, B.t)[i]);
  const R = quaternionToMatrix(q);
  return { q, t, R, T: poseMatrixFromRt(R, t) };
}

export function composePoses(...items) {
  const flat = items.flat();
  if (flat.length === 0) return toPose({ q: [1, 0, 0, 0], t: [0, 0, 0] });
  return flat.reduce((acc, p) => multiplyPoses(acc, p));
}

export const composePose = composePoses;
export const se3Compose = composePoses;
export const poseInverse = invertPose;
export const inversePose = invertPose;
export const Adj = adjoint;

export function relativePose(a, b) {
  return multiplyPoses(invertPose(a), b);
}

// Solve T_total = prefix * X * suffix for X.  Either side may be omitted.
export function solveIncrement(total, { prefix, suffix } = {}) {
  let x = toPose(total);
  if (prefix) x = multiplyPoses(invertPose(prefix), x);
  if (suffix) x = multiplyPoses(x, invertPose(suffix));
  return x;
}

export const modelInverse = solveIncrement;

export function normalizeSide(side, fallback = 'right') {
  const s = String(side ?? fallback).toLowerCase().replace(/-/g, '');
  if (s === 'left' || s === 'l') return 'left';
  if (s === 'right' || s === 'r') return 'right';
  throw new PoseError(`unknown perturbation side: ${side}`);
}

export function covarianceFromVariance(variance) {
  if (!variance) return null;
  if (Array.isArray(variance) && Array.isArray(variance[0])) return variance;
  if (variance.length === 6 && typeof variance[0] === 'number') {
    return variance.map((v, i) => [0, 1, 2, 3, 4, 5].map((j) => (i === j ? v : 0)));
  }
  throw new PoseError('unsupported covariance/variance', { variance });
}

// weights are often stored as diagonal sqrt-information.  By default treat
// them as residual multipliers, so precision is diag(weight^2).
export function covarianceFromWeights(weights, { kind = 'sqrt-information' } = {}) {
  const w = weights.slice(0, 6);
  if (kind === 'precision') return w.map((wi, i) => w.map((_, j) => (i === j ? 1 / wi : 0)));
  if (kind === 'variance-weights') return w.map((wi, i) => w.map((_, j) => (i === j ? 1 / wi : 0)));
  return w.map((wi, i) => w.map((_, j) => (i === j ? 1 / (wi * wi) : 0)));
}

export const quaternionLog = rotationLog;

export { matmul, matvec, transpose };

export const quatMultiply = quaternionMultiply;
export const quatExp = quaternionExp;
export const quatConjugate = quaternionConjugate;
export const so3Log = rotationLog;
export const logSO3 = rotationLog;
export const expSO3 = so3Exp;
export const logSE3 = se3Log;
export const expSE3 = se3Exp;
export const se3Adj = adjoint;
export const AdjSE3 = adjoint;
export const normalizePose = toPose;
