// Small dense linear-algebra helpers used by the pose/calibration code.
// The matrices here are deliberately plain arrays so results are easy to
// inspect and serialize in device reports.

export function createMatrix(rows, cols, value = 0) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => value));
}

export function zeros(n) {
  if (typeof n === 'number') return createMatrix(n, n, 0);
  return createMatrix(n[0], n[1], 0);
}

export function eye(n) {
  const a = createMatrix(n, n, 0);
  for (let i = 0; i < n; i++) a[i][i] = 1;
  return a;
}

export function isMatrix(a) {
  return Array.isArray(a) && Array.isArray(a[0]);
}

export function dimensions(a) {
  return [a.length, a[0]?.length ?? 0];
}

export function cloneMatrix(a) {
  return a.map((row) => row.slice());
}

export function transpose(a) {
  const [r, c] = dimensions(a);
  const out = createMatrix(c, r);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) out[j][i] = a[i][j];
  }
  return out;
}

export function add(a, b) {
  const [r, c] = dimensions(a);
  return Array.from({ length: r }, (_, i) =>
    Array.from({ length: c }, (_, j) => a[i][j] + b[i][j]),
  );
}

export function sub(a, b) {
  const [r, c] = dimensions(a);
  return Array.from({ length: r }, (_, i) =>
    Array.from({ length: c }, (_, j) => a[i][j] - b[i][j]),
  );
}

export function scale(a, s) {
  return a.map((row) => row.map((v) => v * s));
}

export function matmul(a, b) {
  if (!isMatrix(b)) return matvec(a, b);
  const [ra, ca] = dimensions(a);
  const [rb, cb] = dimensions(b);
  if (ca !== rb) throw new Error(`matrix shape mismatch: ${ra}x${ca} * ${rb}x${cb}`);
  const out = createMatrix(ra, cb);
  for (let i = 0; i < ra; i++) {
    for (let k = 0; k < ca; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < cb; j++) out[i][j] += aik * b[k][j];
    }
  }
  return out;
}

export function matvec(a, x) {
  const [r, c] = dimensions(a);
  if (x.length !== c) throw new Error(`matrix/vector shape mismatch: ${r}x${c} * ${x.length}`);
  const out = new Array(r).fill(0);
  for (let i = 0; i < r; i++) {
    let s = 0;
    for (let j = 0; j < c; j++) s += a[i][j] * x[j];
    out[i] = s;
  }
  return out;
}

export function mul(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return a * b;
  if (isMatrix(a)) return matmul(a, b);
  if (isMatrix(b)) return transpose(matmul(transpose(b), transpose([a])))[0];
  return a * b;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function skew(v) {
  const [x, y, z] = v;
  return [
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0],
  ];
}

export function vee(m) {
  return [m[2][1], m[0][2], m[1][0]];
}

export function block(a, rows, cols) {
  return rows.map((i) => cols.map((j) => a[i][j]));
}

export function setBlock(out, rows, cols, b) {
  rows.forEach((i, bi) => cols.forEach((j, bj) => {
    out[i][j] = b[bi][bj];
  }));
  return out;
}

// Symmetric eigen-decomposition using cyclic Jacobi rotations.  The pose
// problems in this repository are at most a few dozen coordinates, where a
// dependency-free dense implementation is preferable to a native LAPACK addon.
export function symmetricEigen(input, { maxSweeps = 100, tolerance = 1e-15 } = {}) {
  const n = input.length;
  const a = cloneMatrix(input);
  const v = eye(n);
  let maxOff = 0;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    maxOff = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) maxOff = Math.max(maxOff, Math.abs(a[p][q]));
    }
    const scale = Math.max(1, ...a.flat().map(Math.abs));
    if (maxOff <= tolerance * scale * n) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) <= tolerance * scale) continue;
        const app = a[p][p];
        const aqq = a[q][q];
        const tau = (aqq - app) / (2 * apq);
        let t;
        if (tau === 0) t = 1;
        else t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        a[p][q] = 0;
        a[q][p] = 0;

        for (let j = 0; j < n; j++) {
          if (j === p || j === q) continue;
          const ajp = a[j][p];
          const ajq = a[j][q];
          a[j][p] = c * ajp - s * ajq;
          a[p][j] = a[j][p];
          a[j][q] = s * ajp + c * ajq;
          a[q][j] = a[j][q];
        }

        for (let j = 0; j < n; j++) {
          const vjp = v[j][p];
          const vjq = v[j][q];
          v[j][p] = c * vjp - s * vjq;
          v[j][q] = s * vjp + c * vjq;
        }
      }
    }
  }

  const values = a.map((row, i) => row[i]);
  // Sort descending while keeping eigenvectors aligned.
  const order = values.map((_, i) => i).sort((i, j) => values[j] - values[i]);
  return {
    values: order.map((i) => values[i]),
    vectors: order.map((i) => v.map((row) => row[i])), // columns as arrays
    iterations: maxSweeps,
    residual: maxOff,
  };
}

export function pseudoinverseSym(a, options = {}) {
  const n = a.length;
  if (n === 0) return { value: [], rank: 0, nullBasis: [], tolerance: 0, eigenvalues: [] };
  const eig = symmetricEigen(a, options);
  const maxAbs = Math.max(...eig.values.map(Math.abs), 1);
  const tol = options.rankTolerance ?? options.tolerance ?? 1e-10 * n * maxAbs;
  const inv = createMatrix(n, n, 0);
  const nullBasis = [];
  let rank = 0;

  for (let k = 0; k < n; k++) {
    const lambda = eig.values[k];
    const col = eig.vectors[k];
    if (Math.abs(lambda) > tol) {
      rank++;
      const factor = 1 / lambda;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) inv[i][j] += factor * col[i] * col[j];
      }
    } else {
      nullBasis.push(col.slice());
    }
  }

  return {
    value: inv,
    rank,
    nullity: n - rank,
    nullBasis,
    tolerance: tol,
    eigenvalues: eig.values,
    eigenvectors: eig.vectors,
  };
}

export function solveSym(a, b, options = {}) {
  const pin = pseudoinverseSym(a, options);
  return { x: matvec(pin.value, b), ...pin };
}

export function inverseMatrix(a) {
  const n = a.length;
  const m = a.map((row, i) => [...row, ...eye(n)[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    if (Math.abs(m[pivot][col]) < 1e-14) {
      const err = new Error('matrix is singular');
      err.profile = { col, pivotValue: m[pivot][col] };
      throw err;
    }
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const d = m[col][col];
    for (let j = 0; j < 2 * n; j++) m[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = m[r][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) m[r][j] -= factor * m[col][j];
    }
  }
  return m.map((row) => row.slice(n));
}

export function maxAbsDifference(a, b) {
  let m = 0;
  const walk = (x, y) => {
    if (Array.isArray(x)) {
      for (let i = 0; i < x.length; i++) walk(x[i], y[i]);
    } else {
      m = Math.max(m, Math.abs(x - y));
    }
  };
  walk(a, b);
  return m;
}

// Deterministic Box-Muller-ish samples for checking that a basis change really
// preserves x^T H x - rhs^T x rather than only copying metadata.
export function seededNormal(seed = 23) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    const u = (s + 1) / 4294967297;
    s = (1664525 * s + 1013904223) >>> 0;
    const v = (s + 1) / 4294967297;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}
