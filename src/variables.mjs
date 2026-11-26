import { createMatrix, matmul, transpose } from './linalg.mjs';

export const DEFAULT_CAMERAS = Object.freeze(['wide', 'main', 'zoom', 'ultrawide', 'telephoto', 'front']);

function isRevision(x) {
  return typeof x === 'string' && /^(?:rev|revision|r|v)?\d+$/i.test(x);
}

function isSourceId(x) {
  return typeof x === 'string' && /^[ab]\d+$/i.test(x);
}

// Physical extrinsic identity.  Input tuples in the sample are stored as
// [mount/body, camera, revision], while prose names the semantic triple as
// (camera, mount, rev).  Known camera names make both orderings unambiguous.
export function extrinsicIdentity(input, options = {}) {
  if (input == null) throw new Error('missing extrinsic variable');
  if (typeof input === 'string') {
    return { camera: null, mount: null, rev: null, key: input, source: input };
  }

  const knownCameras = new Set([...(options.cameras ?? []), ...DEFAULT_CAMERAS]);
  let camera = input.camera ?? input.cam ?? input.to ?? null;
  let mount = input.mount ?? input.body ?? input.base ?? input.b ?? null;
  let rev = input.rev ?? input.revision ?? input.version ?? null;
  let source = input.source ?? input.id ?? input.alias ?? input[0] ?? null;

  if (Array.isArray(input)) {
    const arr = input.slice();
    if (arr.length === 4) {
      // Sample aliased rows: [source, mount, camera, rev].
      [source, mount, camera, rev] = arr;
      if (knownCameras.has(mount) && !knownCameras.has(camera)) {
        // Explicit [source, camera, mount, rev] ordering.
        [mount, camera] = [camera, mount];
      }
    } else if (arr.length === 3) {
      rev = arr.find(isRevision) ?? arr[2];
      const cameraIndex = arr.findIndex((x) => knownCameras.has(x));
      if (cameraIndex >= 0) {
        camera = arr[cameraIndex];
        const others = arr.filter((_, i) => i !== cameraIndex && arr[i] !== rev);
        mount = others[0] ?? null;
      } else if (options.arrayOrder === 'camera,mount,rev') {
        [camera, mount, rev] = arr;
      } else if (options.arrayOrder === 'mount,camera,rev') {
        [mount, camera, rev] = arr;
      } else {
        // README's semantic order is (camera, mount, rev); sample's known
        // camera list normally resolves the reversed first two fields.
        [camera, mount, rev] = arr;
      }
      source = null;
    } else if (arr.length >= 3) {
      [camera, mount, rev] = arr;
    }
  }

  const key = [camera ?? '*', mount ?? '*', rev ?? '*'].join('|');
  const tuple = [camera, mount, rev];
  return { camera, mount, rev, tuple, semanticTuple: tuple, key, source, raw: input };
}

export function extrinsicKey(input, options = {}) {
  if (typeof input === 'string') return input;
  return extrinsicIdentity(input, options).key;
}

export function sameExtrinsic(a, b, options = {}) {
  return extrinsicKey(a, options) === extrinsicKey(b, options);
}

export const areAliased = sameExtrinsic;
export const sharedVariableKey = extrinsicKey;
export const keyForVariable = extrinsicKey;
export const variableKey = extrinsicKey;

// Build a linear alias/folding map E such that source coordinates x_source =
// E * x_physical.  R_physical = R_source * E.
export function aliasMap(variables, { blockSize = 6, cameras } = {}) {
  const ids = variables.map((v) => {
    if (typeof v === 'object' && v.key && !v.descriptor) return v;
    const descriptor = v.descriptor ?? v.variable ?? v;
    const id = descriptor?.key ? descriptor : extrinsicIdentity(descriptor, { cameras });
    return { ...v, key: v.key ?? id.key, identity: id };
  });
  const keys = [];
  const targetOffset = new Map();
  for (const v of ids) {
    if (!targetOffset.has(v.key)) {
      targetOffset.set(v.key, keys.length * blockSize);
      keys.push(v.key);
    }
  }
  const sourceDim = ids.length * blockSize;
  const targetDim = keys.length * blockSize;
  const E = createMatrix(sourceDim, targetDim);
  ids.forEach((v, i) => {
    const to = targetOffset.get(v.key);
    for (let k = 0; k < blockSize; k++) E[i * blockSize + k][to + k] = 1;
  });
  return { keys, E, targetOffset, sourceDim, targetDim, variables: ids };
}

export function foldResidual(R, b, variables, options = {}) {
  const { keys, E, targetDim } = aliasMap(variables, options);
  const Rphys = matmul(R, E);
  return {
    R: Rphys,
    b: b ? b.slice() : new Array(R.length).fill(0),
    keys,
    dimension: targetDim,
    map: E,
  };
}

export function foldInformation(H, rhs, variables, options = {}) {
  const { keys, E } = aliasMap(variables, options);
  const Hphys = matmul(transpose(E), matmul(H, E));
  const rhsPhys = rhs ? matmul(transpose(E), rhs) : null;
  return { H: Hphys, rhs: rhsPhys, keys, map: E };
}

// Add two 6x6 adjoint/Jacobian blocks belonging to the same physical key.
export function sumBlocks(blocks) {
  const out = createMatrix(6, 6);
  for (const B of blocks) {
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) out[i][j] += B[i][j];
  }
  return out;
}
