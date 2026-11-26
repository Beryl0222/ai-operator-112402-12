import { rotationLog, toPose } from './pose.mjs';

export class AnchorLedger {
  constructor({ stableAxisTol = 1e-5 } = {}) {
    this.anchors = new Map();
    this.stableAxisTol = stableAxisTol;
  }

  place(input) {
    const id = input.id ?? input.anchor;
    if (!id) throw new Error('anchor id required');
    if (this.anchors.has(id) && input.immutable !== false) {
      const existing = this.anchors.get(id);
      if (input.frame != null && existing.frame !== input.frame) {
        throw new Error(`anchor ${id} was already delivered at frame ${existing.frame}`);
      }
      return existing;
    }
    const pose = input.pose ? toPose(input.pose) : null;
    const q = input.q ?? pose?.q;
    const branch = input.branch ?? (q ? rotationLog(q, { stableAxisTol: this.stableAxisTol, branch: input.branchIndex }) : null);
    const positionDelivered = input.positionDelivered ?? (Number.isFinite(pose?.t?.[0]) || Number.isFinite(input.t?.[0]));
    const rotationDirectionStable = branch?.stable ?? true;
    const record = {
      id,
      frame: input.frame ?? null,
      status: 'placed-original',
      delivered: true,
      placedAsOriginalResult: true,
      pose: pose ? { q: pose.q.slice(), t: pose.t.slice() } : { q: q?.slice?.() ?? null, t: (input.t ?? pose?.t ?? null)?.slice?.() ?? null },
      tangentSpace: {
        convention: input.perturbation ?? input.tangentConvention ?? input.side ?? 'right',
        order: ['wx', 'wy', 'wz', 'vx', 'vy', 'vz'],
        coordinateFrame: input.coordinateFrame ?? input.tangentFrame ?? null,
      },
      perturbation: input.perturbation ?? input.tangentConvention ?? input.side ?? 'right',
      side: input.perturbation ?? input.tangentConvention ?? input.side ?? 'right',
      branchIndex: input.branchIndex ?? branch?.branch ?? 0,
      batchId: input.batch ?? null,
      branch: branch ? serializeBranch(branch) : null,
      batch: input.batch ?? null,
      positionDelivered,
      rotationDelivered: input.rotationDelivered ?? true,
      rotationDirectionStable,
      continuousDifferentiable: input.continuousDifferentiable ?? rotationDirectionStable,
      batchContinuity: input.batchContinuity ?? null,
      delivery: {
        position: positionDelivered ? 'delivered-original' : 'unavailable',
        rotation: branch?.ambiguous ? 'principal-pi-ambiguous' : 'delivered-original',
        rotationDirectionStable,
        adjacentBranchSmoothed: false,
      },
      source: input.source ?? 'posePacket',
      createdAt: input.now ?? null,
    };
    this.anchors.set(id, record);
    return record;
  }

  get(id) {
    return this.anchors.get(id) ?? null;
  }

  all() {
    return [...this.anchors.values()];
  }

  originalPlacements() {
    return this.all().filter((a) => a.placedAsOriginalResult);
  }
}

export function serializeBranch(branch) {
  if (!branch) return null;
  return {
    vector: branch.vector?.slice?.() ?? null,
    angle: branch.angle,
    axis: branch.axis?.slice?.() ?? null,
    branch: branch.branch ?? 0,
    principalBranch: branch.principalBranch ?? 0,
    nearPi: branch.nearPi ?? false,
    ambiguous: branch.ambiguous ?? false,
    stable: branch.stable ?? true,
    piDistance: branch.piDistance ?? null,
    stableAxisTol: branch.stableAxisTol ?? null,
    canonicalSign: branch.canonicalSign ?? null,
    continuouslyDifferentiable: branch.continuouslyDifferentiable ?? branch.stable ?? true,
  };
}
