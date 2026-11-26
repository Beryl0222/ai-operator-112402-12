// Compact derived history.  This intentionally stores poses/Jacobians/
// gradients for touched segments; it is not a store for released raw packets.
export class SegmentHistory {
  constructor(options = {}) {
    this.radius = options.radius ?? 32;
    this.segments = [];
  }

  add(segment) {
    const record = {
      id: segment.id ?? `segment:${this.segments.length}`,
      start: segment.start ?? segment.frame ?? null,
      stop: segment.stop ?? (segment.frame != null ? segment.frame + 1 : null),
      frame: segment.frame ?? null,
      pose: segment.pose ?? null,
      q: segment.q ?? segment.pose?.q ?? null,
      t: segment.t ?? segment.pose?.t ?? null,
      adjoint: segment.adjoint ?? segment.cumulativeAdjoint ?? null,
      cumulativeAdjoint: segment.cumulativeAdjoint ?? segment.adjoint ?? null,
      jacobians: segment.jacobians ?? null,
      gradient: segment.gradient ?? null,
      parameterGradients: segment.parameterGradients ?? null,
      branch: segment.branch ?? segment.branchReport ?? null,
      batch: segment.batch ?? null,
      tangentSpace: segment.tangentSpace ?? null,
      rawFrames: segment.rawFrames ?? null,
      sourceObservationsReleased: segment.sourceObservationsReleased ?? false,
    };
    this.segments.push(record);
    return record;
  }

  get(id) {
    return this.segments.find((s) => s.id === id) ?? null;
  }

  near(frame, radius = this.radius) {
    return this.segments.filter((s) => {
      if (s.frame != null) return Math.abs(s.frame - frame) <= radius;
      if (s.start == null || s.stop == null) return false;
      return frame >= s.start - radius && frame < s.stop + radius;
    });
  }

  gradientsNear(frame, radius = this.radius) {
    return this.near(frame, radius).map((s) => ({
      id: s.id,
      start: s.start,
      stop: s.stop,
      gradient: s.gradient,
      parameterGradients: s.parameterGradients,
      jacobians: s.jacobians,
    }));
  }

  markRawReleased(frame) {
    for (const s of this.segments) {
      if (s.rawFrames?.some((f) => f <= frame)) s.sourceObservationsReleased = true;
    }
    return this.segments;
  }
}
