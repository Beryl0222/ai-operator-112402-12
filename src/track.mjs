import { composeTransforms } from './compose.mjs';

export class FrameMismatchError extends Error {
  constructor(message, profile) {
    super(message);
    this.name = 'FrameMismatchError';
    this.code = 'FRAME_CHAIN_MISMATCH';
    this.profile = profile;
  }
}

// Ordered tracking chain.  Steps follow the direction start -> end.  A step
// {from,to,pose:T_to_from} maps coordinates named `from` into `to`.
export class PoseTrack {
  constructor(options = {}) {
    this.start = options.start ?? options.startFrame ?? options.from ?? null;
    this.end = options.end ?? options.endFrame ?? options.to ?? null;
    this.startCoordinate = options.startCoordinate ?? this.start;
    this.endCoordinate = options.endCoordinate ?? this.end;
    this.options = options;
    this.steps = [];
  }

  add(step) {
    const normalized = typeof step === 'object' && step.pose ? step : { pose: step };
    if (this.steps.length === 0) {
      if (this.start == null) this.start = normalized.from ?? null;
      if (normalized.from != null && this.start != null && !sameFrame(normalized.from, this.start)) {
        throw new FrameMismatchError('first transform does not match track start frame', {
          expected: this.start, got: normalized.from,
        });
      }
    } else {
      const prev = this.steps.at(-1);
      if (normalized.from != null && prev.to != null && !sameFrame(normalized.from, prev.to)) {
        throw new FrameMismatchError('ordered transforms do not chain', {
          previousTo: prev.to, nextFrom: normalized.from, index: this.steps.length,
        });
      }
    }
    this.steps.push(normalized);
    if (normalized.to != null) this.end = normalized.to;
    return this;
  }

  compose(options = {}) {
    const result = composeTransforms(this.steps, {
      ...this.options,
      ...options,
      order: 'travel',
      startFrame: this.startCoordinate ?? this.start,
      endFrame: this.endCoordinate ?? this.end,
    });
    if (this.endCoordinate == null) this.endCoordinate = this.end;
    result.startCoordinate = this.startCoordinate ?? this.start;
    result.endCoordinate = this.endCoordinate ?? this.end;
    result.startCoordinateSystem = result.startCoordinate;
    result.endCoordinateSystem = result.endCoordinate;
    return result;
  }

  backward(gradient, options = {}) {
    return this.compose(options).backward(gradient, options);
  }
}

function sameFrame(a, b) {
  if (a === b) return true;
  return String(a) === String(b);
}

export function trackTransforms(stepsOrOptions, maybeOptions = {}) {
  const options = Array.isArray(stepsOrOptions) ? { ...maybeOptions, steps: stepsOrOptions } : stepsOrOptions;
  const track = new PoseTrack(options);
  for (const step of options.steps ?? options.transforms ?? []) track.add(step);
  return track.compose(options.compose ?? {});
}

export const createPoseTrack = (options) => new PoseTrack(options);
