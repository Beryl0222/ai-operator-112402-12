export class WindowBudgetError extends Error {
  constructor(message, profile) {
    super(`[WINDOW_BUDGET_EXCEEDED] ${message}`);
    this.name = 'WindowBudgetError';
    this.code = 'WINDOW_BUDGET_EXCEEDED';
    this.profile = profile;
  }
}

export class RawCoordinatesReleasedError extends Error {
  constructor(message, profile) {
    super(`[RAW_COORDINATES_RELEASED] ${message}`);
    this.name = 'RawCoordinatesReleasedError';
    this.code = 'RAW_COORDINATES_RELEASED';
    this.profile = profile;
  }
}

export function pageForFrame(frame, pageSize = 32) {
  return Math.floor(frame / pageSize);
}

// Half-open [start, stop).
export function pagesForRange(start, stop, pageSize = 32) {
  if (!Number.isInteger(start) || !Number.isInteger(stop) || stop < start) {
    throw new Error(`invalid half-open frame range [${start},${stop})`);
  }
  if (start === stop) return [];
  const first = pageForFrame(start, pageSize);
  const last = pageForFrame(stop - 1, pageSize);
  return Array.from({ length: last - first + 1 }, (_, i) => first + i);
}

export function pagesForFrames(frames, pageSize = 32) {
  return [...new Set(frames.map((f) => pageForFrame(f, pageSize)))].sort((a, b) => a - b);
}

// Each request is billed by the observation pages it touches; a batch spanning
// pages bills one read per page.  Repeated requests are intentionally summed.
export function countWindowReads(requests = [], { pageSize = 32 } = {}) {
  let total = 0;
  const details = [];
  for (const req of requests) {
    const pages = req.pages ?? (req.frames
      ? pagesForFrames(req.frames, pageSize)
      : pagesForRange(req.start, req.stop, pageSize));
    total += pages.length;
    details.push({ ...req, pages, pageCount: pages.length });
  }
  return { total, reads: total, details };
}

export class WindowReadMeter {
  constructor({ pageSize = 32, maxWindowReads = 13 } = {}) {
    this.pageSize = pageSize;
    this.maxWindowReads = maxWindowReads;
    this.used = 0;
    this.records = [];
  }

  pagesFor({ start, stop, frames } = {}) {
    if (frames) return pagesForFrames(frames, this.pageSize);
    if (start != null && stop != null) return pagesForRange(start, stop, this.pageSize);
    throw new Error('read meter requires frames or start/stop');
  }

  charge(request = {}) {
    const pages = this.pagesFor(request);
    const next = this.used + pages.length;
    const record = {
      reason: request.reason ?? null,
      start: request.start ?? null,
      stop: request.stop ?? null,
      frames: request.frames ? request.frames.slice() : null,
      pages,
      pageCount: pages.length,
      usedAfter: next,
      at: request.at ?? null,
    };
    if (this.maxWindowReads != null && next > this.maxWindowReads) {
      record.allowed = false;
      this.records.push(record);
      throw new WindowBudgetError(`maxWindowReads=${this.maxWindowReads} exceeded`, {
        used: this.used,
        requested: pages.length,
        next,
        pages,
        record,
      });
    }
    this.used = next;
    record.allowed = true;
    this.records.push(record);
    return record;
  }

  read(request = {}) { return this.charge(request); }
  requestPage(request = {}) { return this.charge(request); }
  reset() { this.used = 0; this.records = []; return this; }

  toJSON() {
    return {
      pageSize: this.pageSize,
      maxWindowReads: this.maxWindowReads,
      used: this.used,
      remaining: this.maxWindowReads == null ? null : this.maxWindowReads - this.used,
      records: this.records,
    };
  }
}

// A single finite device pose window.  It intentionally stores raw packets
// separately from compact derived archives; releasing raw coordinates never
// deletes an already delivered anchor snapshot.
export class PoseWindow {
  constructor({ window = 96, frames = Number.POSITIVE_INFINITY, meter } = {}) {
    this.capacity = window;
    this.frameCount = frames;
    this.meter = meter ?? new WindowReadMeter();
    this.raw = new Map();
    this.accessOrder = [];
    this.rawWatermark = -1;
    this.releasedFrames = new Set();
  }

  touch(frame) {
    this.accessOrder = this.accessOrder.filter((f) => f !== frame);
    this.accessOrder.push(frame);
    while (this.raw.size > this.capacity) {
      const oldest = this.accessOrder.shift();
      if (oldest == null) break;
      this.raw.delete(oldest);
    }
  }

  remember(frame, value) {
    if (frame <= this.rawWatermark) {
      this.releasedFrames.add(frame);
      return false;
    }
    this.raw.set(frame, value);
    this.touch(frame);
    return true;
  }

  isRawReleased(frame) {
    return this.releasedFrames.has(frame) || frame <= this.rawWatermark;
  }

  getRaw(frame) {
    if (this.isRawReleased(frame)) {
      throw new RawCoordinatesReleasedError(`raw coordinates for frame ${frame} were released`, {
        frame,
        rawWatermark: this.rawWatermark,
        released: [...this.releasedFrames],
      });
    }
    const value = this.raw.get(frame);
    if (value) this.touch(frame);
    return value ?? null;
  }

  releaseRaw(frame) {
    this.rawWatermark = Math.max(this.rawWatermark, frame);
    for (const f of [...this.raw.keys()]) {
      if (f <= frame) {
        this.raw.delete(f);
        this.releasedFrames.add(f);
      }
    }
    return { frame, rawWatermark: this.rawWatermark, retained: this.raw.size };
  }

  retainedFrames() {
    return [...this.raw.keys()].sort((a, b) => a - b);
  }
}

// One nearest collaborative window, never two implicitly copied windows.
export function nearestWindow(target, size = 96, totalFrames = Number.POSITIVE_INFINITY) {
  let center;
  let span = 0;
  if (Array.isArray(target)) {
    const [start, stop] = target;
    center = (start + stop) / 2;
    span = stop - start;
  } else if (target && typeof target === 'object') {
    center = ((target.start ?? target.frame) + (target.stop ?? (target.frame + 1))) / 2;
    span = (target.stop ?? target.frame + 1) - (target.start ?? target.frame);
  } else {
    center = target + 0.5;
  }
  if (span > size) return Array.isArray(target) ? target.slice() : [Math.floor(center), Math.floor(center) + size];
  let start = Math.round(center - size / 2);
  if (Number.isFinite(totalFrames)) start = Math.max(0, Math.min(start, totalFrames - size));
  else start = Math.max(0, start);
  return [start, start + size];
}
