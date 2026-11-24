import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const d = JSON.parse(readFileSync(new URL('../inputs/session.json', import.meta.url), 'utf8'));

test("相同时间点仍有明确的平滑边", () => {
  assert.ok(d.edges.length+1===d.points.length && d.points[2].t===d.points[3].t && d.edges[2][1]);
});

test("固定零锚和可撤锚分别存在", () => {
  assert.ok(d.anchors.some(a=>a.locked&&a.value===0) && d.anchors.some(a=>!a.locked));
});

test("有与锚隔离的不可定区间", () => {
  assert.ok(d.points.slice(6).every(p=>p.w===0) && !d.edges[5][1] && d.edges[6][1]);
});
