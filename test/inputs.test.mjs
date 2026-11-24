import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import { posePacket, poseFactors } from '../inputs/pose-observations.mjs';

const data=JSON.parse(readFileSync(new URL('../inputs/pose-observations.json',import.meta.url),'utf8'));
test('位姿轨迹和窗口因子的格式与关联', () => {
  assert.equal(posePacket(177).q.length,4);
  assert.ok(Math.abs(posePacket(178).q[0])<data.stableAxisTol);
  assert.ok(posePacket(177).q[0]*posePacket(179).q[0]<0);
  assert.deepEqual(data.aliased[0].slice(1),data.aliased[1].slice(1));
  assert.ok(data.priorRows.every(row=>row.length===12));
  assert.ok(data.window<data.frames/10);
});
