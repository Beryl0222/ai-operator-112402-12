import {readFileSync} from 'node:fs';
const data = JSON.parse(readFileSync(new URL('./pose-observations.json', import.meta.url), 'utf8'));
export function posePacket(frame) {
  const theta=frame>=177&&frame<=179?Math.PI+(frame-178)*1e-4:0.015*Math.sin(frame*.528);
  const half=theta/2, kind=frame%data.cameras.length;
  return {frame,q:[Math.cos(half),0,0,Math.sin(half)], t:[frame*.001,Math.sin(frame/54)*.02,Math.cos(frame/62)*.1],
    variable:data.mounts[kind],from:'object',to:data.mounts[kind][1],
    variance:[.002,.002,.003,.0005,.001,.0005], perturbation:frame%2 ? 'left':'right'};
}
export function poseFactors(start, stop) {
  return Array.from({length:stop-start},(_,i)=>({from:start+i,to:start+i+1,
    weights:[1,1,2,1,1,.5], side:i%2?'left':'right', external:data.mounts[i%2]}));
}
