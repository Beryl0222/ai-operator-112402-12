import {readFileSync} from 'node:fs';
import * as input from './inputs/pose-observations.mjs';

const source=JSON.parse(readFileSync(new URL('./inputs/pose-observations.json',import.meta.url),'utf8'));
console.log(JSON.stringify({名称:'位姿轨迹和窗口因子',首项:source['frames'],字段:Object.keys(source),取数函数:Object.keys(input)},null,2));
