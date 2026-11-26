# 手机多镜头手持定位的 SE(3) 批量伴随与边缘化

这里的位姿轨迹和窗口因子是用于设备联调的合成资料。使用 Node.js 22。运行 `npm ci --offline` 核对依赖锁，`npm run inspect` 查看资料字段，`npm test` 检查资料格式及相互引用。检视脚本只读取元数据。

## 数据格式

位姿列向量约定 T_ab 从 b 到 a，乘法先右后左。四元数 [w,x,y,z]，SO(3) log 的范数在 [0,π]，π 的歧义依据 stableAxisTol 报告。扰动坐标顺序 [ωx,ωy,ωz,vx,vy,vz]，左右扰动分别在 T 前后复合。组合的源变量可以同名，只有 (camera,mount,rev) 相同才共享。先验以平方残差 R·δ−b 信息形式和页内明确的变量坐标为准。边缘化语义是最小化待退休变量后保留参数上的同一二次形式，允许秩亏、不可精确表示时报告受损方向，误差用 profile；非线性换版与重线性化需要对应观测和外参，源观测退休后不能伪称新精确先验。 位姿协方差和输出伴随在包选定扰动约定内解释，换基时验同一二次型。窗口上限计一次可协作的最近窗口，不能用第二个向量副本隐式翻倍。maxWindowReads 对已经得到过当前版本局部校准后，只改某一帧的重复局部请求按取观测页次数计，每页 32 帧，批量取多个页逐页计。

## 取数

`inputs/pose-observations.json` 保存标定、规格和调用记录。记录中的范围除明确说明外从零开始，使用半开区间。批量数组按调用记录列出的顺序交付。posePacket(frame) 给位姿包，poseFactors(start,stop) 给半开帧范围的测量边。

## Node.js API

核心实现不引入第三方依赖，入口为 `index.mjs`（也可从 `pose.mjs`、`calibration.mjs`、`window.mjs`、`track.mjs`、`model.mjs`、`session.mjs` 等语义入口导入）。

```js
import {
  composeTransforms, trackTransforms, composeDeltas,
  marginalizePrior, changePriorBasis, createSession,
} from './index.mjs';

// 每个 step 的 perturbation/side 可为 'left' 或 'right'；
// 默认按数学乘法 M1*M2... 合成，trackTransforms 按 start -> end 的行走顺序合成。
const composed = composeTransforms(steps, { outputPerturbation: 'right' });
composed.cumulativeAdjoint;      // Adj(T)
composed.jacobians[i].jacobian; // {left: 6x6, right: 6x6}
composed.covariance;            // 已按各步左右约定传播
composed.backward(gradient);    // 模型反向 VJP，并按共同 key 汇总变量梯度

// 转动向量不能逐元素相加；composeRotationVectors 走 Exp/四元数乘积并保留主分支报告。
// Schur 边缘化保留 H_cc 中所有公共外参交叉块；秩亏且零空间仍有耦合时
// profile.exact=false、profile.damagedDirections 给出受损方向。
const prior = marginalizePrior({ H, rhs, variables, retire: ['pose:160'] });
const moved = prior.changeBasis({ J, offset });
moved.profile.basisVerification.ok; // 同一二次型校验

// 设备联调：按 inputs 的 events 跑完整流程。
const session = createSession();
const { report } = session.runEvents();
// report.anchorsPlacedAsOriginal
// report.calibrationsMigrated
// report.needsReobservation
// report.windowReads.used / records
```

运行 `npm run solve` 输出样例事件的完整结果与报告；`npm test` 除资料契约外还会校验混合扰动 Jacobian、半圈分支、共享外参、Schur 交叉块、换基二次型和窗口/原始坐标释放语义。

设备窗口每页 32 帧；`WindowReadMeter` 对一次请求触及的不同页逐页计数，批量跨页累加。`PoseWindow.releaseRaw(frame)` 后再读取该原始帧会抛出带 `code='RAW_COORDINATES_RELEASED'` 的错误；已交付锚点快照、累计伴随和梯度归档仍可由 `session.history(anchorId)` 检索，但不会伪称重新取得了已释放的原始坐标。
