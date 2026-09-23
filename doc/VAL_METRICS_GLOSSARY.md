# 验证指标词典(epoch-end val 输出逐项解释)

本文解释 METEOR `train.py` **每个 epoch 结束时**打印的验证指标(以及 `closed_loop_eval` 等
评测输出的同名项),用于读数时避免误判。所有定义均来自当前训练脚本的实现。

> 指标由 `train.py` 的 9 个 `evaluate_*` 函数产出,`[valX ep{n}]` 行逐条打印。
> 默认 val 只取每场景前 8 帧(`max_per_scene=8`);加 **`--val-full`** 后评估**整个 val 集合**
> (本项目 2,967 帧),样本数 `n` 会大 2~3 个数量级,指标的统计意义随之成立。

---

## 0. 一行输出的结构

```
[val3D ep2] veh P=0.82 R=0.17 R50=0.18 Rn=0.31 err=0.67m L/W=0.32/0.08m corner=0.79m yaw=5.2deg flip=0.01 | vru P=0.72 R=0.04 ...
  └ 标签      └ 类别  └ 该项指标(键=值)                    └ 第二个类别
```

- **`ep{n}`** = 第几个 epoch;
- **`n=...`** 出现在各行的末尾或 `stat` 行内,是**该指标的样本数**——**读数的第一件事**:
  `n=2` 的 `R=0.00` 与 `n=3458` 的 `R=0.17` 完全不是一回事。

---

## 1. 通用缩写

| 缩写 | 全称 | 方向 |
|---|---|---|
| **BEV** | Bird's-Eye View | — |
| **mIoU** | mean Intersection over Union | ↑ 越大越好(0–1) |
| **P / R** | Precision / Recall | ↑ |
| **n** | 样本计数 | 判断可信度 |
| **ADE** | Average Displacement Error,各航点误差均值(m) | ↓ |
| **FDE** | Final Displacement Error,末航点误差(m) | ↓ |
| **MAE** | Mean Absolute Error | ↓ |

---

## 2. BEV seg(鸟瞰图 9 类语义分割)

```
[val ep2] mIoU=0.158 road=0.476 sidewalk=0.231 laneline=0.204 ... | road P=0.352 R=0.570 | redge P=0.008 R=0.061 x8.18
```

| 项 | 含义 |
|---|---|
| `mIoU` | 9 类平均 IoU(`NaN` 的类不参与平均) |
| `road / sidewalk / crosswalk / laneline / stopline / road_edge / marking / parking` | **各类别 IoU**;`nan` = 该类别在本 val 切片里没有 GT |
| `road P / R`、`redge P / R` | 选定类别的精确率/召回率 |
| **`x8.18`** | **预测面积 ÷ GT 面积**(`class_pr` 的 pred/GT area ratio)。`x8.18` = 该类被**过预测到 8 倍**,这解释了为何 `road_edge` IoU 只有 0.026 |

---

## 3. 2D seg(6 相机图像 21 类语义分割)

```
[val2d ep2] mIoU=0.479 bg=0.345 mark=0.315 road=0.912 swalk=0.503 lane=0.599 pole=0.614
```

| 项 | 含义 |
|---|---|
| `mIoU` | 21 类平均 IoU |
| `bg / mark / road / swalk / lane / pole` | 背景 / 路面标线 / 道路 / 人行道 / 车道线 / 杆状物 的**类别 IoU** |

---

## 4. 3D det(3D 检测框)

```
[val3D ep2] veh P=0.82 R=0.17 R50=0.18 Rn=0.31 err=0.67m L/W=0.32/0.08m corner=0.79m yaw=5.2deg flip=0.01 | vru P=0.72 R=0.04 ...
[val3D-zone ep2] F0-40:R=0.17/e=0.62/n=3458/corner=0.74/L=0.34/W=0.07/yaw=5.3 F40-80:R=0.00/... R0-40:R=0.23/...
```

| 缩写 | 全称 / 定义 | 方向 |
|---|---|---|
| **veh / vru** | vehicle / vulnerable road user(行人+骑行者) | — |
| **P** | Precision(全场景,中心匹配 < 2 m) | ↑ |
| **R** | Recall(全场景) | ↑ |
| **R50** | **半径 50 m 内**的召回率(`x²+y² < 50²`) | ↑ |
| **Rn** | **自车走廊**召回率(30 m 内且横向 `\|y\| < 12 m`) | ↑ |
| **err** | 中心位置误差(m) | ↓ |
| **L / W** | 长 / 宽误差(m) | ↓ |
| **corner** | 四角点平均误差(m) | ↓ |
| **yaw** | 朝向角误差(度) | ↓ |
| **flip** | **朝向翻转率**(误差 > 90° 的占比;车辆"掉头"误判) | ↓ |
| **F0-40 / F40-80** | **Front 前向** 0–40 m / 40–80 m 区间 | — |
| **R0-40 / R40-80** | **Rear 后向** 0–40 m / 40–80 m 区间 | — |

> **读法示例**:`R=0.17`(全场景)但 `Rn=0.31`(走廊内)→ 越靠近自车召回越高,说明主要短板在远距。
> 上表 3D 的匹配口径是"中心距 < 2 m",与 COCO 的 IoU 阈值无关。

---

## 5. Stat(静止标志,动/静二分类)

```
[valStat ep2]       P=0.64 R=0.69 acc=0.75 movAcc=0.78 n=1682
[valStat-zone ep2]  F0-40:P=0.60/R=0.64/movAcc=0.66/n=867/statFrac=0.44 ...
[valStat-cal ep2]   balanced:t=0.037/P=0.60/R=0.90/acc=0.75/movAcc=0.66/bal=0.78 safe95:t=0.862/...
```

| 缩写 | 含义 |
|---|---|
| **acc** | **准确率**(此处是准确率,不是加速度误!) |
| **movAcc** | **运动样本上**的准确率(比总 acc 更严苛) |
| **statFrac** | 该区间内**静止样本占比**(看类别是否失衡) |
| **cal** 行 | 阈值标定:**balanced:t=…** 是用"平衡点"阈值时的表现;**safe95:t=…** 是"95% 安全"(高精度优先)阈值下的表现;`bal` = 两类准确率的平衡值 |

---

## 6. Traj(他车轨迹预测)

```
[valTraj ep2] agentADE=3.05m agentFDE=4.93m statAcc=0.54 vehADE=2.57 vruADE=7.81 vehHead=23deg vruHead=29deg
```

| 缩写 | 含义 | 方向 |
|---|---|---|
| **agentADE / agentFDE** | 全体目标的平均 / 终点位移误差(m) | ↓ |
| **vehADE / vruADE** | 车辆 / VRU 分组的 ADE | ↓ |
| **vehHead / vruHead** | **航向角误差**(度)——预测方向是否对 | ↓ |
| **statAcc** | 静止标志准确率 | ↑ |

---

## 7. TL(交通灯状态)

```
[valTL ep2] acc=0.98 none=0.99 green=0.00 yellow=0.00 red=0.55
```

| 项 | 含义 |
|---|---|
| `acc` | 总准确率(**准确率**) |
| `none / green / yellow / red` | **各类召回率**。`green=0.00` 多为该类样本极少;`red=0.55` 表示红灯漏检近一半 |

---

## 8. E2E(自车端到端规划)

```
[valE2E ep2]   ADE=1.11m ADEc=1.57m FDE=1.81m steer=0.040rad acc=0.22m/s2 brakeAcc=0.93
[valE2Ed ep2]  oracle=0.85 (selection loss +0.26) moving=1.16 stopped=0.65 const-vel-straight=0.79 hs-wp0-bias=+0.426m (n=2191)
[valHS ep2]    high-speed wp0 bias=+0.430m ADE=1.154 (n=240)
```

| 缩写 | 含义 | 方向 |
|---|---|---|
| **ADE / FDE** | 规划轨迹平均 / 终点位移误差(m) | ↓ |
| **ADEc** | **弯道子集 ADE**(`c`=curve,即 `\|lat@3s\| > 2 m`);全直行切片会输出 `nan` | ↓ |
| **steer** | 转向角误差(rad) | ↓ |
| **acc** | ⚠️ **加速度误差(m/s²)——此处不是准确率** | ↓ |
| **brakeAcc** | 刹车状态准确率 | ↑ |
| **oracle** | K=3 多模态中**最佳模式**的 ADE(理想选择下的下限) | ↓ |
| **selection loss** | **ADE − oracle**:全部来自"从 3 个模态里选错"的额外误差 | ↓ |
| **moving / stopped** | 运动 / 静止样本子集上的 ADE | ↓ |
| **const-vel-straight** | **匀速直行基线**的 ADE(模型必须显著打败它才有价值) | ↓(参照系) |
| **hs-wp0-bias** | **高速(≥8 m/s)区间第 1 个航点(0.5 s)的纵向偏差**(m),是"链式发散"的代理量 | 越接近 0 越好 |
| **HS** | **high-speed slice**,只统计 `v0 ≥ 8 m/s` 的帧 | — |

> **`oracle` 与 `selection loss` 的用法**:若 `oracle` 明显小于 `ADE`,说明回归能力够、短板在
> **模式选择**(应调 WTA/选择损失);若两者都大,才是轨迹回归本身不行。

---

## 9. 五个最容易误读的点

1. **`acc` 是双关**:`Stat` / `TL` 行表示**准确率**;`E2E` 行表示**加速度误差(m/s²)**;
2. **`R` ≠ `R50` ≠ `Rn`**:全场景 → 50 m 内 → 自车走廊,三者递进,可用于区分"模型真差"还是"只是看不远";
3. **`x{ratio}`**:预测面积/Gt 面积,**不是** IoU 的倍数;`x8.18` 说明严重过预测;
4. **`n` 必须先看**:本项目旧配置(每场景前 8 帧)下 `val3D n=2`、`valStat n=7`,当时的 `R=0.00`、`acc=0.29` 全是小样本噪声;`--val-full` 后 `n=3458 / 1682` 才可信;
5. **`nan` 不等于 0**:分割类别 `nan` = 该切片无此类 GT;`ADEc=nan` = 该切片无弯道帧——都是**样本问题**,不是模型崩了。

---

## 10. 实例:ep2 全量 val(2,967 帧)怎么读

| 头 | 读数 | 判断 |
|---|---|---|
| BEV seg | mIoU 0.158, road 0.476, laneline 0.204 | 早期水平;`redge x8.18` 过预测是主要拖累 |
| 2D seg | mIoU 0.479, road 0.912 | 健康 |
| 3D det | veh P=0.82 **R=0.17**, Rn=0.31 | **精度好、召回低**;走廊内比全场景高 → 远距是短板 |
| Stat | acc 0.75 / movAcc 0.78 (n=1682) | 可用 |
| Traj | agentADE 3.05 m, vru 7.81 m | VRU 轨迹明显更难 |
| TL | acc 0.98, red 0.55 | 红灯召回待提升 |
| E2E | ADE 1.11 m, oracle 0.85(选择损失 +0.26), const-vel 0.79 | **模式选择损失占误差的 23%**;对匀速直行的优势只有 29% → 主要改进方向 |
| HS | wp0 bias +0.43 m (n=240) | 比 ep0 的 +0.81 m 明显改善 |

---

## 11. 代码出处(便于核对)

| 指标 | 产出函数(`METEOR/bevlane/train.py`) |
|---|---|
| `[val]`(BEV seg mIoU、`x{ratio}`) | `evaluate()` + `class_pr()`(docstring: "Precision/recall + pred/GT area ratio") |
| `[val2d]` | `evaluate_seg2d()` |
| `[val3D]` / `[val3D-zone]` / `[valStat*]` 的 R50/Rn | `evaluate_det3d()`(R50:`x²+y²<50²`;Rn:`x²+y²<30²` 且 `\|y\|<12`) |
| `[valStat]` / `[valStat-cal]` | `evaluate_det3d()` 内的 stat 段 + 阈值标定 |
| `[valTraj]` | `evaluate_traj()` |
| `[valTL]` | `evaluate_tl()` |
| `[valE2E]` / `[valE2Ed]` / `[valHS]` | `evaluate_ego()`(HS 用 `dv_hs` 高速子集,`--val-hs` 控制取样数) |

> 相关:`evaluate_occ()`(`[valOCC]`)、`evaluate_risk()`(`[valRisk]`)、`evaluate_flow()`(`[valFlow]`)、
> `evaluate_lanegraph()`(`[valLane]`)、`evaluate_unknown*()`(`[valUnk]`)在当前开启的头未激活时不会打印。
