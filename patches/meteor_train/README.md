# METEOR 训练相关本地修改快照(2026-09-23)

本目录是 **METEOR 训练相关脚本的本地修改快照**。原因:METEOR 仓库的 `origin`
指向**上游官方仓库** `https://github.com/tier4/METEOR.git`(无推送权限),所以把
改动同步一份到本仓库进行版本管理与备份。

## 来源

| 项 | 值 |
|---|---|
| 上游仓库 | `https://github.com/tier4/METEOR.git` |
| 基线 commit | `dc193a8`(English-only source tree) |
| 本地 commit ① | `57ef360` — train.py:no_grad 修复 + resume + 存档 + val-full |
| 本地 commit ② | `9702345` — 6 相机适配 + OCC 顶视图渲染 |
| 本地 commit ③ | `8d37459` — `--val-workers`:val 数据加载并行(需 `--ipc=host`) |
| 本地 commit ④ | `74330ab` — `--depth-head-only`:外参重标定后只微调 depth 头 |
| License | Apache License 2.0(见 `LICENSE-METEOR-Apache-2.0`) |

## 目录内容

| 文件 | 上游路径 | 说明 |
|---|---|---|
| `train.py` | `bevlane/train.py` | 训练主脚本(4,116 行,含本快照 4 项改动) |
| `dataset.py` | `bevlane/dataset.py` | 数据加载(含 6 相机 seg2d 补齐、bev_box_p 保护) |
| `model.py` | `bevlane/model.py` | 网络定义(含 tl_head 6 相机零分支) |
| `orin_render.py` | `deploy/orin_render.py` | Orin 渲染(含 OCC 顶视图新函数) |
| `meteor_changes_20260923.patch` | — | **全部改动的 diff**(328 insertions / 55 deletions,4 文件) |
| `LICENSE-METEOR-Apache-2.0` | `LICENSE` | 上游许可副本(复制源码时保留) |

## 改动清单

### A. `train.py`(commit `57ef360`)

#### A1. `@torch.no_grad()` 修复 —— 根治 val OOM(最关键)

`evaluate_traj` / `evaluate_ego` / `evaluate_unknown_dense` 是**仅有三个没有**
`@torch.no_grad()` 的评估函数(其余 `evaluate_*` 都有)。`model.eval()` 不关闭
autograd,`torch.autocast` 也只降精度不阻止建图,于是它们 19 头的全量前向会
**构建完整计算图**并让所有激活驻留,等一个永不到来的 backward:

- 实测 **+13 GB** 显存尖峰 → epoch-end val 从 11 GB 冲到 24 GB → 24 GB 4090 **OOM**;
- 后果:连续 5/5 轮 val 全部 `SKIPPED (OOM)`,而保存代码在 try 块内 →
  **checkpoint 目录为空,9.5 小时训练成果全丢**;
- 训练中的 `probeE2E` 调用同一函数,官方注释记录它曾在 **44 GB** 卡上也被顶爆;
- 修复后 val 单步真实占用 **≈3.1 GB**。

#### A2. `--resume` 全状态续训

恢复 model + optimizer + OneCycleLR + AMP scaler + epoch + `best/best_e2e/best_chain`,
并从 `epoch+1` 继续。**仅有权重的旧 ckpt** 会把 OneCycleLR 对齐到"已完成步数"
位置,使 LR 轨迹接续而非重新 warmup(实测:中断前 8.85e-05 → 续训起步 **9.50e-05**)。
对比:`--init-ckpt` 只加载权重,Adam 与 LR 调度从头开始。

#### A3. 每 epoch 存档 + OOM 强制补存

每个 epoch 同时写 **`ep{N}.pt`(永不覆盖)** 与 `last.pt`,两者都含完整优化状态;
val 阶段 OOM 时**强制补存**该 epoch 权重(此前 OOM 会跳过整个保存块)。

#### A4. `--val-full` 全量验证

一个开关放开 **三层 val 节流**:

| 层 | 位置 | 默认(抽样) | `--val-full` |
|---|---|---|---|
| ① dataset | `max_per_scene=8` | 只取场景**前 8 帧** | `None`(全场景 2967 帧) |
| ② dv_ep | `_st = len(va)//160` | 抽稀到 ~165 帧 | `_st = 1`(不抽稀) |
| ③ 各评估 | `max_batches=vcap(n)` | 20~166 批次封顶 | `len(va_ep)+1`(放开) |

**代价(实测)**:2970 帧 × ~9 个评估 ≈ **75-80 分钟/次**(旧配置,单进程读图);batch 大小
**不改变**吞吐(GPU 已饱和),故用 batch 1,峰值仅 ~3.5 GB。

#### A5. `--val-workers` val 数据加载并行(commit `8d37459`)

`--val-full` 打开后发现 val 的 4 个 DataLoader 共用训练的 `--workers`(=0,为规避 depth 张量的
共享内存问题)→ **单进程解码 ~71,208 张图**(2967 帧 × 24 张:6 当前 + 18 历史),GPU 长期只有
2~47% 利用率,val 的 75-80 分钟几乎全是 CPU 读图。

| 改动 | 说明 |
|---|---|
| 新增 `--val-workers`(默认 0 = 跟随 `--workers`) | val loader 与训练 loader **worker 数解耦**;val 是 `with_depth=False` + batch 1,worker 安全 |
| `persistent_workers=True` + `prefetch_factor=4`(workers>0 时) | val 要跑 ~9 个 evaluator,每个都重新迭代同一 loader,worker 池需常驻 |

> 🔴 **硬依赖**:`--val-workers > 0` 必须给容器**真实的 `/dev/shm`** —— 启动加 **`--ipc=host`**
> (或 `--shm-size=8g`)。Docker 默认 `/dev/shm` = **64 MB**,8 workers × prefetch 4 × 24 张/样本
> 需要约 **1 GB**,否则报 `DataLoader worker ... Bus error`(实测:batch=2 验证时正是死在
> 这个错误上,而不是 CUDA OOM)。

#### A6. `--depth-head-only` 只微调 depth 头(commit `74330ab`)

XCalib 外参重标定后,冻结的 depth 头与新的投影几何不匹配(实测:深度系统性低估,
40-70m 段 −27%,导致 BEV seg 纵向错位与 OCC 左侧植被过度预测)。本参数:

- 只保留 `depth_head` / `depth_up` / `log_sigma` 可训练(**0.9M / 51.8M**),其余冻结且
  BN 钉死(复用 `--det-head-only` 的 `_freeze_eval_keep` 机制);
- 其他损失权重置 0,只用 depth 损失;
- 用法:`--init-ckpt <picked.pt> --depth-head-only --depth-w 0.3`(其余 `-w 0`),
  配合**新外参重投影的 depth4 GT** 重新训练 2 epoch;
- ⚠️ batch 不能放大(实测 batch=2 峰值 23.9GB OOM):全模型前向峰值激活被 CUDA
  缓存分配器保留,`--grad-ckpt` 对冻结层无效,24GB 卡上 batch=1 是唯一安全值。

### B. 6 相机适配(commit `9702345`)

METEOR 原版按 8 相机(含 FRONT_NARROW / BACK_NARROW)编写,本项目的实车为
**6 相机**环视,以下改动是训练可运行的前提:

| 文件 | 改动 |
|---|---|
| `bevlane/dataset.py` | seg2d 的相机数少于 `len(CAMS)` 时用 255(ignore)补齐;`bev_box_p` 为 None/空时给 `(0,6)` 空数组并 reshape,避免 6 相机场景加载报错 |
| `bevlane/model.py` | v27 `tl_head` 拼接 `f[:,0] + f[:,6]`(FRONT_WIDE + FRONT_NARROW);6 相机没有 slot 6 会 IndexError → 改为对缺失的窄视相机输出零 logits |
| `deploy/orin_render.py` | 新增 `draw_occ_topview()`:正北朝上的 OCC 俯视图面板,可叠加 LiDAR 对齐检查点 |

## 应用方法

```bash
# 方式 A:一次性打全部补丁(推荐)
cd /path/to/METEOR
git apply /path/to/meteor_changes_20260923.patch
# 或: patch -p1 < /path/to/meteor_changes_20260923.patch

# 方式 B:按文件覆盖
cp train.py dataset.py model.py  /path/to/METEOR/bevlane/
cp orin_render.py               /path/to/METEOR/deploy/
```

## 本次训练配置(5 epoch 微调,10 头)

```bash
docker run --rm --gpus all -v <workspace>:/work meteor_training:v1 bash -c "
python3 /work/METEOR/bevlane/train.py \
  --root /work/meteor_6cam_lidar_deploy/scenes --model v52 \
  --init-ckpt /work/METEOR/models/meteor_v157.pt \
  --resume <ckpt_dir>/last.pt --val-full \
  --train-list /work/meteor_6cam_lidar_deploy/train_scenes.txt \
  --val-scenes-file /work/meteor_6cam_lidar_deploy/val_scenes.txt \
  --out <ckpt_dir> \
  --epochs 5 --batch 1 --lr 1e-4 --workers 0 --trim-start 0 --trim-end 0 \
  --seg-w 1.0 --seg2d-w 0.4 --seg2d-key seg2d21 --n-seg2d 21 \
  --depth-w 0.3 --freeze-depth --occ-w 0.5 --ego-w 1.0 \
  --box-w 0.5 --bbox2d-w 0.3 --traj-w 0.5 --stat-w 0.5 --tl-w 0.3 \
  --n-cams 8 --val-batch 1"
```

- 开启头:`seg / seg2d / depth(冻结) / occ / ego / box / bbox2d / traj / stat / tl`
- 冻结头:`risk / flow / lanegraph / unk / pl`
- 数据:`agent_traj` 由 SimpleTrack(ICRA 2022)生成,见 `doc/simpletrack_ab_report.md`
- 诊断基准脚本见本仓库 `tests/`(val 显存 / OOM 复现 / 全量 val 吞吐)

## 同目录 / 相关文档

- **`PR3_vla_overlay_review.md`** —— METEOR 上游 PR #3(`vla-overlay-int8-probes`)的评估笔记:
  VLA overlay 集成 + INT8 车道细化调查,含逐文件改动清单与"对我们部署的价值"判断
- **`../meteor_deploy/`** —— 从该 PR 摘取并**适配 6 相机**的部署工具(`int8_lane_local.py` /
  `int8_lane_x86.py` / `probe_int8_lane.py`)+ INT8 量化防坑手册

## 注意

METEOR 工作区中的 `data/`(655 MB 数据集)、`models/`(536 MB 权重与查表)、
`docker/` 为本地资产,**不应提交到 git**(当前未跟踪状态即正确)。
