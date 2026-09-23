# METEOR `bevlane/train.py` 本地修改快照(2026-09-23)

本目录是 **METEOR 训练脚本的本地修改版快照**,原因是 METEOR 仓库的 `origin`
指向**上游官方仓库** `https://github.com/tier4/METEOR.git`(无推送权限),
所以把改动同步一份到本仓库进行版本管理与备份。

## 来源

| 项 | 值 |
|---|---|
| 上游仓库 | `https://github.com/tier4/METEOR.git` |
| 基线 commit | `dc193a8`(English-only source tree) |
| 本地修改 commit | `57ef360`(fix(train): no_grad on all evaluate_* + full-state resume + per-epoch ckpt + --val-full) |
| 文件 | `bevlane/train.py`(4,116 行) |
| License | Apache License 2.0(见 `LICENSE-METEOR-Apache-2.0`,随上游仓库) |

## 目录内容

| 文件 | 说明 |
|---|---|
| `train.py` | 修改后的**完整文件**(可直接覆盖到 `METEOR/bevlane/train.py`) |
| `train_20260923.patch` | 相对上游基线 `dc193a8` 的 **diff**(290 行,推荐用这个) |
| `LICENSE-METEOR-Apache-2.0` | 上游许可副本(复制源码时保留) |

## 改动清单(4 项)

### 1. `@torch.no_grad()` 修复 —— 根治 val OOM(最关键)

`evaluate_traj` / `evaluate_ego` / `evaluate_unknown_dense` 是**仅有三个没有**
`@torch.no_grad()` 的评估函数(其余 `evaluate_*` 都有)。`model.eval()` 不会关闭
autograd,`torch.autocast` 也只降精度不阻止建图,于是它们的 19 头全量前向会**构建
完整计算图**并让所有激活驻留,等一个永不到来的 backward:

- 实测 **+13 GB** 显存尖峰 → epoch-end val 从 11 GB 冲到 24 GB → 24 GB 4090 **OOM**;
- 后果:连续 5/5 轮 val 全部 `SKIPPED (OOM)`,而保存代码在 try 块内 →
  **checkpoint 目录为空,9.5 小时训练成果全丢**;
- 训练中的 `probeE2E` 同一函数,官方注释记录它曾在 **44 GB** 卡上也被顶爆;
- 修复后 val 单步真实占用 **≈3.1 GB**。

### 2. `--resume` 全状态续训

`--resume <ckpt>` 恢复 model + optimizer + OneCycleLR + AMP scaler + epoch +
`best/best_e2e/best_chain` 跟踪量,并从 `epoch+1` 继续。
**仅有权重的旧 ckpt** 会把 OneCycleLR 对齐到"已完成步数"位置,使 LR 轨迹接续
而不是重新 warmup(实测:中断前 8.85e-05 → 续训起步 **9.50e-05**,成功接上)。

> 对比:`--init-ckpt` 只加载权重,Adam 与 LR 调度从头开始。

### 3. 每 epoch 存档 + OOM 强制补存

- 每个 epoch 同时写 **`ep{N}.pt`(永不覆盖)** 与 `last.pt`,两者都含完整优化状态;
- val 阶段 OOM 时**强制补存**该 epoch 权重(此前 OOM 会跳过整个保存块)。

### 4. `--val-full` 全量验证

一个开关放开 **三层 val 节流**:

| 层 | 位置 | 默认(抽样) | `--val-full` |
|---|---|---|---|
| ① dataset | `max_per_scene=8` | 只取场景**前 8 帧** | `None`(全场景 2967 帧) |
| ② dv_ep | `_st = len(va)//160` | 抽稀到 ~165 帧 | `_st = 1`(不抽稀) |
| ③ 各评估 | `max_batches=vcap(n)` | 20~166 批次封顶 | `len(va_ep)+1`(放开) |

**代价(实测)**:2970 帧 × ~9 个评估 ≈ **75-80 分钟/次**;batch 大小**不改变**
吞吐(GPU 已饱和),故用 batch 1,峰值仅 ~3.5 GB。

## 应用方法

```bash
# 方式 A:打补丁(推荐,只含改动)
cd /path/to/METEOR
git apply /path/to/train_20260923.patch
# 或: patch -p1 < /path/to/train_20260923.patch

# 方式 B:直接覆盖
cp train.py /path/to/METEOR/bevlane/train.py
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
