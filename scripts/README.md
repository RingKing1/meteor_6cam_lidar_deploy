# 自动标注与生产流水线脚本索引 (Scripts Directory Guide)

本目录汇集 METEOR 6 相机 + 1 激光雷达多模态端到端自动驾驶系统的**数据自动生产、标定转换、训练微调与可视化评估**工具链,按流水线分门别类归档。

- 主线生产入口:**`run_phase3_full_production.sh`**(6 场景 17,845 帧四步全流程)
- 一键总控入口:**`run_production_pipeline.py`**(多工序串行)
- 文档索引:轨迹引擎细节见 [`../doc/simpletrack_ab_report.md`](../doc/simpletrack_ab_report.md);训练头清单与网络拓扑见 [`../README.md`](../README.md)

---

## 📂 脚本目录架构

```text
scripts/
├── README.md                          # 本说明文件
├── run_production_pipeline.py         # 🌟 全自动一键多工序生产总控
├── run_phase3_full_production.sh      # 🌟 Phase-3 主线:6 场景四步全流程(推理→融合→追踪→导出)
├── train_meteor_full_5ep.sh           # ⚠️ 历史 5-epoch 训练启动脚本(配置说明见文末)
│
├── autolabel_ego_motion/              # 🧭 【基础】数据转换与自车运动/位姿
│   ├── convert_custom.py              # 原始 6 相机 + LiDAR 时间戳对齐、manifest 资产清单
│   └── build_ego_motion.py            # WGS-84 → ENU、位姿平滑、3.0s 未来航向点解算
│
├── autolabel_3d_box/                  # 🚗 【已实现】3D 检测框(FocalFormer 路线)
│   ├── infer_focalformer_dataset.py   # FocalFormer3D-LC 全量推理 + 右手系转换 + 自车框过滤
│   ├── build_3d_box_gt.py             # (旧路线)5 帧位姿补偿点云融合 + 6 相机多视反投影过滤
│   ├── prune_ego_boxes.py             # 自车误检二次清理(is_ego_vehicle_box 批处理)
│   ├── compare_focalformer_centerpoint.py # FocalFormer3D vs CenterPoint 对比评测
│   ├── bev_box_visualizer.py          # 6 路环视 3D 线框投影 + BEV 栅格渲染
│   └── watch_and_push.sh              # ⚠️ 一次性 watcher(等推理完成→统计→推送,含硬编码容器名)
│
├── autolabel_bbox2d/                  # 🎯 【已实现】三模态共识融合(2D 检测 + 净化 3D)
│   ├── tri_modal_consensus_fusion.py  # FocalFormer3D + YOLOv8x + Mask2Former 三模态投票
│   └── compare_baseline_vs_consensus.py # 基线 0.25 vs 共识 0.15 定量/可视化对比
│
├── autolabel_agent_traj/              # 🛤️ 【已实现】3D MOT + 3.0s 未来轨迹
│   ├── simpletrack_mot.py             # SimpleTrack(ICRA 2022)接入模块:bev_box→world→FrameData
│   ├── build_agent_traj_simpletrack.py # ✅ 当前引擎:输出格式兼容 + 追加 track_ids
│   ├── build_agent_traj_gt.py         # (旧引擎)自研 MOTTracker3D:匀速外推 + L2 关联
│   ├── compare_mot_trackers.py        # A/B 对比:轨迹持久性/碎片率/tvalid 覆盖/平滑度
│   └── visualize_agent_traj.py        # 2D 框 + 3D 框 + 未来 3s 轨迹多模态可视化
│
├── autolabel_instance_tracking/       # 🧩 【已实现】2D 实例追踪与 T4 导出
│   └── export_t4_instance.py          # 3D 框投影↔2D 框 IoU 关联,实例 token = 真实 track id
│
├── autolabel_traffic_light/           # 🚦 【已实现】红绿灯状态识别
│   └── build_tl_gt.py                 # 6 相机 ROI 提取、时序追踪、HSV 色彩投票
│
├── autolabel_semantic_occ/            # 🌐 【已实现/持续升级】语义分割与 3D Occupancy
│   ├── batch_2d_panoptic.py           # Mask2Former 2D 全景语义掩码批量推理
│   ├── test_2d_panoptic.py            # 2D 全景分割模型选型评测
│   ├── build_bev_gt.py                # 点云时序累加 + BEV 车道地图 GT
│   ├── build_depth_and_occ.py         # 10 类语义 Occupancy + 度量深度(depth4)
│   ├── promote_solid_crosswalk.py     # 6 场景实心连续人行横道提升(17,845 帧)
│   ├── generate_crosswalk_test_vis.py # 20 例人行横道微调测试可视化
│   ├── occ_visualizer.py              # 3D 线框 + 10 类 OCC 俯视可视化
│   └── diag_occ.py                    # OCC 路面体素 vs LiDAR 地面点诊断
│
└── training_eval/                     # 🚀 微调训练与端到端评测
    ├── run_finetune.py                # 微调启动器(单卡/多卡,任务损失权重可配)
    ├── pause_after_epoch1.py          # 监控日志,epoch 1 完成时安全暂停
    ├── infer_custom.py                # 端到端推理 → mp4
    ├── generate_3d_videos.sh          # 3D 等轴测 OCC 评测视频批量生成
    ├── visualize_val.py               # 验证集多任务高分辨率仪表盘
    ├── visualize_intent.py            # 导航意图条件引导多模态轨迹可视化
    ├── smoke_infer.py                 # 单帧推理冒烟测试
    └── trace_model.py                 # 逐层张量规格与算子探测
```

> 说明:`__init__.py` 为包标记文件,未在上表逐一列出。

---

## 🚀 Phase-3 主线全流程(`run_phase3_full_production.sh`)

对 6 场景 **17,845 帧**执行四步生产,每场景串行:

| 步 | 脚本 | 作用 | 产物 |
|---|---|---|---|
| 1/4 | `autolabel_3d_box/infer_focalformer_dataset.py` | FocalFormer3D-LC 推理(score≥0.15,自车框过滤) | `box3d_artifacts/focalformer_bev_box_015/` |
| 2/4 | `autolabel_bbox2d/tri_modal_consensus_fusion.py` | 三模态共识投票(3D 候选 + YOLOv8x + Mask2Former) | `scenes/<s>/bev_box/`、`bbox2d/` |
| 3/4 | `autolabel_agent_traj/build_agent_traj_simpletrack.py` | **SimpleTrack** MOT + 3.0s 未来轨迹 | `scenes/<s>/agent_traj/`(+`track_ids`) |
| 4/4 | `autolabel_instance_tracking/export_t4_instance.py` | 2D 实例追踪 + TIER IV t4dataset 导出 | `scenes/<s>/annotation/` |

```bash
# 全量运行(6 场景,GPU 主导;第 3 步为纯 CPU 主机 venv)
bash scripts/run_phase3_full_production.sh

# 单场景(第 3 步示例,主机 venv)
tools/SimpleTrack-main/venv_simpletrack/bin/python \
  scripts/autolabel_agent_traj/build_agent_traj_simpletrack.py \
  --root scenes --scenes data_20260910_061820 --out-subdir agent_traj --force
```

**实测耗时参考**(RTX 4090):场景 1–2 各 ~60 min(推理占 57–61 min),第 3 步 <1 s/场景(3,000 帧),第 4 步 ~3 s/场景。

---

## 🛠️ 各流水线典型指令

### 1. 数据转换与自车运动 (`autolabel_ego_motion/`)
```bash
python3 scripts/autolabel_ego_motion/convert_custom.py --help      # 原始数据 → METEOR 场景
python3 scripts/autolabel_ego_motion/build_ego_motion.py --scenes all
```

### 2. 3D 检测框标注 (`autolabel_3d_box/`)
```bash
# FocalFormer 路线(当前主线)
python3 scripts/autolabel_3d_box/infer_focalformer_dataset.py --scenes all --score-thresh 0.15
# 自车框清理 / 可视化抽检
python3 scripts/autolabel_3d_box/prune_ego_boxes.py --scenes all
python3 scripts/autolabel_3d_box/bev_box_visualizer.py --scene data_20260910_061820 --frames 100,200,400,600
```

### 3. 三模态共识融合 (`autolabel_bbox2d/`)
```bash
python3 scripts/autolabel_bbox2d/tri_modal_consensus_fusion.py --scenes all \
  --focal-dir box3d_artifacts/focalformer_bev_box_015
python3 scripts/autolabel_bbox2d/compare_baseline_vs_consensus.py --scene data_20260910_061820
```

### 4. 轨迹生成与 A/B 对比 (`autolabel_agent_traj/`)
```bash
# 生产(SimpleTrack 引擎)
tools/SimpleTrack-main/venv_simpletrack/bin/python \
  scripts/autolabel_agent_traj/build_agent_traj_simpletrack.py --root scenes --scenes all --force
# 对比旧自研引擎(轨迹持久性/碎片率/tvalid 覆盖)
tools/SimpleTrack-main/venv_simpletrack/bin/python \
  scripts/autolabel_agent_traj/compare_mot_trackers.py data_20260910_061820
python3 scripts/autolabel_agent_traj/visualize_agent_traj.py --scene data_20260910_061820 --frames 100,300
```

### 5. 实例追踪导出 (`autolabel_instance_tracking/`)
```bash
python3 scripts/autolabel_instance_tracking/export_t4_instance.py --root scenes --scenes all
# 产物:annotation/{category,instance,object_ann}.json(instance_token = <scene>_track_<cls>_<tid>)
```

### 6. 红绿灯 (`autolabel_traffic_light/`)
```bash
python3 scripts/autolabel_traffic_light/build_tl_gt.py --scenes all --workers 8
```

### 7. BEV 车道与 Occupancy (`autolabel_semantic_occ/`)
```bash
python3 scripts/autolabel_semantic_occ/batch_2d_panoptic.py --scene data_20260910_061820
python3 scripts/autolabel_semantic_occ/build_bev_gt.py --scene data_20260910_061820
python3 scripts/autolabel_semantic_occ/build_depth_and_occ.py --scene data_20260910_061820
python3 scripts/autolabel_semantic_occ/promote_solid_crosswalk.py          # 6 场景人行横道提升
python3 scripts/autolabel_semantic_occ/occ_visualizer.py --scene data_20260910_061820 --frame 500
```

### 8. 训练与评测 (`training_eval/`)
```bash
# 微调(容器内运行;实际生产命令见下节)
python3 scripts/training_eval/run_finetune.py --model v52 \
  --init-ckpt /work/METEOR/models/meteor_v157.pt \
  --train-scenes train_scenes.txt --val-scenes val_scenes.txt

python3 scripts/training_eval/visualize_val.py --weights checkpoints/best.pt
python3 scripts/training_eval/visualize_intent.py --frame 100 --weights checkpoints/best.pt
bash    scripts/training_eval/generate_3d_videos.sh
```

---

## 🏋️ 当前训练命令(10 头 · 全量 val · 可续训)

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
- `--resume` 恢复 optimizer + OneCycleLR + scaler + epoch;每 epoch 另存 `ep{N}.pt`
- `--val-full` 评估全量 2,967 帧(val 时间 ~75–80 min/epoch)
- 训练脚本补丁快照与说明:[`../patches/meteor_train/`](../patches/meteor_train/)

---

## 📌 脚本状态说明

| 标记 | 含义 | 代表脚本 |
|---|---|---|
| ✅ 当前在用 | 生产主线 | `build_agent_traj_simpletrack.py`、`tri_modal_consensus_fusion.py`、`export_t4_instance.py`、`infer_focalformer_dataset.py` |
| 🔄 保留备查 | 已被替代但保留可对比/回退 | `build_agent_traj_gt.py`(旧 MOT)、`build_3d_box_gt.py`(点云聚类路线) |
| ⚠️ 历史/一次性 | 依赖特定容器名或旧配置,勿直接复用 | `watch_and_push.sh`(硬编码容器 `silly_elbakyan`)、`train_meteor_full_5ep.sh`(配置为旧 5 头:`box-w 1.0 / tl-w 0.5`,且 `traj/stat/bbox2d` 权重为 0) |

> 复用 `train_meteor_full_5ep.sh` 前请对照上文"当前训练命令"更新损失权重与 `--resume/--val-full` 参数。
