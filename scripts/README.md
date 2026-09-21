# 自动标注与生产流水线脚本索引说明 (Scripts Directory Guide)

本目录汇集了 METEOR 6 相机 + 激光雷达多模态端到端自动驾驶系统的数据自动生产、标定转换、训练微调与可视化评估工具链。各功能模块已按专业流水线分门别类归档。

---

## 📂 脚本目录架构

```text
scripts/
├── run_production_pipeline.py    # 🌟 全自动一键多工序生产总控入口
├── README.md                     # 本说明文件
│
├── autolabel_3d_box/             # 🚗 【已实现】3D 目标检测边界框自动标注流水线
│   ├── build_3d_box_gt.py        # 5 帧位姿补偿点云融合 + 6 相机多视反投影语义过滤
│   ├── infer_focalformer_dataset.py # FocalFormer3D-LC 多模态全量推理与右手坐标系转换
│   ├── compare_focalformer_centerpoint.py # FocalFormer3D 与 CenterPoint 对比评测
│   └── bev_box_visualizer.py     # 6 路环视 3D 线框投影 + BEV 栅格高精渲染工具
│
├── autolabel_traffic_light/      # 🚦 【已实现】红绿灯状态识别自动标注流水线
│   └── build_tl_gt.py            # 6 相机 ROI 提取、多目标时序追踪与 HSV/色彩直方图投票
│
├── autolabel_semantic_occ/       # 🌐 【待升级】语义分割与 3D 占据体素自动标注流水线
│   ├── batch_2d_panoptic.py      # Mask2Former 2D 全景语义掩码批量推理
│   ├── test_2d_panoptic.py       # 2D 全景分割模型性能与选型评测
│   ├── build_bev_gt.py           # 点云时序累加 + 伪标签 BEV 车道地图生成
│   ├── build_depth_and_occ.py    # 稠密化几何深度与 3D 占据体素（Occupancy）构建
│   └── diag_occ.py               # 3D 体素类间分布与空网格诊断分析
│
├── autolabel_ego_motion/         # 🧭 自车运动与位姿解算流水线
│   ├── convert_custom.py         # 原始相机与点云时间戳对齐与 manifest 资产清单构建
│   └── build_ego_motion.py       # WGS-84 经纬度转 ENU、位姿平滑与 3.0s 未来航向点解算
│
└── training_eval/                # 🚀 模型微调训练与端到端规划评测
    ├── run_finetune.py           # 51.83M 模型微调启动器（单卡/多卡，支持任务损失权重）
    ├── pause_after_epoch1.py     # 训练进度监控与定点安全拦截钩子
    ├── infer_custom.py           # 模型端到端推理引擎与 1080p 多任务视频导出
    ├── generate_3d_videos.sh     # 3D 等轴测 Occupancy 体素评测视频批量生成脚本
    ├── visualize_val.py          # 验证集多任务高分辨率仪表盘输出
    ├── visualize_intent.py       # 导航意图条件引导多模态轨迹验证与可视化
    ├── smoke_infer.py            # 快速单帧推理冒烟测试
    └── trace_model.py            # PyTorch 逐层张量规格与算子探测工具
```

---

## 🛠️ 各流水线典型执行指令

### 1. 3D 目标检测标注流水线 (`autolabel_3d_box/`)
```bash
# 全量构建 5 帧位姿补偿 3D Box 真值
python3 scripts/autolabel_3d_box/build_3d_box_gt.py --scenes all --n-sweeps 5 --workers 16

# 启动 6 相机多视角线框投影 + BEV 地图可视化验证
python3 scripts/autolabel_3d_box/bev_box_visualizer.py --scene data_20260910_061820 --frames 100,200,400,600
```

### 2. 红绿灯状态识别标注流水线 (`autolabel_traffic_light/`)
```bash
# 全量构建 6 相机红绿灯真值
python3 scripts/autolabel_traffic_light/build_tl_gt.py --scenes all --workers 8
```

### 3. BEV 车道与 Occupancy 流水线 (`autolabel_semantic_occ/`)
```bash
# 提取 2D 全景语义掩码
python3 scripts/autolabel_semantic_occ/batch_2d_panoptic.py --scene data_20260910_061820

# 构建 BEV 车道地图与 Occupancy 3D 体素
python3 scripts/autolabel_semantic_occ/build_bev_gt.py --scene data_20260910_061820
python3 scripts/autolabel_semantic_occ/build_depth_and_occ.py --scene data_20260910_061820
```

### 4. 模型训练与多模态规划评测 (`training_eval/`)
```bash
# 启动包含 3D Box 与 红绿灯的多任务微调
python3 scripts/training_eval/run_finetune.py \
  --model v52 \
  --init-ckpt /work/METEOR/checkpoints/meteor_v157.pt \
  --train-scenes train_scenes.txt \
  --val-scenes val_scenes.txt \
  --box-w 1.0 \
  --tl-w 1.0

# 导航意图条件引导轨迹验证
python3 scripts/training_eval/visualize_intent.py --frame 100 --weights checkpoints/best.pt
```
