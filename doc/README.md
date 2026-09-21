# METEOR 6 相机 + LiDAR 自建数据推理包

自包含目录：6 路环视相机 + 1 个激光雷达的自建数据，经转换后用 METEOR
v157 零样本推理并出 1080p 可视化视频。模型/渲染代码仍引用同级 `../METEOR`
仓库（可用环境变量 `METEOR_REPO` 覆盖）。

## 目录结构

```
meteor_6cam_lidar_deploy/
  engine/   meteor_6cam_lidar.onnx + meteor_6cam_lidar_fp16.engine
  scripts/  convert_custom.py / infer_custom.py / smoke_infer.py
  scenes/   <seq>/ 转换后的 METEOR 场景（manifest.json, img/, lidar_bev/, ego_motion.npz）
  videos/   推理输出 mp4
  doc/      文档中心 (README.md, MODEL_ARCHITECTURE.md, DATASET_AUTOLABEL_PLAN.md)
```

## 输入数据约定（每序列）

- `calib/camera/camera_*.json`：内参（1920x1080，已去畸变）+
  `extrinsic = T_cam_lidar`（LiDAR→相机，光学系 z前/x右/y下）
- `calib/lidar/lidar2imu_calib.txt`：末段 4x4 = `T_ego_lidar`
  （LiDAR→车身，x前 y左 z上）
- `camera/camera_*/<ts>.jpg`：6 路 1920x1080，10 Hz 时间同步
- `lidar/<ts>.pcd`：PCD v0.7 binary，x y z intensity float32
- `localization/<ts>.yaml`：车身系 vel.x / vel.y（用于 v0）
- 相机映射：front100→CAM_FRONT_WIDE, left_front→CAM_FRONT_LEFT,
  right_front→CAM_FRONT_RIGHT, back→CAM_BACK_WIDE,
  left_back→CAM_BACK_LEFT, right_back→CAM_BACK_RIGHT

## 坐标链与栅格

- `T_ego_cam = T_ego_lidar @ inv(T_cam_lidar)`；运行时喂 `T_cam_ego=inv(T_ego_cam)`
- 点云：`p_ego = p_lidar @ R_el^T + t_el`
- LiDAR BEV：自车系，0.4 m，行 `(80-x)/0.4`、列 `(50-y)/0.4`，
  4 通道 = [log1p 点数, max z(裁剪 -1..4), mean z, 占据]
- 内参按 0.4 等比缩放到 768x432

## 使用（容器 meteor_trt10_cu124_py310:v1，--gpus all --network host）

```bash
REPO=/home/nvidia/working_ppt/Postdoc_Materials/论文3
PKG=$REPO/meteor_6cam_lidar_deploy

# 1) 转换一个原始序列（先加 --limit 100 冒烟）
python $PKG/scripts/convert_custom.py \
  --src  $REPO/METEOR/data/data_20260910_064331 \
  --out  $PKG/scenes/data_20260910_064331 --workers 16

# 2) 推理 + 出 1080p mp4（--no-lidar 切纯相机对照）
python $PKG/scripts/infer_custom.py \
  --scene  $PKG/scenes/data_20260910_064331 \
  --engine $PKG/engine/meteor_6cam_lidar_fp16.engine \
  --out    $PKG/videos/data_20260910_064331.mp4 --fps 10

# 3) 仅接口/时延冒烟（不渲染）
python $PKG/scripts/smoke_infer.py \
  $PKG/scenes/data_20260910_064331 \
  $PKG/engine/meteor_6cam_lidar_fp16.engine
```

## 引擎接口

- 输入：`imgs[1,6,3,432,768] uint8 RGB`、`K[1,6,3,3]`（768x432）、
  `T_cam_ego[1,6,4,4]`、`v0[1]` m/s、
  `lidar_bev[1,4,400,250] fp32`、`lidar_flag[1]`（1.0）
- 相机顺序：CAM_FRONT_WIDE, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
  CAM_BACK_WIDE, CAM_BACK_LEFT, CAM_BACK_RIGHT
- 无 FRONT_NARROW 长焦，TL 红绿灯头无有效输入，导出时 `--drop tl`
- 4090 fp16：纯推理约 25 ms/帧；含渲染/编码约 7 fps
- no-hist 引擎按训练条件零历史运行

## 已知限制

- 零样本，未在本车数据微调；后视仅 52°（back fx≈1950），远距小目标偏弱
- 更换标定/车型必须重跑 convert，不可复用旧 manifest 的 K/T
