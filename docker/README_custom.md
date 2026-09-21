# 自建 6 相机 + LiDAR 数据集适配 METEOR

适用数据（每序列）：6 路 1920x1080 环视 JPG + 1 个 PCD v0.7 binary
（x y z intensity float32），10 Hz 时间同步，定位 YAML 含车身系速度，
相机 JSON 含内参与 `extrinsic = T_cam_lidar`（LiDAR→相机，光学系），
`calib/lidar/lidar2imu_calib.txt` 末段 4x4 = `T_ego_lidar`（LiDAR→车身，
x前 y左 z上）。

## 产物

- `out/meteor_6cam_lidar.onnx` — 6 相机 + LiDAR、no-hist、uint8 输入
- `out/meteor_6cam_lidar_fp16.engine` — 4090 fp16 引擎，约 25 ms/帧
- 输入：`imgs[1,6,3,432,768] uint8 RGB`、`K[1,6,3,3]`（768x432 尺度）、
  `T_cam_ego[1,6,4,4]`、`v0[1]` m/s、`lidar_bev[1,4,400,250]` fp32、
  `lidar_flag[1]`（传 1.0）
- 相机顺序：CAM_FRONT_WIDE, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
  CAM_BACK_WIDE, CAM_BACK_LEFT, CAM_BACK_RIGHT
- 无 FRONT_NARROW 长焦，TL 红绿灯头无有效输入，导出时 `--drop tl` 剪枝

## 坐标链

- 相机：`T_ego_cam = T_ego_lidar @ inv(T_cam_lidar)`，存 manifest；
  运行时喂 `T_cam_ego = inv(T_ego_cam)`
- 点云：`p_ego = p_lidar @ R_el^T + t_el`
- LiDAR 栅格：自车系，0.4 m，行 `(80-x)/0.4`、列 `(50-y)/0.4`，
  4 通道 = [log1p 点数, max z(裁剪 -1..4), mean z, 占据]
- 内参按 0.4 等比缩放到 768x432（图像已去畸变，直接 AREA 缩放）

## 命令（容器 meteor_trt10_cu124_py310:v1，--network host）

```bash
# 1) 转换数据（可先 --limit 100 冒烟）
python docker/convert_custom.py \
  --src data/data_20260910_064331 \
  --out out/custom/data_20260910_064331 --workers 16

# 2) 推理 + 出 1080p mp4（--no-lidar 可切纯相机对照）
python docker/infer_custom.py \
  --scene out/custom/data_20260910_064331 \
  --engine out/meteor_6cam_lidar_fp16.engine \
  --out out/custom_full_064331.mp4 --fps 10

# 3) 最小接口冒烟（只打印输出形状/时延，不渲染）
python docker/smoke_infer.py out/custom/data_20260910_064331 \
  out/meteor_6cam_lidar_fp16.engine
```

## 重新导出/建引擎（如需）

```bash
python deploy/export_onnx.py --ckpt models/meteor_v157.pt --model v52 \
  --n-cams 6 --drop unk,pl,flow,tl --uint8-in --argmax-out --no-hist \
  --depth-mean --with-lidar --out out/meteor_6cam_lidar.onnx
python deploy/build_engine_fp16.py out/meteor_6cam_lidar.onnx \
  out/meteor_6cam_lidar_fp16.engine 8
```

## 已知限制

- 零样本：v157 未在本车数据上微调；后视仅 52°（back fx≈1950），
  远距小目标/红绿灯偏弱。
- no-hist：引擎按训练条件零历史运行；occ/占用预测正常，时序融合关闭。
- 若更换标定/车型，重跑 convert 即可；不要复用旧 manifest 的 K/T。
