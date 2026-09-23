# 标定脚本说明

本目录是 XCalib 相机–激光雷达标定的可复现管线脚本。脚本按编号顺序执行，
每个脚本的输出供后续脚本使用。所有路径由 `common_paths.py` 和
`config/calibration.json` 解析，默认输入为 `../../raw_data`、`../../scenes`，
输出写入 `xcalib_calibration/` 下各产物目录。

## 运行环境

- 独立虚拟环境：`xcalib_calibration/.venv`（Python 3.12，含 CUDA PyTorch）
- 如遇代理报错（`Unknown scheme for proxy URL ... socks://`），运行前加：

  ```bash
  env -u ALL_PROXY -u all_proxy
  ```

## 标定链路（重要）

`bev_box/boxes_3d` 存储在 **ego/IMU 坐标系**（由 `lidar2imu_calib.txt` 变换后
聚类生成）。因此每个 3D 点投影前必须先做 **ego → lidar** 逆变换，再用直接的
`T_lidar_camera` 投影：

```
3D 框(ego) ──逆 lidar2imu──> lidar ──T_lidar_camera──> camera
```

不涉及、也不修改 IMU/ego 的其他变换。缩放图（768×432，比例 0.4）只改变内参和
像素坐标，外参与全分辨率完全一致；部署全分辨率时搭配原始全分辨率内参。

## XCalib 工作原理

XCalib 不是传统标靶标定，而是**基于检测框的跨模态匹配器 + PnP 标定器**：
输入一帧的图像、点云及各自的检测框，模型判断图像中的哪个 2D 框对应点云中的
哪个 3D 框，再用这些 2D–3D 对应关系解外参，属于无靶标（targetless）标定。

1. **输入与裁剪**：输入为图像、点云、`bboxes_2d`（xyxy）、`bboxes_3d`、内参 K。
   每个 2D 框裁出并 resize 成固定尺寸小图；点云按 3D 框裁出局部点云（保留全局
   XYZ，约 1.25 倍扩展），再子采样/补齐到固定点数。模型处理的是逐个目标的局部
   外观而非整图。
2. **CRLite 两阶段匹配**：
   - 特征：2D 框走 ResNet、3D 框走 PointNet，并融合 2D/3D 位置编码，同时利用
     外观、几何和位置先验。
   - 阶段 1（粗筛）：图像向量与点云向量做余弦相似度得 N×M 矩阵，每个 2D 框取
     top-k 个 3D 候选。
   - 阶段 2（精排）：仅对 top-k 候选，拼接两向量后过融合 MLP + 相似度头，输出
     归一化到 [0,1] 的最终分数，Top-1 即匹配结果（`xcalib_match_pairs.csv` 来源）。
3. **匹配到外参**：每对匹配给出一组「2D 框中心 ↔ 3D 框中心」的 2D–3D 点对应；
   在已知 K 下用 `cv2.solvePnPRansac` 求 [R|t] 并做 LM 精修，单帧不足时跨帧累积
   高置信对应后统一 RANSAC。

**本项目为何只取其匹配能力**：XCalib 自带 `calibrate` 直接在匹配对上做 RANSAC
PnP，但本数据 3D 框为 ego 坐标且自动框噪声较大，直接 PnP 不稳定。因此管线仅借用
其最可靠的匹配结果（判断谁对应谁），残差由脚本 `04`/`05` 自行做物理分解
（俯仰 vs 相机 y 向平移）与单自由度有界 LOSO 精修，比直接 PnP 更稳健。

XCalib 还支持 ONNX/TensorRT 导出（边缘部署）、HDF5 训练/微调和一次性
（one-shot）自适应更新。

## 脚本一览（按执行顺序）

| 脚本 | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `01_build_frame_index.py` | 扫描全部场景的点云/六相机文件，建立逐帧索引并核对时间对齐 | `raw_data/*/lidar`、`raw_data/*/camera` | `index/frame_index.csv`、`index/frame_index_summary.json` |
| `02_convert_scaled_calibrations.py` | 读取原始相机内外参，生成缩放图内参，保留原始直接外参 | `raw_data/*/calib/camera/*.json`、`scenes/*/manifest.json` | `converted/scaled_calibrations.json` |
| `03_baseline_projection.py` | 用原始外参在正确链路上投影 3D 框，与 2D 框配对，统计基线重投影误差并出叠加图 | `scenes/<baseline_scene>/bev_box`、`bbox2d` | `baseline/baseline_summary.json`、`baseline/baseline_pairs.csv`、`overlays/baseline_grid_*.jpg` |
| `04_cross_scene_residual_analysis.py` | 六场景各均匀抽 30 帧，几何配对后按相机分析有符号残差，并做 `dv = a + b/depth` 分解（常数项=俯仰，`1/depth` 项=相机 y 向平移） | 六场景的 `bev_box`、`bbox2d` | `validation/cross_scene_residuals.csv`、`validation/cross_scene_residual_summary.json`、`overlays/cross_scene_residuals.png` |
| `05_single_dof_refine.py` | 六场景各均匀抽 200 帧，对指定相机做**单自由度** LOSO 精修（俯仰或相机 y 向平移），生成最终部署外参 | 六场景的 `bev_box`、`bbox2d` | `refined_single_dof/single_dof_results.json`、`refined_single_dof/deploy_extrinsics_final.json`、`validation/single_dof_loso.csv`、`overlays/single_dof_refine.png` |
| `06_compare_before_after.py` | 每场景随机抽 20 帧，在同图上用红色（优化前）/绿色（优化后）叠加 3D 投影框，六相机拼网格 | `deploy_extrinsics_final.json`、六场景数据 | `overlays/compare_before_after/<scene>/frame_*.jpg`（共 120 张） |

## 共享模块

- `common_paths.py`：解析标定目录、部署目录，加载配置。
- `_calibration_utils.py`：3D 框角点、AABB、点云读取（PCD binary）、坐标变换、
  针孔投影、IoU、均匀抽样等通用函数。

## 最终标定结论（2026-09-23）

- 不修正：`front_wide`、`back_right`（残差不显著或修正反而恶化）。
- 采用精修：
  - `front_left`：相机俯仰 −0.378°（绕光心）
  - `front_right`：相机俯仰 −0.469°（绕光心）
  - `back_wide`：相机 y 向平移 +0.174 m
  - `back_left`：相机 y 向平移 +0.084 m
- 修正后四相机留出集纵向中位偏差均归零；最终矩阵见
  `refined_single_dof/deploy_extrinsics_final.json`。
- 上述精修建立在 2D/3D 自动框可靠的假设上，建议上线后用实际投影效果再抽查。
