# METEOR 部署工具(6 相机适配)与 INT8 量化防坑手册

本目录从 **METEOR 上游 PR 分支 `vla-overlay-int8-probes`** 摘取部署相关工具,并**适配为
6 相机实车配置**(本项目 rig)。上游来源与许可见文末。

| 文件 | 来源(上游) | 说明 |
|---|---|---|
| `int8_lane_local.py` | `deploy/int8_lane_local.py` | 工作站 TensorRT 8.6(pycuda):构建 fp16 / INT8 / INT8+sparse 引擎 + 逐类 BEV 像素探针 |
| `int8_lane_x86.py` | `deploy/int8_lane_x86.py` | TensorRT 10(cuda-python)版,同上 |
| `probe_int8_lane.py` | `bevlane/probe_int8_lane.py` | **纯 PyTorch 假 INT8 代理**(训练主机无 TensorRT 时用):per-channel 权重量化 + per-tensor 激活量化,报告车道像素比与 IoU 变化 |

三者的共同用途:**回答"INT8 量化会不会把细车道线吃掉"**,并定位是量化、是校准、还是
输入几何的问题。

### `docker/` —— 部署容器资产(原样归档)

上游 METEOR 仓库 `docker/` 的 10 个通用文件(本地 METEOR commit `1203fa3`,因上游无推送权限,
在此做远端备份;sha256 已与 METEOR 工作区逐一核对一致):

| 文件 | 用途 |
|---|---|
| `Dockerfile` | CUDA 12.4 + TensorRT 10.3 + ONNXRuntime 运行底座 |
| `build.sh` / `run.sh` | 镜像构建 / 容器启动(挂载工作区) |
| `pip.conf` / `sources.list.jammy` | 国内 pip / Ubuntu 22.04 apt 镜像源 |
| `convert_custom.py` | 自采数据 → METEOR 场景转换 |
| `infer_custom.py` / `smoke_infer.py` | 端到端推理 / 单帧冒烟测试 |
| `dl_dataset.py` | 数据集下载辅助 |
| `README_custom.md` | 自采数据部署备忘 |

> 这些文件是**原样归档**(未做 6 相机适配):`convert_custom.py` 等按 8 相机 `CAMS` 编写,
> 用在我们的 6 相机 rig 前需核对相机列表(参考本项目 `scripts/autolabel_ego_motion/convert_custom.py`)。

---

## ⚠️ 部署防坑要点(上游 §6.x 实测结论,直接适用)

以下结论来自上游对 "Orin 上车道线变细/消失" 的系统排查,**我们部署 6 相机 Orin 引擎前必须逐条核对**:

### 1. INT8 + 2:4 稀疏:必须把 `layer1` 留在 fp16

| 观测 | 结果 |
|---|---|
| Orin 上 **INT8+sparse** 引擎 | 车道像素 **−23 ~ −56%**(密集模型仅 −1~−9%) |
| 工作站 TRT 8.6 / 数据中心 TRT 10.16 同 ONNX | 只复现 **−5%** |
| PyTorch 假 INT8 代理 | **完全不复现**(说明不是纯数值问题) |
| 二分定位 | **INT8+sparse 引擎完全崩(无车道像素)**,除非把**第一个 trunk stage(`stem`, `layer1`)保持 fp16**;只保 `layer2`/`layer3/4`/FPN/BEV decoder **无效**;整个 trunk 保 fp16 则完全一致 |

→ **根因**:`layer1` 的小通道高分辨率稀疏 INT8 kernel。
→ **部署修复**:INT8 构建时加 `--fp16-keep layer1`(或 `stem,layer1,layer2`);
→ **训练修复**:2:4 剪枝时排除这些 stage(`--sparse-exclude stem,layer1,layer2,...`)——反正那里稀疏收益很小。

### 2. lift-plugin 的表是 **rig-specific**(换车必须换表)

- `make_plugin_onnx.py` 会把投影(**pair tables**,来自 `dump_lift.py`,由**某一个场景的
  `K` / `T_cam_ego`** 推导)烘焙进计算图,**之后引擎不再读 `K` / `T_cam_ego` 输入**;
- 因此:plugin 引擎**只对建表时那套相机有效**。用另一辆车的场景 → 几何投影全错 → **车道位移、变细**;
- 上游实测:两个 demo 场景来自第二辆车(**内参差最多 250 px、相机位置差 0.5 m**),"Orin 车道变细"
  的报告**大部分是这个原因**,不是量化;
- **多 rig 方案**:① 每套标定各建一个引擎(各自 dump 表);② 或**用不带 plugin 的 plain ONNX**
  (运行时逐帧读 `K`/`T_cam_ego`,代价 **+4~5 ms** on Orin)。

> 🔴 **对我们的直接含义**:我们跑的是自采 6 相机车(`data_20260910_*`,标定在
> `tools/xcalib_calibration/`)。**任何部署/评测都必须用这辆车标定生成的 lift 表**;
> 拿别车的引擎评测会出现"车道莫名变细",别误判成量化问题。

### 3. 校准:只用真实帧,且 dtype 必须对齐

- **禁用随机张量校准**:噪声喂出来的激活范围在真实数据里不存在 → 量程完全错。默认 **96 帧、stride 4**;
- **循环输入要做真实 recurrence**:带 `hist_bev` 的引擎,校准前先用**同图 companion fp16 引擎**跑一遍,
  让它像运行时一样维护 device 端 BEV ring,再把它的 device 指针交给校准器;喂**零历史**会让时间路径
  校准在"只在场景开头出现"的张量上 → 某些层 scale = 0 → 直接 build 失败;
- **dtype 陷阱(最耗时的坑)**:companion 的 `hist_bev` buffer 是 **fp16**(I/O 格式),而网络按 ONNX
  声明读作 **fp32**。把 fp16 位当 fp32 解释 → 注入 ~1e9 垃圾 → `hist`/`tfuse` 的 scale 在 cache 里
  **大了 5×10⁷ 倍** → v95~v120 全部出现**自车轨迹冻结、静止标志恒定**,而 **BEV 分割和 3D 框看起来完全正常**;
  - 修复:`dtoh → astype(fp32) → htod`(在 host 端转换);
  - 规则:**喂给校准器的是网络读取的 dtype,不是指针持有的 dtype**;
  - 快速诊断:对比健康/损坏两次 build 的 `.calib` cache 里 per-tensor scale,元凶会以数量级差距跳出来。

### 4. 训练侧量化准备(低成本、非 QAT)

| 手段 | 作用 |
|---|---|
| `--quant-noise 1.0`(v45+) | 在 image-feature fuse 与 temporal BEV fuse 上挂 forward hook,加 **1 个 INT8 step 的逐通道均匀噪声**(`ch_absmax/127`);零参数,让融合特征对舍入更鲁棒 |
| fp16 溢出首先是**训练问题** | conv→BN 的尺度是自由的(BN 会除回来),loss 里没有任何东西把它拉回 1 → 漂移直到 fp16 激活超过 65504;有过一轮因此**静默丢弃 69% 的 step**而 val 数字依然"看起来合理";`renorm_convbn.py` 做函数保持的缩放(BEV mIoU 四位小数不变),`--bn-guard` + 非有限 loss 快照/恢复防复发 |
| `--ema-exclude seg_head.` | EMA 只平均参数,**BN running stats 保持原始**;高 LR 微调时二者漂移 → 2D seg 塌成单类,而 PointPainting 会把 2D seg 反注入 BEV → **2D 头死了 = BEV 更弱** |

---

## 6 相机适配清单(相对上游的改动,代码内以 `# [6cam]` 标记)

| 改动 | 上游 | 本适配版 |
|---|---|---|
| 相机列表 | 硬编码 8 个(含 `CAM_FRONT_NARROW` / `CAM_BACK_NARROW`) | **从每个场景的 `manifest.json` 的 `cams` 键动态读取**(6 相机 rig 无 NARROW 槽) |
| 校准显存 | `cuda.mem_alloc(8 * 3 * 432 * 768)` 等硬编码 | 新增 `--n-cams`(默认 **6**)按实际相机数分配 |
| 场景根目录 | `out/hf_stage/data`、`out/bevlane`、`/data/dataset/bevlane` | 默认 `meteor_6cam_lidar_deploy/scenes`,可用 `--scenes-root` / `METEOR_SCENES_ROOT` 覆盖 |
| 验证列表 | `val.lst`、`~/work/BevLane/val.lst` | 默认 `meteor_6cam_lidar_deploy/val_scenes.txt`,可用 `--val-list` / `METEOR_VAL_LIST` 覆盖 |
| 模型 | `MODELS["v52"]` 硬编码(仅 fake-INT8 代理) | 新增 `--model`(默认 `v52`) |
| 类别映射 | 不变 | `NAMES = {1:road, 3:crosswalk, 4:laneline, 5:stopline, 6:road_edge}`;fake-INT8 代理用训练口径 `ROAD=(1,3,4,5) / LANE=(2,6)` |

---

## 用法

```bash
# ① 工作站 TRT 8.6:建 fp16 参考引擎
python3 patches/meteor_deploy/int8_lane_local.py build \
    --onnx out/meteor_v157.onnx --out engine/int8_probe/v157_fp16.engine

# ② 建 INT8 引擎(layer1 留 fp16!校准用真实帧)
python3 patches/meteor_deploy/int8_lane_local.py build \
    --onnx out/meteor_v157.onnx --out engine/int8_probe/v157_int8.engine \
    --int8 --fp16-keep layer1 --calib 96 --cache engine/int8_probe/v157_int8.calib

# ③ 建 INT8+sparse(2:4)
python3 patches/meteor_deploy/int8_lane_local.py build \
    --onnx out/meteor_v157.onnx --out engine/int8_probe/v157_int8_sparse.engine \
    --int8 --sparse --fp16-keep layer1

# ④ 对比:逐类车道像素比 + IoU + 整体一致率
python3 patches/meteor_deploy/int8_lane_local.py probe \
    --ref engine/int8_probe/v157_fp16.engine \
    --eng engine/int8_probe/v157_int8.engine --tag v157_int8

# ⑤ TRT 10 版本(数据中心 GPU)
python3 patches/meteor_deploy/int8_lane_x86.py build --onnx out/meteor_v157.onnx \
    --out engine/v157_int8.plan --int8 --fp16-keep layer1 --n-cams 6

# ⑥ 无需 TensorRT:PyTorch 假 INT8 快速预判(需独占 GPU,勿在训练时跑)
python3 patches/meteor_deploy/probe_int8_lane.py \
    --ckpt checkpoints/meteor_custom_v52_st5ep_20260923/ep2.pt --tag ep2 --n-cams 6
```

**运行前提**
- `build` 只需要 TensorRT(+pycuda 或 cuda-python);
- `probe` 会 `import deploy.runtime.MeteorRT` → **必须在 METEOR 树内运行**(PYTHONPATH 含它);
- `probe_int8_lane.py` 需要 CUDA,且会在 GPU 上再放一份模型 → **不要在训练占卡时运行**。

---

## 来源与许可

- 上游:`https://github.com/tier4/METEOR.git`,分支 `vla-overlay-int8-probes`(PR #3),
  基线 commit `dc193a8`,Apache License 2.0(见 `../meteor_train/LICENSE-METEOR-Apache-2.0`);
- 本目录仅做 **6 相机适配** 与路径参数化,算法逻辑未改动;
- 完整 PR 评估见 [`../meteor_train/PR3_vla_overlay_review.md`](../meteor_train/PR3_vla_overlay_review.md);
- 本地只读克隆:`论文3/METEOR_pr_vla_overlay/repo`(用于 diff 与后续摘取)。
