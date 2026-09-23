# PR #3 评估笔记:`deploy: VLA overlay in the demo renderer, robust --stop, INT8 lane-thinning probes`

评估日期:2026-09-23 · 评估人:本会话(DeepSeek Harness) · 状态:**Open,未合并,无人 review**

## 一、PR 基本信息

| 项 | 值 |
|---|---|
| 仓库 / 分支 | `https://github.com/tier4/METEOR.git` → `vla-overlay-int8-probes` → `main` |
| 作者 | `dan-dnn`(PR 显示 1 参与者,无 reviewer/assignee/label) |
| 规模 | **6 commits · 13 files · +875 / −7** |
| merge-base(与本项目本地基线) | **`dc193a8`** —— 与我们本地 METEOR 完全同基线 ✅ |
| 本地只读克隆 | `论文3/METEOR_pr_vla_overlay/repo`(浅克隆 + deepen 100,可随时 `git diff origin/main...HEAD`) |

### 6 个 commit

| Commit | 主题 |
|---|---|
| `bf42270` | 主提交:VLA overlay in the demo renderer / robust `--stop` / INT8 lane-thinning probes |
| `c92df2f` | orin_render:overlay strip 也省略**国家名**(与天气/光照同理——BEV 不可观测) |
| `418dbae` | deploy/README:lift-plugin 表 **rig-specific**;int8_lane_local 的 `--plugin` 与校准缓存清洗 |
| `23df20c` | export:`--bev-tokens` 输出(fused BEV 池化 25×16);runtime:进程内/worker VLA hook |
| `c1814c5` | deploy/README:INT8+sparse 二分结果(layer1);runtime:VLA worker pool、record dump、更宽的 ungrounded-word 过滤 |
| `1aa3b02` | `closed_loop_eval`:`--list`(任意 held-out 场景列表)与 `--intent`(按训练口径从 GT 未来取路由意图) |

## 二、改动分两条工作线

### A. METEOR-VLA(视觉语言动作模型)集成 —— 新功能

| 文件 | 行数 | 作用 |
|---|---|---|
| `deploy/vla_live.py` | +154(新) | 同进程跑 VLA:读引擎 **`bev_tok`**(fused BEV 池化 25×16,第 13 个头)→ Qwen3-VL-8B + LoRA + BEV projector + 回归头 → 输出 `{cmd_in, v0, wp_reg, json, raw}` |
| `deploy/vla_server.py` | +34(新) | VLA worker 独立进程,TCP JSON-lines(`{"tok": base64 fp16 [96,25,16], "v0", "cmd"}`)→ 让一次 METEOR 前向同时喂 E2E 规划与 VLA |
| `deploy/export_onnx.py` | +14 | **新增 `--bev-tokens`**:追加 `bev_tok [B,96,25,16] fp16` 为尾部输出(VLA 的输入接口) |
| `deploy/orin_realtime.py` | +70 | 三种 VLA 模式:`METEOR_VLA_JSONL`(离线叠加)/ `METEOR_VLA_LIVE`(进程内)/ `METEOR_VLA_SERVER`(worker 池);+ `METEOR_VLA_DUMP` 记录 |
| `deploy/orin_render.py` | +114 | 渲染器:front tile 与 BEV 面板画**粉色 VLA 轨迹**(与绿色 E2E 并存)+ 深度行下方**推理条**(scene/hazards/rationale/decision) |
| `deploy/runtime.py` | +2 | runtime 接入 VLA hook |
| `deploy/orin/demo_public6.sh` | +2/−2 | **`--stop` 现在匹配任意 data root**(此前只匹配 public6,换 root 的 demo 停不掉) |
| `docs/QUICKSTART.md` | +1 | 环境变量表补 `METEOR_VLA_JSONL` |

### B. INT8 车道细化调查 —— 诊断工具 + 结论(部署价值最高)

| 文件 | 行数 | 作用 |
|---|---|---|
| `deploy/int8_lane_local.py` | +185(新) | 工作站 TRT 8.6 + pycuda:构建 fp16/INT8/INT8+sparse,`--fp16-keep` 二分,逐类 BEV 像素探针,校准缓存清洗 |
| `deploy/int8_lane_x86.py` | +121(新) | TRT 10 + cuda-python 版 |
| `bevlane/probe_int8_lane.py` | +140(新) | **纯 PyTorch 假 INT8 代理**(训练主机无 TRT 时用) |
| `deploy/README.md` | +30 | §6.x 记录三条结论(见下) |

**§6.x 的核心结论**(已提炼进 [`../meteor_deploy/README.md`](../meteor_deploy/README.md)):

1. **INT8+sparse 崩在 `layer1`**:Orin 上车道像素 −23~−56%,工作站只复现 −5%,假 INT8 不复现;二分定位到 `layer1` 小通道高分辨率稀疏 INT8 kernel → 修复:`--fp16-keep layer1`(部署)/ `--sparse-exclude stem,layer1,layer2`(训练);
2. **lift-plugin 表是 rig-specific**:表由某场景的 `K`/`T_cam_ego` 烘焙进图,引擎忽略这两个输入 → 换车必须换表,否则车道位移+变细(第二辆车内参差 250 px、位置差 0.5 m);多 rig 要么每车一表,要么用 plain ONNX(+4~5 ms);
3. **校准纪律**:只用真实帧(96 帧/stride 4);循环输入用 companion fp16 引擎做真实 recurrence;**dtype 陷阱**——fp16 buffer 被当 fp32 读会使 scale 大 5×10⁷ 倍,表现为"自车轨迹冻结 + 静止标志恒定,而 BEV seg/3D 框正常"。

### C. 其他

- `bevlane/closed_loop_eval.py` +15:`--list`(默认 `val.lst`;`out/heldout549.lst` = val+test 549 场景)与 `--intent`(从 GT 未来按训练口径 `|y@1.5-3s|>2m` 推 straight/left/right one-hot 喂给规划器;不加则与 demo 一样"无导航")。

## 三、对我们部署的价值评估

| 内容 | 价值 | 判断 |
|---|---|---|
| **§6.x INT8 三条结论** | 🔴 **高** | 直接指导我们的 Orin 部署:必须 `--fp16-keep layer1`、**必须按我们 6 相机标定建表**、校准 dtype 纪律 |
| **`int8_lane_local.py` / `int8_lane_x86.py` / `probe_int8_lane.py`** | 🔴 **高** | 现成的量化前后车道质量对比工具;已在 `../meteor_deploy/` 完成 6 相机适配 |
| **`demo_public6.sh` 的 `--stop` 修复** | 🟠 中 | 2 行纯收益(换 root 也能停 demo) |
| **`closed_loop_eval --list / --intent`** | 🟠 中 | `--intent` 让闭环评测带导航意图(与训练口径一致);`--list` 便于换评测集 |
| **`--bev-tokens` 输出** | 🟡 低(条件性) | 只有要接 VLA 才有用;否则是多余输出 |
| **VLA 集成(vla_live/vla_server/orin_render overlay)** | ⚪ **暂不需要** | 需要 Qwen3-VL-8B + LoRA 权重 + **VLA 训练产物**,我们目前没有这条训练线 |

**结论**:对我们的**部署**确有帮助,但价值几乎全部来自 **B(INT8 调查)**;A(VLA)是与我们无关的独立功能线。

## 四、建议动作(已执行 / 待定)

| # | 动作 | 状态 |
|---|---|---|
| 1 | INT8 三条防坑结论落到部署文档 | ✅ 已写入 `../meteor_deploy/README.md` |
| 2 | 三个 INT8 探针工具**适配 6 相机**并入库 | ✅ `../meteor_deploy/`(rig 从 manifest 动态读、`--n-cams`、路径参数化) |
| 3 | 摘取 `--stop` 修复(2 行) | ⏸ 待定(需改 `deploy/orin/demo_public6.sh`,我们当前无 demo 流程) |
| 4 | 摘取 `closed_loop_eval --list/--intent` | ⏸ 待定(评测增强,建议等训练收尾后做) |
| 5 | VLA 部分 | ❌ 不引入(无 VLA 训练产物与需求) |
| 6 | 整分支合并 | ❌ 不建议(冲突面大、A 线无用) |

## 五、冲突与风险提醒

- **`deploy/orin_render.py` 双方都改过**:上游加 overlay 绘制(+114),我们本地 `9702345` 加了
  `draw_occ_topview()`(OCC 顶视图)。两处**功能不重叠**,但若日后要合 PR,需**三方合并**;
- 上游 PR 基于 `dc193a8`,与我们本地 3 个提交同基线 → **理论上可干净地按文件摘取**,但不要
  `git merge` 整分支(会把 VLA 那套一起带进来假设我们不要);
- `int8_lane_*` 的 `probe` 子命令依赖 `deploy.runtime.MeteorRT` → 需在 METEOR 树内运行;
  `probe_int8_lane.py` 需独占 GPU(勿与训练同跑)。

## 六、复现本评估

```bash
# 只读克隆(已建好,浅+deepen 100)
cd 论文3/METEOR_pr_vla_overlay/repo
git log --oneline origin/main..HEAD                  # 6 个提交
git diff origin/main...HEAD --stat                   # 13 文件 +875/-7
git diff origin/main...HEAD -- deploy/README.md      # §6.x 结论
git show HEAD:deploy/int8_lane_local.py | head -60   # 工具用法
```
