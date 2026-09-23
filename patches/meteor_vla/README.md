# METEOR-VLA 拓展归档(视觉语言动作模型的部署侧集成)

本目录归档上游 PR #3(`tier4/METEOR` 分支 `vla-overlay-int8-probes`)中 **VLA(Visual-Language-Action)
大模型集成**的全部部署侧代码,作为**未来的能力拓展**保留。

> **当前状态:不在生产链路上。** 我们还没有 VLA 的训练产物与训练侧代码(见"前置条件"),
> 因此本目录只做资产归档 + 启用路径说明,不参与现有推理/训练流程。
> INT8 量化调查(同一 PR 的另一条线)已单独归档在 [`../meteor_deploy/`](../meteor_deploy/)。
> 完整 PR 评估见 [`../meteor_train/PR3_vla_overlay_review.md`](../meteor_train/PR3_vla_overlay_review.md)。

---

## 一、它是什么

在 demo 渲染器里,**并行展示第二条轨迹**:METEOR 引擎照常输出绿色 E2E 规划,同时把引擎的
`bev_tok`(fused BEV 池化到 **25×16**,第 13 个头)喂给一个 **VLA 大模型**(Qwen3-VL-8B +
LoRA + BEV projector + 回归头),由它给出**粉色轨迹 + 文字推理条**(scene / hazards /
rationale / decision)。

关键设计:**一次 METEOR 前向同时喂 E2E 规划和 VLA** —— 不需要第二次感知前向。

数据流:

```
相机 ×6 / LiDAR ──> METEOR 引擎(fp16/INT8) ──┬──> E2E 规划(绿)
                                             └──> bev_tok [B,96,25,16] fp16
                                                        │
                                        BEVProjector(MLP 96→1024→4096 + 位置编码)
                                                        │
                                    Qwen3-VL-8B-Instruct + LoRA ──> 文本(JSON)
                                                        │
                                        回归头 ──> wp_reg(6×2 航点,粉)
                                                        │
                             orin_render VLA overlay(粉色路径 + 推理条)
```

VLA 输出的记录格式(与离线 overlay 一致):
```json
{"cmd_in": "...", "v0": 8.0, "wp_reg": [[x,y]×6], "json": {"scene": "...", "hazards": "...",
 "rationale": "...", "command": "keep_lane|turn_left|turn_right|stop"}, "raw": "..."}
```
> 注意:天气/光照/国家名等**从 BEV 特征不可观测**的内容被**主动过滤**(上游 `c92df2f`),
> 避免模型编造。

## 二、目录内容

| 文件 | 来源(上游) | 说明 |
|---|---|---|
| `vla_live.py` | `deploy/vla_live.py`(+154,新) | VLA 运行时:`BEVProjector` + Qwen3-VL + LoRA + 回归头;含 `VLALive` / `VLAClient` / `VLAPool` |
| `vla_server.py` | `deploy/vla_server.py`(+34,新) | VLA worker 进程:TCP JSON-lines(`{"tok": base64 fp16 [96,25,16], "v0", "cmd"}`)→ overlay 记录 |
| `vla_overlay.patch` | 5 文件 +197/−4 | 其余部署侧改动 diff(见下表) |

`vla_overlay.patch` 覆盖的文件:

| 文件 | 行数 | 改动 |
|---|---|---|
| `deploy/orin_render.py` | +114 | 6 个新函数:粉色 VLA 路径投影(`_draw_vla_path_cam`)、文字推理条(`_draw_vla_strip`)、换行与**不可观测内容过滤**(`_wrap` / `_has_ungrounded` / `_filter_scene` / `_filter_rationale`) |
| `deploy/orin_realtime.py` | +70 | 三种 VLA 接入模式 + `METEOR_VLA_DUMP` 记录 + 用 GT 未来推导航命令喂 VLA |
| `deploy/export_onnx.py` | +14 | **`--bev-tokens`**:追加 `bev_tok [B,96,25,16] fp16` 尾部输出 |
| `deploy/runtime.py` | +1/−1 | 输出名列表加 `bev_tok` |
| `docs/QUICKSTART.md` | +1 | 环境变量表加 `METEOR_VLA_JSONL` |

## 三、启用前置条件(⚠️ 缺一不可)

| # | 条件 | 状态 |
|---|---|---|
| 1 | **VLA 训练产物** `ckpt_last.pt`(Qwen3-VL + LoRA + projector + 回归头) | ❌ 我们没有(PR 不含训练侧代码) |
| 2 | VLA 训练侧代码(数据 dump 与训练脚本) | ❌ 不在本 PR 中(PR 只含部署侧) |
| 3 | 依赖库:`transformers`(AutoModelForImageTextToText)、`peft`(LoRA) | ⚠️ 需装入部署环境 |
| 4 | 模型权重 `Qwen/Qwen3-VL-8B-Instruct`(HF 下载,~16GB)+ `HF_HOME` 缓存目录 | ⚠️ 需下载 |
| 5 | 引擎带 `bev_tok` 输出 → 导出时用 `--bev-tokens` | ✅ 有 patch |
| 6 | GPU 显存:8B 模型 + METEOR 引擎同机(推荐 **worker 进程隔离**,上游即为此提供 `vla_server.py`) | ⚠️ 需评估(Orin 上资源紧张) |

**结论**:要真正跑起来,必须先补 **#1 + #2(VLA 训练线)**,这不是部署侧能解决的。

## 四、启用方式(补全前置条件后)

```bash
# 方式 1:离线叠加 —— 只需一份逐帧 JSONL,不需要在设备上跑 VLA(最轻)
METEOR_VLA_JSONL=/path/vla_records.jsonl  <demo 启动命令>

# 方式 2:进程内 —— VLA 与引擎同一进程(简单,但 8B 模型吃显存)
METEOR_VLA_LIVE=/path/vla_v17/ckpt_last.pt \
METEOR_VLA_MODEL=Qwen/Qwen3-VL-8B-Instruct \
METEOR_VLA_EVERY=10 \
HF_HOME=~/hf_cache  <demo 启动命令>

# 方式 3:worker 池(推荐)—— VLA 跑在独立 Python 环境,通过 TCP 通信
HF_HOME=~/hf_cache python3 deploy/vla_server.py --ckpt out/vla_v17/ckpt_last.pt --port 5771
METEOR_VLA_SERVER=127.0.0.1:5771  <demo 启动命令>

# 可选:把每帧 VLA 记录 dump 成 JSONL(供离线复看/回归测试)
METEOR_VLA_DUMP=/path/vla_dump.jsonl  <demo 启动命令>
```

环境变量一览:`METEOR_VLA_JSONL`(离线)/ `METEOR_VLA_LIVE`(ckpt 路径)/ `METEOR_VLA_SERVER`
(`host:port`,可逗号分隔多 worker)/ `METEOR_VLA_MODEL`(默认 `Qwen/Qwen3-VL-8B-Instruct`)/
`METEOR_VLA_EVERY`(默认 10 帧一次)/ `METEOR_VLA_DUMP` / `HF_HOME`。

## 五、如果将来要启用:建议路径

1. **先走方式 1(离线 JSONL)**:用外部算力(工作站/云)跑 VLA,设备端只做叠加渲染 —— 零设备负担,
   风险最低;
2. 若要**上设备实时**:用方式 3(worker 隔离),并**先测显存与延迟**(8B 模型 + Orin 资源紧张);
3. **补训练线**:VLA 需要 `bev_tok` → 文本/航点的配对数据(训练侧 dump 与训练脚本需另行获取),
   建议先确认上游是否有对应仓库/分支;
4. **不要整分支 merge**:`vla_overlay.patch` 中 `deploy/orin_render.py` 与**我们本地的
   `draw_occ_topview()`**(commit `9702345`)会冲突 —— 需**手工三方合并**(功能不重叠:
   他们加 VLA 路径与文字条,我们加 OCC 顶视图面板)。

## 六、来源与许可

- 上游:`https://github.com/tier4/METEOR.git`,分支 `vla-overlay-int8-probes`(PR #3),
  基线 commit `dc193a8`,作者 `dan-dnn`,Apache License 2.0
  (见 [`../meteor_train/LICENSE-METEOR-Apache-2.0`](../meteor_train/LICENSE-METEOR-Apache-2.0));
- 文件为**原样归档**(未做 6 相机适配 —— 该 VLA 路径按 `CAMS` 顺序投影,若启用需按我们的
  6 相机 rig 核对 `_draw_vla_path_cam` 的相机索引);
- 本地只读克隆:`论文3/METEOR_pr_vla_overlay/repo`。
