# SimpleTrack 替换可行性报告(A/B 对比)

日期:2026-09-22 · 数据:`meteor_6cam_lidar_deploy/scenes` 场景 1/2 的 FocalFormer bev_box(同一份检测输入)

## 结论

SimpleTrack(ICRA 2022,`mot_3d` 库)替换自研 `MOTTracker3D` **可行且明显更优**:
轨迹持久性(中位长度 3~4.4 倍)、碎片率(降 55~68%)、tvalid 覆盖(+2~2.4pp)全面领先;
输出格式与旧版逐字段兼容(`boxes/count` 逐帧逐字节一致),仅追加向后兼容的 `track_ids`。

## 关键数字

| 指标 | 场景1 legacy → SimpleTrack | 场景2 legacy → SimpleTrack |
|---|---|---|
| 唯一轨迹数 | 1290 → 594 (-54%) | 1171 → 621 (-47%) |
| 平均轨迹长度(帧) | 22.7 → 49.3 (2.2×) | 16.3 → 30.7 (1.9×) |
| 中位轨迹长度(帧) | 5.0 → 22.0 (4.4×) | 3.0 → 9.0 (3.0×) |
| p90 轨迹长度(帧) | 75.3 → 140.1 | 44.0 → 86.0 |
| 单帧碎片率 | 24.0% → 7.6% | 32.8% → 14.3% |
| 持久轨迹(≥10帧)占比 | 92.0% → 97.5% | 88.3% → 94.3% |
| tvalid 覆盖(有未来航点比例) | 0.600 → 0.624 | 0.575 → 0.596 |
| 航点加速度中位数 (m/s²) | 0.11 → 0.12 | 0.12 → 0.13 |
| 航点加速度 p95 (m/s²) | 0.94 → 1.37 | 1.83 → 4.41 |
| 运行耗时(CPU,整场景) | 1.4s → 108s | 0.6s → 39s |

说明:
- `boxes/count` 2993/2971 帧全部逐字节一致(排序规则"按 ego 距离近→远"保持);
- `traj/tvalid` 变化即改进本身(不同 ID 分配 → 更完整的未来轨迹);
- p95 加速度升高是"遮挡期轨迹被 SimpleTrack 保留、重识别后位置跳变"的完整 GT 表现
  (旧版该场景直接 tvalid=0),中位数无变化,不是噪声恶化;
- 运行慢 80 倍源于 GIoU 的 shapely 多边形运算(每帧 ≤45 框,全场景 1~2 分钟,纯 CPU 可接受)。

## 配置(高召回 GT 取向)

```
asso=giou(bipartite, 阈值1.5) · motion_model=kf · max_age=10 · min_hits=1
redundancy=mm(两阶段关联) · 全帧 is_key_frame · 检测合成 score=1.0
```

## 产物

- `tools/SimpleTrack-main/venv_simpletrack/` — py3.10 独立 venv(filterpy/numba/shapely/pyquaternion)
- `tools/SimpleTrack-main/` — mot_3d 库(只读依赖)
- `scripts/autolabel_agent_traj/simpletrack_mot.py` — 接入模块(bev_box→world→FrameData→track ids)
- `scripts/autolabel_agent_traj/build_agent_traj_simpletrack.py` — 新 step3 引擎
- `scripts/autolabel_agent_traj/compare_mot_trackers.py` — A/B 对比脚本
- `scenes/*/agent_traj_st/` — A/B 试运行输出(未动 manifest/正式 agent_traj)

## Phase 4(已完成):全量切换 SimpleTrack

1. ✅ 6 场景重跑 `build_agent_traj_simpletrack.py --out-subdir agent_traj`(覆盖正式输出 + manifest,count/boxes 与旧版逐字节一致)
2. ✅ `run_phase3_full_production.sh` 第 3 步切换为新引擎(旧脚本保留备查)
3. ✅ 抽查:长轨道跨 771 帧持续、跨遮挡保持 ID;tvalid 覆盖 0.53~0.69/场景

## Phase 5(已完成):T4 导出改用真实 track id

- `export_t4_instance.py` 从 `agent_traj/*.npz` 读取 `track_ids`,instance token 由
  **行号 `{scene}_track_{k+1:04d}` → 真实轨道 `{scene}_track_{cls}_{tid:05d}`**
  (SimpleTrack 按类别独立计数,故 token 组合类别保证全局唯一;旧 npz 无
  `track_ids` 键时自动回退行号行为)。
- 顺带修复:bbox 字段强制 `float()` 转换(np.float32 无法 JSON 序列化,
  `round(np.float32)` 保留 float32 导致 json.dump 崩溃)。
- 重跑 6 场景后对比(annotation_legacy/ 保留旧版):

| 场景 | 旧版实例 | 新版实例 | anns/实例 | ≥5标注实例 | 2D兜底占比 |
|---|---|---|---|---|---|
| 061820 | 110 | 939 | 46.9 | 595 | 6.9% |
| 062659 | 82 | 838 | 33.1 | 467 | 6.8% |
| 063822 | 57 | 736 | 30.0 | 410 | 5.2% |
| 064331 | 44 | 553 | 22.7 | 263 | 4.3% |
| 073823 | 100 | 983 | 43.8 | 671 | 6.1% |
| 074912 | 45 | 589 | 46.8 | 306 | 4.6% |

- 旧版 token 是"每帧按距离排序的行号"(最多 64 个 3D token),把每帧第 k 近的
  不同物体错误合并为同一实例;新版 token = 真实 SimpleTrack 轨道,
  跨帧/跨相机/跨遮挡恒定(抽查 146 个轨道 0 冲突,全部注册于 instance.json)。
