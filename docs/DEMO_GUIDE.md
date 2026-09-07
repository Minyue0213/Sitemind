# SiteMind 演示说明

## 当前唯一故事线

SiteMind 面向挖掘机作业现场的局部移位辅助：操作人员先在相机画面中选择“下一作业点 A”，系统用真实标定把目标锁定到三维世界坐标；相机识别地面语义，LiDAR 判断坡度、台阶和障碍，两类证据进入同一 BEV 风险地图；规划器从挖掘机当前位置持续重规划。没有连续安全通道时，系统显示保守停车，而不是强行给出路线。

主演示固定为五栏：

1. 现场相机与目标 A。
2. 相机—LiDAR 真实标定投影。
3. 相机语义 BEV。
4. LiDAR 几何 BEV。
5. 位姿对齐的 7 帧融合风险 BEV、挖掘机当前位置与规划路线。

## 最终交付成果

1. `artifacts/sitemind_next_worksite_demo.gif`：推荐主演示，约 29.8 秒；149 帧中 144 帧有路线，并保留一段连续 5 帧的保守停车。
2. `artifacts/sitemind_next_worksite_demo.json`：主演示逐帧规划记录。
3. `artifacts/sitemind_next_worksite_final.gif`：完整技术版，约 42.6 秒；212 帧中 183 帧有路线、29 帧保守停车。
4. `artifacts/sitemind_next_worksite_final.json`：完整技术版逐帧记录。
5. `artifacts/sitemind_next_worksite_check_64.png`：保守停车样例，证明系统会在地图不连通时拒绝冒险。

## 技术证据图

- `artifacts/evaluation_scorecard.png`：407 帧验证集的训练前后指标，重点展示 ALICE 挖掘机平台。
- `artifacts/model_comparison_frame0207.png`：同一相机画面的原模型、微调模型和不确定性兜底对比。
- `artifacts/goose3d_geometry_70.png`：真实 LiDAR 点云生成的高程、坡度与几何风险。
- `artifacts/real_planning_demo.png`：同一真实几何地图上的最短路线与风险感知路线对比。
- `artifacts/calibrated_fusion_demo.png`：真实 CameraInfo/TF 标定下，相机语义和 LiDAR 几何进入统一 BEV 的完整链路。
- `artifacts/alice_seq02_temporal_comparison.gif`：单帧与位姿对齐 7 帧短时记忆对比，展示地图连通性和路线稳定性改善。

建议按“模型看得准 → 两种传感器落到同一地图 → 时序更稳定 → 选点后持续规划/必要时停车”的顺序展示。旧固定坐标目标、并行双证据故事图、Hybrid A* 和车宽红色膨胀结果不再作为当前技术主线。

## 本机一键重建静态技术图

```bash
.venv/bin/python scripts/build_presentation_assets.py
```

该命令只重建成绩卡、真实 LiDAR 规划对比和标定融合图，不需要租 GPU。

## 重新构建主演示

完整 231 帧缓存目前保存在 AutoDL 数据盘。服务器开机且代码同步后，可在远端执行：

```bash
python scripts/render_sequence_preview.py \
  artifacts/alice_seq02_sequence_full_temporal \
  artifacts/alice_seq02_sequence_full \
  artifacts/alice_seq02_sequence_full_fusion \
  artifacts/alice_seq02_sequence_full/sequence.json \
  --goal-frame 19 \
  --goal-pixel 286 1063 \
  --start-frame 19 \
  --count 149 \
  --fps 5 \
  --output artifacts/sitemind_next_worksite_demo.gif
```

已有分割、融合和时序缓存时，重渲染不需要 GPU；只有重新推理模型时才需要显卡。

## 项目简介

> SiteMind 让操作人员在相机中选择下一作业点，再用真实标定把相机语义和 LiDAR 几何写入同一张俯视风险地图。系统从挖掘机当前位置滚动规划局部移位路线；当已观测区域没有连续安全通道时，它会保守停车，避免把未知区域误判为安全。

## 必须诚实说明的边界

- SiteMind 当前是环境智能与局部安全决策软件模块，不宣称已经实现整机无人驾驶或轨迹跟踪控制。
- 2D 定量结果来自未参与训练的 GOOSE-Ex validation；Sequence02 的 5 张精标图片属于训练集，只用于验证真实融合工程链路。
- 407 帧 validation 精简包没有相机—LiDAR 外参，因此全量评估分别衡量 2D 识别和 3D 规划；真实像素级融合使用 Sequence02 原始 ROS bag 中的 `CameraInfo`、`/tf` 和 `/tf_static`。
- 当前路线是约 8–12 m 的滚动局部规划段，表示向下一作业点移位，不等于视频中的挖掘机已经沿路线执行。
- 规划器为 risk-aware A*；未采用展示效果和工程完成度不足的车宽膨胀与 Hybrid A* 版本。
