# 万达动态行人验证

本文记录在 Habitat-GS 万达场景中加载 SMPL-X GaussianAvatar、沿 NavMesh 回放行人动作并进行双人胶囊碰撞验证的工作流。代码依赖 Habitat-GS 的本地扩展、CUDA、SMPL-X 模型和万达全质量场景资产；这些大文件、上游源码快照和生成视频不进入本仓库。

## 已提交实现

- `src/dynamic_pedestrians.py` 提供同步帧 SMPL-X 代理胶囊最小表面净距计算，以及对 Avatar driver 矩阵/胶囊形状的校验。
- `tests/test_dynamic_pedestrians.py` 覆盖胶囊相交、分离、端点最近点、零长度胶囊、逐帧检测及帧数不匹配。
- `.scratch/wanda-gamma/` 保存本机端到端实验脚本、输入路线和验证产物，不属于正式接口。需要复现端到端渲染时，参照该目录 README 的命令和实验环境。

## 实验结论与边界

本轮使用 Habitat-GS 发布的 `scene01.driver.pkl` 步态重定向到两条手选 NavMesh 路线，没有运行 GAMMA 新推理，也没有实现通用自动多人路线采样器。每个 driver 包含 322 帧、55 个关节矩阵和 54 个代理胶囊；两人同步胶囊检测的最小表面净距为 6.820 m。该结果只对本次路线、动作时间对齐和代理胶囊模型成立。

原生 Avatar 在 161/161 帧产生人体前景 RGB-D 差异，且可观察到人物遮挡关系。背景高斯存在模糊/拉伸，人物在镜头中偏小；实验仅验证 Avatar 接入、动作播放和局部遮挡，不代表背景渲染质量或任意多人自动规划已经验收。

## 本地复现要求

实验依赖以下本机资源，路径需按环境调整：

- Habitat-GS Python 环境及 `habitat_sim`、GaussianAvatar 扩展。
- Habitat-GS 上游源码中的 `tools_gs/generate_trajectory.py`。
- SMPL-X 模型目录及其规范授权数据。
- 万达全质量场景的高斯点云、碰撞网格和 NavMesh。
- Habitat-GS 发布的参考 driver；实验脚本会对该输入校验 SHA-256。

重定向脚本根据 waypoint JSON 烘焙 driver、URDF 和报告；渲染入口支持单人、双人及无人对照，并输出视频、关键帧和 RGB-D 比较。详细复现命令、资产位置和参数保存在被 Git 忽略的 `.scratch/wanda-gamma/README.md`，其中产物应保留在 scratch，不要复制到版本控制。
