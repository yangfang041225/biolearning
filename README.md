# biolearning

计算生物学方向的学习笔记与研究项目。

## 目录

### [mini-regvelo](mini-regvelo/)

对 **RegVelo: Gene-regulatory-informed dynamics of single cells**
（Cell 189, 3773–3800, 2026）的教学版最小实现，用来研究
“如何从静态的单细胞快照中反推出一个动力系统”。

包含内容：

- 完整的 mini-RegVelo 流程：输入观测 `(U_obs, S_obs)` 与先验图 `G_prior`，
  输出潜时间、耦合的 RNA 常微分方程、重建损失，并据此更新参数。
- 六个单元测试：解析解对照、时间梯度与权重梯度的有限差分检验、
  先验掩码（硬约束）、以及敲除逻辑的方向性。
- 一次 400 轮训练的完整记录：指标、训练历史、六张诊断图，以及训练好的模型参数。
- [`mini-regvelo/ENVIRONMENT_NOTES.md`](mini-regvelo/ENVIRONMENT_NOTES.md)：
  运行环境、验证结果，以及六张图能说明和不能说明什么。
- [`mini-regvelo/README.md`](mini-regvelo/README.md)：逐个模块说明代码的作用、
  每个文件背后的数学，并明确列出本教学版相对论文原版做了哪些简化。

**记录在案的这次运行，核心结论是**：模型几乎完美地重建了观测数据
（每个矩阵的重建 MSE 约 1.6e-4），却只找回了三条真实调控边中的一条。
**重建误差低，并不能作为“已识别出真实调控网络”的证据。**
