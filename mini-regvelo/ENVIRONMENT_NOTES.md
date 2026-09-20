# 环境与验证记录

本文件记录代码实际运行的环境，以及那次运行的结果。它不属于上游教学材料。

## 1. 使用的环境

- Python 3.13.14（Windows x64）
- torch 2.14.0+cpu
- torchdiffeq 0.2.5
- numpy 2.5.3
- matplotlib 3.11.2
- pillow 12.3.0

复现环境的命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pillow
```

`pillow` 是 `run_validation.py` 需要的（它要解码导出的 PNG 检查完整性），
但没有列在 `requirements.txt` 里。

## 2. 如何运行

```powershell
.\.venv\Scripts\python.exe data.py
.\.venv\Scripts\python.exe transcription.py
.\.venv\Scripts\python.exe -m unittest -v test_core
.\.venv\Scripts\python.exe main.py --epochs 20 --output results_smoke   # 快速冒烟
.\.venv\Scripts\python.exe run_validation.py                            # 完整验证
```

## 3. 验证结果

`validation.json` 在这个环境下报告 `status: passed`，包含的检查项为
`core_numerical_checks`、`free_time_end_to_end`、`training_improves_and_solver_agrees`、
`network_time_smoke_30_epochs`、`plot_files_decode`。

`test_core.py` 中六个单元测试全部通过，其中包括解析解对照，
以及时间梯度与权重梯度的有限差分检验。

free 时间模式、训练 400 轮（`results/` 中保存的那次运行）：

| 指标 | 数值 |
|---|---|
| 重建 MSE 总和，第 1 轮 | 0.081504 |
| 重建 MSE 总和，第 400 轮 | 0.000201 |
| 学习时间与生成时间的 Pearson r | 0.9914 |
| W 的全部元素 RMSE | 0.4278 |
| 最大状态误差，48 步 vs 96 步 RK4 | 2.4e-07 |
| 最大状态误差，RK4 vs dopri5 | 6.0e-07 |

network 时间模式、30 轮冒烟运行：MSE 从 0.1753 降到 0.0067，时间 Pearson r 为 0.9916。

## 4. 这些图说明了什么（含一个负面结果）

对 `results/*.png` 的目视检查结论（也记录在 `validation.json` 中）：

- `loss.png` 在对数坐标下单调下降；总目标与重建 MSE 几乎重合，
  说明惩罚项在本次运行中可忽略。
- `reconstruction.png` 的六个面板中，ODE 曲线都穿过观测散点：
  拟合出的动力学很好地解释了观测。
- `latent_time.png` 显示学习到的时间相对生成时间是单调的，但并不相等：
  映射是非线性的，并在 `t_max` 附近饱和。也就是说，时间轴只能被确定到
  “相差一个单调重参数化”的程度。
- `grn.png` 是最重要的负面结果。三条真实调控边只找回了一条：

  | 边 | 生成用的 W | 学到的 W |
  |---|---|---|
  | A -> B | +1.20 | +0.46 |
  | A -> C | +0.60 | -0.04 |
  | B -> C | -0.80 | +0.03 |

  重建 MSE 已降到 2e-4，而三条真实边里有两条仍接近零。
  低重建误差**并不**意味着还原出了真实的基因调控网络；这正是
  `README.md` 第 7 节讨论的可辨识性问题。在论文的完整 RegVelo 中，
  这个问题由变分推断加上 Equation 6–8 的动力学正则共同处理，
  而本教学版把这些替换成了朴素的 MSE。
- `velocity.png` 与生成用的速度相关，但存在系统性偏差，
  且偏差随速度大小/时间尺度变化——这在上述时间重参数化的前提下是预期的。
- `perturbation.png` 显示删除 A 的调控列会压低 B（唯一被找回的那条边），
  而 A 与 C 不变；C 不变是因为学到的指向 C 的权重接近零。
  也就是说，模拟敲除效果的好坏取决于学到的调控网络，
  而不取决于模型把数据重建得多好。

`run_validation.py` 生成的 `results_network_smoke/` 目录被 `.gitignore` 排除在本仓库之外；
它是一次 30 轮的冒烟运行，不是收敛结果。
