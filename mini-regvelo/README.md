# Mini-RegVelo：从状态点反推动力系统

**这是教学简化，不是原论文完整实现，也不是论文结果复现。**

主要依据：*RegVelo: Gene-regulatory-informed dynamics of single cells*, Cell 189,
3773–3800 (2026)，DOI: https://doi.org/10.1016/j.cell.2026.04.022。
下文 PDF 页码指用户提供的 73 页文件，包含封面与补充材料。

## 1. 这个程序的数学本质

**一句话：手里只有一堆没有时间戳的状态点，要反推出生成它们的动力系统，以及每个点落在轨迹的哪一时刻。**

### 1.1 先把问题写成数学

状态是每个细胞的 RNA 丰度，共 2G 维：

```
x = [u; s] ∈ R^(2G)        u = 未剪接 RNA，s = 已剪接 RNA，各有 G 个基因
```

动力学由参数 θ = (W, b, β, γ) 完全决定：

```
dx/dt = F(x; θ),   x(0) = 0

其中   α(s) = softplus(W·s + b)          ← 转录速率，由当前 s 决定
      du/dt = α(s) − β ⊙ u
      ds/dt = β ⊙ u − γ ⊙ s
```

（`⊙` 是逐元素乘法。）给定一组 (θ, t)，这套 ODE 有唯一解，把解算子记为 Φ：

```
x(t) = Φ_t(θ)        "从零状态积分到时刻 t 得到的状态"
```

### 1.2 本项目做的事，就是把它反过来解

这里有两样东西是被**拿掉**的：

- **θ 未知** —— 调控关系、速率都不知道；
- **时间未知** —— 每个细胞没有采样时刻 t_n。

手里的东西只有 N 个静态点：

```
(U_n, S_n) ∈ R^(2G),   n = 1, 2, ..., N
```

它们是一次快照，**彼此之间没有先后顺序**。要解的是：

```
min over  θ, t_1, ..., t_N      Σ_n ‖ Φ_{t_n}(θ) − (U_n, S_n) ‖²  +  λ‖W‖²
```

也就是：**把 θ 和 t_n 一起调，直到 ODE 的轨迹能穿过这些点。**

这就是全文的骨架。其余所有代码 —— 自动微分、神经网络、先验掩码 —— 都只是为解这个式子服务的工具。

### 1.3 它与"正常解 ODE"的根本区别

- **正问题**（你已经会的）：θ 已知 → 积分 → 得到轨迹。
- **反问题**（本项目）：只有一列散点 → 反推 θ **和** 每个点对应的时刻。

反问题的难处在于：**内层每算一次目标函数，就要解一遍 ODE。** 传统做法需要解一遍伴随方程（adjoint）才能拿到梯度。这里把整个积分过程搭在自动微分上，`loss.backward()` 就能沿着

```
损失 → 重建状态 → ODE 积分 → (θ, t_n)
```

这条链路一次性把梯度回传。

所以本项目里出现的 `nn.Parameter`、`torchdiffeq`、`backward()`、`optimizer.step()`，本质上只做一件事：**让这个梯度算得出来，然后沿梯度下降。** 换成任何其他能求这个梯度的办法，数学都是同一个数学。

```
   未知：θ = (W, b, β, γ)   和   t_1 ... t_N
        │
        ▼
   dx/dt = F(x; θ),  x(0) = 0            ← 正问题：解 ODE
        │
        ▼
   x(t_n) = Φ_{t_n}(θ)
        │
        ▼
   与观测 (U_n, S_n) 比对
   loss = Σ_n ‖ Φ_{t_n}(θ) − (U_n, S_n) ‖²  +  λ‖W‖²
        │
        ▼
   loss.backward()                        ← 自动微分把梯度送回 θ 和 t_n
        │
        └──────────► 更新 (θ, t_n)，回到第一行，循环
```

### 1.4 生物学在这里只做了一件事：给 θ 加约束

如果把 θ 当成 18 个自由数字去硬拟合 384 个观测值，问题会严重欠约束、解也不唯一。所以论文（和本项目）用先验知识把 θ 卡住：

| 约束 | 数学形式 | 作用 |
|---|---|---|
| 转录速率必须是"当前状态的函数" | α = softplus(W·s + b)，而不是任意函数 | 把基因间的调控关系真正写进动力学 |
| 只有先验允许的调控边可以非零 | W = W_raw ⊙ G_prior，先验之外恒为 0 | 直接排除大量错误假设 |
| 转录速率必须为正 | softplus(·) > 0 | 保证物理意义；激活/抑制的符号由 W 的正负承担 |
| 速率 β、γ 必须为正 | β = softplus(raw_beta) | 同上，用"无约束参数 + 正映射"来实现 |

这就是"**用先验缩小搜索空间**"，也是这个逆问题在生物学上还有希望解出来的原因。

### 1.5 抽象掉生物学之后，它是什么

翻译成物理语言：**这是一个系统辨识（system identification）问题。**

> 你有同一个钟摆在**不同相位**被拍下的一叠照片，照片上没有拍摄时刻；你要同时反推出摆长和阻尼，以及每张照片是第几秒拍的。

再换一个更贴近的说法：

> 普通的曲线拟合，是横坐标已知、求曲线的形状；**这里是横坐标也未知，要和曲线形状一起求。**

多出来的这一层，就是"把散点排到时间轴上"的隐式排序问题 —— 也正是本项目里 `latent_time.py` 存在的唯一理由。

### 1.6 本例的具体规模

| 量 | 含义 | 取值 |
|---|---|---|
| G | 基因数 | 3 |
| 2G | 状态维度 | 6 |
| N | 细胞数 = 状态点数 | 64 |
| θ 的有效自由参数 | W 中 3 个（其余 6 个被先验置零）+ b 3 个 + β 3 个 + γ 3 个 | 12 |
| 每个细胞的时刻 | t_1 ... t_N | 64 |
| **未知量合计** | | **76** |
| 观测值 | U_obs 与 S_obs 共 2NG 个数 | 384 |

76 个未知、384 个观测，表面上"绰绰有余"。但**解仍然不唯一**：把所有速率同时乘 c、时间同时除以 c，轨迹可以完全不变。因此本项目分别报告重建误差、时间相关性、W 误差与积分精度，绝不用"MSE 很小"去宣称"GRN 找回来了"。这一点在第 8 节展开。

### 1.7 训练完之后，到底得到了什么

回到"这个程序要干什么"。解开上面那个优化问题之后，四个产物：

| 产物 | 数学上是什么 | 生物学上意味着什么 |
|---|---|---|
| **GRN** | 拟合出的 W | 谁调控谁，是激活还是抑制 |
| **隐时间** | 拟合出的 t_n | 每个细胞在发育轨迹上的位置（给散点排了序） |
| **RNA 速度** | dx/dt = ds/dt 在观测点处的值 | 每个细胞接下来往哪个方向走 |
| **扰动预测** | 把 W 的某一列置零后重新积分 | 敲除某个基因，下游会发生什么 |

一句话概括全部：**把一张没有时间轴的散点图，还原成一个能解释、能预测、能扰动干预的动力学模型。**

## 2. 先运行什么

建议 Python 3.11 或 3.12；使用 CPU 即可。进入本目录后，在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe data.py
.\.venv\Scripts\python.exe transcription.py
.\.venv\Scripts\python.exe -m unittest -v test_core
.\.venv\Scripts\python.exe main.py
```

不需要激活虚拟环境。首次安装 PyTorch 需要下载依赖；运行时间取决于电脑。
运行后查看 `results/metrics.json`、六张 PNG、`history.csv`、`results.npz` 和 `model.pt`。
首次先用 `--epochs 20` 理解流程，再用默认 400 轮训练。

```powershell
.\.venv\Scripts\python.exe main.py --epochs 400 --time-mode free
.\.venv\Scripts\python.exe main.py --epochs 600 --time-mode network --output results_network
```

`free` 是建议先读的模式。`network` 是确定性编码器版本，并不是论文 VI。
若由 Codex 自动创建环境，环境可能位于任务的 `work/mini-regvelo-env`；这不影响上述独立运行方式。

## 3. 论文内容与教学选择

| 内容 | 论文 | 本项目 |
|---|---|---|
| 观测 | 平滑、按基因 min-max 缩放的 RNA 丰度 | 3 基因合成单轨迹、任意丰度单位，不做平滑或缩放 |
| 初始条件 | u(0)=s(0)=0 | 相同；观测矩阵不是初值 |
| 转录 | softplus(Ws+b) | 保留 |
| 时间 | gene-cell-specific t_ng | 每个细胞共享 t_n |
| 编码 | q_phi(z\|u,s)，潜变量采样 | 直接学习时间；可选确定性 2 维编码器 |
| 网络先验 | 硬约束或软约束 | 只实现硬约束 |
| 训练 | 负 ELBO 加先验和动力学正则 | MSE_U+MSE_S+lambda*sum(W²) |
| 求解 | torchode / dopri5 | torchdiffeq / 固定步长 rk4；dopri5 用于生成及精度核对 |
| 优化 | AdamW | Adam，显式权重惩罚 |
| 偏置初始化 | 论文描述的初始化方案 | b=0；此时 alpha=log(2)，不是零 |
| 不确定性 | 后验速度分布等 | 未实现；不把单次预测当作后验 |
| 扰动 | 删除 TF 调控列，再积分，结合 CellRank | 保留删除列与重积分，不实现命运概率 |

论文对应位置：Figure 1A（PDF 4）；Equation 2–3（PDF 34）；Equation 4、初值与生成过程（PDF 35）；
Equation 5–8 与目标函数（PDF 36）；速度、GRN 表示、扰动（PDF 38–39）。

## 4. 需要的 Python 概念

- `tensor` 是带数据类型和 shape 的数值数组；`(N,G)` 表示 N 行 G 列。
- `nn.Module` 管理模型中的参数和子模块；调用 `model(...)` 会执行 `forward(...)`。
- `nn.Parameter` 标记可训练数值；定义参数不会自动训练它。
- `self` 指当前模型对象，例如 `self.b` 是这个对象保存的偏置。
- `*` 是逐元素乘法，`@` 是矩阵乘法，`.T` 是转置。
- `x[:, :G]` 取所有行的前 G 列；`x[:, G:]` 取后 G 列。
- `torch.cat([u,s], dim=-1)` 沿最后一维拼接。
- `t[:, None]` 将 `(N,)` 变成 `(N,1)`，便于逐行乘状态导数。
- broadcasting（广播）使 `(G,)` 的速率向量自动作用于 `(N,G)` 每一行。
- `detach()` 切断梯度连接；可以用于保存结果，不能用于训练中的预测或时间。
- `torch.no_grad()` 用于生成固定数据、评估或手动赋值，不包住训练 forward。

## 5. 每个模块先看数学，再看代码

### data.py：定义和生成观测

数学作用：存储静态状态点；可选从已知动力系统生成它们。
输入 `prepare_data(U_obs,S_obs,G_prior,gene_names)`；输出同名字段的字典。
U/S 为 `(N,G)`，prior 为 `(G,G)`，名称长度 G。
必要性：保证矩阵方向、数据类型和细胞对应关系正确。
对应论文生成过程的数据定义与 Equation 3。

`G_prior[g,j]=1` 表示允许 j→g。先验只给结构，不给权重或正负。
`make_format_example()` 是前两步用的手写格式数据，不能当作 ODE 训练真值。
`make_synthetic_data()` 才生成训练数据：设置 W,b,beta,gamma，零初值积分，加入小噪声，打乱细胞顺序。
返回的 `truth` 包含时间、网络、无噪声状态和速度，**只用于评估，训练器不接收 truth**。
合成数据可能超过 1；这里有意保留同一动力系统的丰度单位，不伪装成论文预处理。

### transcription.py：当前状态决定转录速率

数学作用：alpha=softplus(Ws+b)。输入 s 为 `(G,)` 或 `(B,G)`，输出同 shape。
W 为 `(G,G)`，b 为 `(G,)`。必要性：把基因之间的依赖放入动力学；对应 Equation 2。

```python
W = self.W_raw * self.G_prior
regulatory_input = s @ W.T + self.b
alpha = F.softplus(regulatory_input)
```

对单个列向量，数学是 W@s；对逐行存放的多个状态，代码是 s@W.T。
softplus 使 alpha 为正；b 可以为负。无调控输入时，真正的基础速率是 softplus(b)。
模型不会预先把观测 S 转成固定 alpha；ODE 每次求导都传入当前模拟的 s(t)。

### dynamics.py：瞬时导数

数学作用：du/dt=alpha-beta*u，ds/dt=beta*u-gamma*s。
输入 t 和 x，x 最后一维是 `[u_1,...,u_G,s_1,...,s_G]`，shape `(B,2G)`。
输出 dx/dt，shape 不变；beta/gamma 的 shape 为 `(G,)`。
必要性：求解器只能推进一个已知导数函数；对应正文动力学和 Equation 4。

`rates()` 用 softplus 将 raw 参数映射到正速率。初始 beta=gamma=1。
不对 W 加正值限制，因为负权重代表抑制。
如果 W 只有零项或对角项，各基因子系统互相独立；跨基因边使系统耦合。

### latent_time.py：在哪里读取状态

输入 U/S `(N,G)`；输出共享时间 `(N,)`；必要性：snapshots 没有已知采样时刻。
与 Figure 1A 时间分支对应，但这里是教学替代。

`free`：每个训练细胞一个 raw_time，t=Tmax*sigmoid(raw_time)。它不能直接泛化到新细胞，
也不能在不同时重排时间参数的情况下重新排列训练数据。时间初始化只用观测总丰度。
“丰度越大越晚”只作为本例初始化启发，绝非普遍生物规律。

`network`：拼接输入 `(N,2G)` → Linear/Tanh → 低维 z `(N,2)` → Linear → `(N,1)`
→ sigmoid → t `(N,)`。Linear 实现可学习的矩阵乘法与偏置，Tanh 加非线性。
这里 z 是确定值，不是 q_phi 的样本。它增加了共享参数，但没有实现完整 VI。

### model.py：可微积分与重建

输入 U/S `(N,G)`；输出 U_pred/S_pred/velocity `(N,G)` 和 time `(N,)`。
必要性：把参数与时间转换为可以跟观测比较的量；对应 Figure 1A、Equation 4。

为每个细胞在不同 t_n 处读取同一系统，令 t=r*t_n：

```
dx/dr = t_n * F(x),  r in [0,1],  x(r=0)=0
```

在 r=1 时，得到原系统在 t_n 的状态。此换元在当前自治系统中是精确数学等价，
不是更换 ODE。数值计算仍有离散误差。每一行使用相同参数，不是每个细胞单独学习 W。
若将来加入显式时间依赖，则必须正确传入各行的物理时间。

所有积分以零状态开始；observations 仅用于时间推断与损失。
默认 48 个 RK4 步，物理步长为 t_n/48；通过普通 odeint 的运算图反向传播。
没有实现 adjoint，因本例很小。torchdiffeq 的 rk4 使用四阶 3/8 规则，
不要假设它与所有教材 RK4 的系数完全相同。

### train.py：反问题优化

输入 model 与 data；输出每轮 loss 等标量的列表。
数学目标：MSE_U+MSE_S+lambda*sum(W²)。必要性：使正问题的输出解释观测。
与论文参数联合优化思想对应，但**不是 ELBO，也不是论文完整正则化**。

```python
optimizer.zero_grad()  # 清除旧梯度
prediction = model(U_obs, S_obs)  # 推断时间并积分
loss = ...             # 重建误差与惩罚
loss.backward()        # 求损失对所有参数的导数
optimizer.step()       # 用导数调整参数
```

backward 不会更新参数；step 才更新。梯度沿着损失→ODE→时间和动力学参数回传。
代码检查非有限损失/梯度，裁剪梯度范数，并恢复所见目标最小的一组参数。
这里全批量训练，N 很小；没有验证集，所以训练图不证明泛化能力。

### perturb.py：改系统，再积分

输入已学习 dynamics、TF 索引、times `(N,)`；输出 WT/KO 的 U/S/velocity `(N,G)`。
必要性：分析某个调控因子缺失时系统的模型预测；对应 Methods 的 perturbation section。
复制模型后，`W_raw[:, tf_index]=0`；其他参数和比较时间保持不变。

不是把观测 S 的一列改成 0，也不会把 TF 自身 RNA 强制清零。
这里从零初值重新模拟整个 KO 发育过程，不模拟“中途注射药物”的瞬时干预。
在相同 u,s 上 ds/dt=beta*u-gamma*s 不直接含 W，所以即刻 spliced velocity 不变。
重新积分后，alpha 改变 u，再影响 s，从而改变后续 velocity。

### visualize.py 与 main.py

visualize 输入真实运行结果，输出六张 PNG；main 串联以上模块并保存结果。
图的输入 shape 已由模型定义，热图是 `(G,G)`，轨迹为 `(120,G)`，训练损失为每轮一个标量。
这是教学诊断，不是论文图重绘。

## 6. 图应该怎么读

- `loss.png`：重建误差能否下降；不是 ELBO 曲线。
- `latent_time.png`：已知合成时间与学习时间；不要求点严格落在对角线。
- `grn.png`：生成 W 与拟合 W，统一色标；行是 target，列是 regulator。
- `reconstruction.png`：观测散点与 ODE 拟合，横轴为学习时间。
- `velocity.png`：真值和学习的 ds/dt；大小受时间尺度影响。
- `perturbation.png`：在相同时间网格比较已学习系统的 WT/KO，实线/虚线可区分。

图中 a.u. 表示任意单位；没有物理小时单位。KO 图不是与真实 KO 实验的验证。
NPZ 保存逐细胞结果，行顺序与 U_obs/S_obs 完全相同。

## 7. 如何换成自己的数据

先在理解合成例子的基础上，再替换数据。下面只展示接口，不做真实 scRNA 数据预处理：

```python
import numpy as np
from data import prepare_data
from model import MiniRegVelo
from train import fit

arrays = np.load("my_data.npz")
data = prepare_data(arrays["U_obs"], arrays["S_obs"], arrays["G_prior"],
                    arrays["gene_names"].tolist())
model = MiniRegVelo(data, time_mode="free")
history = fit(model, data)
```

三个矩阵必须对齐基因顺序，两个观测矩阵必须对齐细胞顺序。
核心模型可接受 3 或 4 个基因；当前示例生成器、命令行 KO 范围与诊断布局固定为 3 基因。
不要直接将真实的多分支发育数据交给本版并期待复现论文结果。

## 8. 反问题为什么不保证找回真参数

若所有动力学速率同时乘 c（包括 alpha），时间除以 c，状态轨迹可能不变。
对 softplus 参数化，不应把 W,b 简单乘 c 当作严格等价变换，但速率/时间的辨识问题仍存在。
固定 t_max 只是尺度约定，不足以证明所有参数唯一。
因此项目分别报告重建 MSE、时间相关性、W 误差与积分精度，不以低 MSE 宣称恢复真 GRN。

固定参数、固定初值的确定性自治 ODE 只有一条解轨迹；本例只用于单轨迹。
论文的 gene-specific 时间允许更灵活的观测组合，不能将本模型当作其多过程完整表达。
真实表达数据还存在先验错误、噪声、隐变量、模型偏差等问题，本例不处理这些问题。

## 9. 验证与求解器

`test_core.py` 检查：常转录且 beta=gamma=1 的解析解；时间梯度和权重梯度的有限差分；
先验外零梯度；KO 的方向和原模型不变；两种时间模式的梯度；合成数据可复现和行对齐。
`main.py` 额外比较 48/96 步与 dopri5 的解，差异写入 metrics.json。
若积分误差大，增加 `--steps` 并重新训练；若损失发散，降低 `--lr`。

torchode/torchdiffeq 是库名，dopri5/rk4 是算法名。本项目 API 依据
[torchdiffeq 官方说明](https://github.com/rtqichen/torchdiffeq)。
原论文使用 torchode 的 dopri5，而本例为可读性和固定计算量使用 torchdiffeq 的 rk4。
安装版本与运行指标应以 `validation.json`、`environment.txt` 和结果文件为准；
没有这些文件或状态不是 passed 时，不应把代码当作已经通过端到端验证。

## 10. 后续恢复论文结构的顺序

先理解 free 时间版本，再看确定性 network 版本，最后才加入 q_phi(z|u,s)、
重参数化采样、gene-cell-specific 时间、Gaussian likelihood、KL、完整正则与不确定性。
这些扩展没有被本项目悄悄用 MSE 替代后宣称已完成。
