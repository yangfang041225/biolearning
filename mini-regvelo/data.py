"""教学版 Mini-RegVelo：观测数据与先验图。

这是教学简化，不是原论文完整实现。
对应 STAR Methods 的 RegVelo generative process 和 Equation 3。
本文件不做近邻平滑、缩放、时间推断或 ODE 积分。
"""

import torch


def prepare_data(U_obs, S_obs, G_prior, gene_names):
    """转换并检查数据，返回包含四个命名字段的字典。

    U_obs、S_obs: (N, G)，行是细胞，列是基因。
    G_prior: (G, G)，G_prior[g, j] = 1 表示允许 j -> g。
    gene_names: G 个不同的基因名称。

    三个矩阵的基因顺序必须一致，U_obs 和 S_obs 的细胞顺序
    也必须一致。仅靠矩阵数值无法自动验证这些顺序。
    输入应是已准备好的非负丰度；这里不把原始 counts 自动变成
    论文所用的近邻平滑、按基因缩放的数据。
    """
    # float32 是后续神经网络和 ODE 计算使用的浮点数类型。
    # copy + detach 让观测保持固定，不参与参数学习。
    U_obs = torch.as_tensor(U_obs, dtype=torch.float32).detach().clone()
    S_obs = torch.as_tensor(S_obs, dtype=torch.float32).detach().clone()
    G_prior = torch.as_tensor(G_prior, dtype=torch.float32).detach().clone()

    if U_obs.ndim != 2 or S_obs.ndim != 2:
        raise ValueError("U_obs 和 S_obs 必须是二维矩阵，shape 为 (N, G)。")
    if U_obs.shape != S_obs.shape:
        raise ValueError("U_obs 和 S_obs 的 shape 必须相同。")

    N, G = U_obs.shape
    if N == 0 or G == 0:
        raise ValueError("至少需要一个细胞和一个基因。")
    if G_prior.shape != (G, G):
        raise ValueError("G_prior 的 shape 必须为 (G, G)。")

    for name, matrix in [("U_obs", U_obs), ("S_obs", S_obs)]:
        if not torch.isfinite(matrix).all().item():
            raise ValueError(f"{name} 不能包含 NaN 或无穷大。")
        if (matrix < 0).any().item():
            raise ValueError(f"{name} 的 RNA 丰度不能为负。")

    is_binary = (G_prior == 0) | (G_prior == 1)
    if not is_binary.all().item():
        raise ValueError("G_prior 只能包含 0 或 1。")

    if isinstance(gene_names, str):
        raise ValueError("gene_names 应为名称列表，不能是一个字符串。")
    gene_names = list(gene_names)
    if len(gene_names) != G:
        raise ValueError("gene_names 的长度必须等于基因数 G。")
    if any(not isinstance(name, str) or not name.strip() for name in gene_names):
        raise ValueError("每个基因名称必须是非空字符串。")
    if len(set(gene_names)) != G:
        raise ValueError("基因名称不能重复。")

    return {
        "U_obs": U_obs,
        "S_obs": S_obs,
        "G_prior": G_prior,
        "gene_names": gene_names,
    }


def make_format_example():
    """手写格式示例：不是论文数据，也不是 ODE 合成训练数据。

    四行代表四个细胞；行顺序不表示时间顺序。
    先验包含 A -> B 和 B -> C，但不指定激活或抑制。
    """
    U_obs = [
        [0.20, 0.05, 0.01],
        [0.70, 0.40, 0.10],
        [0.40, 0.15, 0.03],
        [0.50, 0.60, 0.40],
    ]
    S_obs = [
        [0.10, 0.02, 0.00],
        [0.60, 0.25, 0.05],
        [0.30, 0.08, 0.01],
        [0.80, 0.65, 0.30],
    ]

    #               调控基因（列）
    #                 A  B  C
    G_prior = [
        [0, 0, 0],  # 靶基因 A
        [1, 0, 0],  # 靶基因 B：允许 A -> B
        [0, 1, 0],  # 靶基因 C：允许 B -> C
    ]
    return prepare_data(U_obs, S_obs, G_prior, ["A", "B", "C"])


# 直接执行 python data.py 时才运行这段示例。
# 以后其他文件 import data 时，不会自动打印。
def make_synthetic_data(n_cells=64, t_max=4.0, noise_std=0.01, seed=7):
    """从已知 ODE 生成单轨迹 snapshots；返回 data 和独立的 truth 字典。

    教学简化：3 个基因、任意丰度单位、非负截断高斯噪声。
    不执行论文的平滑/缩放，不声称噪声等于论文观测模型。
    truth 只用于评估，绝不传给训练器。
    """
    from dynamics import RNADynamics, inverse_softplus
    from model import solve_at_times

    if n_cells < 4 or t_max <= 0 or noise_std < 0:
        raise ValueError("至少 4 个细胞、正的时间上限、非负噪声。")
    rng = torch.Generator().manual_seed(seed)
    prior = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    system = RNADynamics(prior)
    W = torch.tensor([[0., 0., 0.], [1.2, 0., 0.], [0.6, -0.8, 0.]])
    beta = torch.tensor([1.0, 0.8, 1.2])
    gamma = torch.tensor([0.7, 0.9, 0.6])
    b = torch.tensor([-0.5, -1.0, 0.2])
    with torch.no_grad():
        system.transcription.W_raw.copy_(W)
        system.transcription.b.copy_(b)
        system.raw_beta.copy_(inverse_softplus(beta))
        system.raw_gamma.copy_(inverse_softplus(gamma))
        times = torch.linspace(0.04 * t_max, 0.96 * t_max, n_cells)
        # 打乱所有细胞的顺序，同时保持 U、S、真值时间按相同行排列。
        times = times[torch.randperm(n_cells, generator=rng)]
        # 用自适应高精度求解生成数据，训练则使用固定步长 RK4。
        clean = solve_at_times(system, times, method="dopri5")
        observed = (clean + noise_std * torch.randn(clean.shape, generator=rng)).clamp_min(0)
    data = prepare_data(observed[:, :3], observed[:, 3:], prior, ["A", "B", "C"])
    truth = {"time": times, "W": W, "b": b, "beta": beta, "gamma": gamma,
             "U_clean": clean[:, :3], "S_clean": clean[:, 3:],
             "velocity": system.spliced_velocity(clean[:, :3], clean[:, 3:]).detach()}
    return data, truth


if __name__ == "__main__":
    data = make_format_example()
    for key in ["U_obs", "S_obs", "G_prior"]:
        print(f"{key}: shape={tuple(data[key].shape)}")
        print(data[key])
    print("gene_names:", data["gene_names"])
    print("先验允许的边：A -> B，B -> C；此处没有指定边的正负。")
