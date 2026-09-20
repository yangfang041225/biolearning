"""教学版转录预测器：alpha = softplus(W @ s + b)。

对应论文 STAR Methods Equation 2 和硬约束 GRN 定义。
这是教学版模块，不是原论文完整实现；这里只实现硬约束模式。
"""

import torch
from torch import nn
from torch.nn import functional as F


class TranscriptionPredictor(nn.Module):
    """把当前 spliced RNA 状态映射为转录速率。

    输入 s: (G,) 或 (B, G)，浮点张量。
    输出 alpha: 与 s 相同的 shape。
    行 g 是靶基因，列 j 是调控基因：W[g, j] 对应 j -> g。
    输入需与模型处于同一设备、使用同一浮点类型。
    """

    def __init__(self, G_prior):
        # nn.Module 帮我们管理可训练参数；先初始化这个父类。
        super().__init__()

        prior = torch.as_tensor(G_prior, dtype=torch.float32).detach().clone()
        if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
            raise ValueError("G_prior 必须是方阵，shape 为 (G, G)。")
        if prior.shape[0] == 0:
            raise ValueError("至少需要一个基因。")
        if not ((prior == 0) | (prior == 1)).all().item():
            raise ValueError("G_prior 只能包含 0 或 1。")

        self.num_genes = prior.shape[0]

        # buffer 是模型携带的固定数据，不是优化器要学习的参数。
        # 它会随模型一起保存，也会随 model.to(...) 一起移动设备。
        self.register_buffer("G_prior", prior)

        # nn.Parameter 告诉 PyTorch：这些数值需要计算梯度并可被更新。
        # W 从零开始与论文描述一致；不是把二值先验当作初始权重。
        self.W_raw = nn.Parameter(torch.zeros_like(prior))

        # 为方便教学，将 b 初始化为零；这不是原论文的偏置初始化。
        # b=0 时初始 alpha=softplus(0)=log(2)，不是零转录。
        self.b = nn.Parameter(prior.new_zeros(self.num_genes))

    def effective_weights(self):
        """返回实际用于计算的 W；先验外的边严格为零。"""
        # 每次重新计算，保留 W_raw -> W 的梯度路径。
        return self.W_raw * self.G_prior

    def forward(self, s):
        """计算转录速率；调用 model(s) 时 PyTorch 会执行本函数。"""
        if s.ndim not in (1, 2) or s.shape[-1] != self.num_genes:
            raise ValueError("s 的 shape 必须为 (G,) 或 (B, G)。")

        W = self.effective_weights()

        # @ 是矩阵乘法，.T 是转置。
        # 单状态也可写 W @ s；这里统一使用最后一维为基因的写法。
        # 批量情况：(B,G) @ (G,G) -> (B,G)。
        # b 的 (G,) 会自动加到每一行，这叫 broadcasting（广播）。
        regulatory_input = s @ W.T + self.b

        # 数值稳定的 softplus，不直接手写 log(1 + exp(x))。
        # 不 detach s：以后积分求梯度需要保留状态对参数的依赖。
        return F.softplus(regulatory_input)


if __name__ == "__main__":
    from data import make_format_example

    data = make_format_example()
    model = TranscriptionPredictor(data["G_prior"])

    # 为解释方向，手动设置两个示例权重。这不是训练结果。
    # no_grad 仅用于手动赋值；真正的 forward 不应放在此块内。
    with torch.no_grad():
        model.W_raw[1, 0] = 2.0   # A 激活 B
        model.W_raw[2, 1] = -1.0  # B 抑制 C

    s = torch.tensor([0.3, 0.4, 0.2])
    alpha = model(s)
    print("W (rows=targets, columns=regulators):")
    print(model.effective_weights())
    print("s:", s)
    print("alpha:", alpha)
    # W@s+b = [0, 0.6, -0.4]，alpha 约为 [0.6931, 1.0375, 0.5130]。

    alpha_batch = model(data["S_obs"])
    print("Batch alpha shape:", tuple(alpha_batch.shape))  # (4, 3)
    # 此处使用观测状态仅演示函数调用。
    # 真正积分时，求解器会传入当前模拟状态 s(t)。
