"""耦合 RNA ODE：状态 x=[u,s]，最后一维长度为 2G。

数学：du/dt=softplus(Ws+b)-beta*u，ds/dt=beta*u-gamma*s。
输入：时间标量和 (...,2G) 状态；输出：同 shape 的导数。
作用：为求解器提供瞬时变化率，而不是预测完整轨迹。
对应论文正文动力学方程、STAR Methods Equations 2、4。
"""
import torch
from torch import nn
from torch.nn import functional as F
from transcription import TranscriptionPredictor


def inverse_softplus(value):
    """正数 -> 未约束参数；使 softplus(返回值) 等于输入。"""
    return value + torch.log(-torch.expm1(-value))


class RNADynamics(nn.Module):
    def __init__(self, G_prior):
        super().__init__()
        self.transcription = TranscriptionPredictor(G_prior)
        self.num_genes = self.transcription.num_genes
        ones = self.transcription.b.new_ones(self.num_genes)
        # raw 参数可以取任意实数，实际速率始终为正。
        self.raw_beta = nn.Parameter(inverse_softplus(ones))
        self.raw_gamma = nn.Parameter(inverse_softplus(ones))

    def rates(self):
        return F.softplus(self.raw_beta), F.softplus(self.raw_gamma)

    def forward(self, t, x):
        # t 是求解器接口的一部分；本模型不显式依赖 t（自治系统）。
        G = self.num_genes
        u, s = x[..., :G], x[..., G:]
        beta, gamma = self.rates()
        alpha = self.transcription(s)
        du = alpha - beta * u
        ds = beta * u - gamma * s
        return torch.cat([du, ds], dim=-1)

    def spliced_velocity(self, u, s):
        beta, gamma = self.rates()
        return beta * u - gamma * s
