"""教学简化：每个 cell 一个共享 t_n，而非论文的 t_ng。

输入 U,S:(N,G)，输出 t:(N,)；目的：确定 ODE 在哪里读取状态。
free 模式直接学习 N 个参数；network 模式使用确定性低维编码器。
两者均不实现 q(z|u,s)、采样、KL 或 ELBO。
对应 Figure 1A 的时间分支，但并非论文完整时间推断模块。
"""
import torch
from torch import nn


class LatentTime(nn.Module):
    def __init__(self, U_obs, S_obs, t_max=4.0, mode="free"):
        super().__init__()
        if t_max <= 0 or mode not in ("free", "network"):
            raise ValueError("t_max 必须为正；mode 为 free 或 network。")
        self.mode = mode
        self.t_max = float(t_max)
        N, G = U_obs.shape
        if mode == "free":
            # 仅利用观测初始化，不读取合成真值时间。
            # 总丰度作为粗略起点只适合本例，不是一般生物学规律。
            score = (U_obs + S_obs).mean(dim=1)
            fraction = 0.05 + 0.90 * (score - score.min()) / (
                score.max() - score.min()
            ).clamp_min(1e-6)
            self.raw_time = nn.Parameter(torch.logit(fraction).detach().clone())
        else:
            # 先拼接成 (N,2G)，编码成 (N,2)，再解码成 (N,1)。
            self.encoder = nn.Sequential(nn.Linear(2 * G, 16), nn.Tanh(), nn.Linear(16, 2))
            self.decoder = nn.Linear(2, 1)

    def forward(self, U_obs, S_obs):
        if self.mode == "free":
            if U_obs.shape[0] != self.raw_time.numel():
                raise ValueError("free 模式每个训练细胞一个参数，不能直接预测新细胞。")
            raw = self.raw_time
        else:
            z = self.encoder(torch.cat([U_obs, S_obs], dim=1))
            raw = self.decoder(z).squeeze(-1)
        return self.t_max * torch.sigmoid(raw)
