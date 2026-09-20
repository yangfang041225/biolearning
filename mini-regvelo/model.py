"""组合时间和 ODE：输入 U,S:(N,G)，输出重建、时间和速度。

对应 Figure 1A 和 Equation 4；共享时间和固定步长是教学选择。
"""
import torch
from torch import nn
from torchdiffeq import odeint
from dynamics import RNADynamics
from latent_time import LatentTime


def solve_at_times(dynamics, times, steps=48, method="rk4"):
    """从零初值积分到每个 times[n]，输出 (N,2G)。

    换元 t=r*t_n，r 从 0 到 1。链式法则给出 dx/dr=t_n*F(x)。
    此处一行是同一动力系统在不同终点时间的求解副本，不是独立参数模型。
    这个写法不需要排序时间，也保留了 times 的梯度。
    """
    if times.ndim != 1 or steps < 1:
        raise ValueError("times 必须是一维张量，steps 必须为正。")
    if not torch.isfinite(times).all().item() or (times < 0).any().item():
        raise ValueError("积分时间必须有限且非负。")
    x0 = times.new_zeros((times.numel(), 2 * dynamics.num_genes))

    def scaled_rhs(r, x):
        return times[:, None] * dynamics(r * times, x)

    interval = times.new_tensor([0.0, 1.0])
    if method == "rk4":
        solution = odeint(scaled_rhs, x0, interval, method="rk4",
                           options={"step_size": 1.0 / steps})
    elif method == "dopri5":
        solution = odeint(scaled_rhs, x0, interval, method="dopri5",
                           rtol=1e-6, atol=1e-8)
    else:
        raise ValueError("支持 rk4 或 dopri5。")
    # solution:(2,N,2G)，只保留 r=1 的终点。
    return solution[-1]


class MiniRegVelo(nn.Module):
    def __init__(self, data, t_max=4.0, time_mode="free", steps=48):
        super().__init__()
        self.dynamics = RNADynamics(data["G_prior"])
        self.latent_time = LatentTime(data["U_obs"], data["S_obs"], t_max, time_mode)
        self.steps = steps

    def forward(self, U_obs, S_obs):
        times = self.latent_time(U_obs, S_obs)
        x = solve_at_times(self.dynamics, times, self.steps)
        G = self.dynamics.num_genes
        u, s = x[:, :G], x[:, G:]
        return {"U_pred": u, "S_pred": s, "time": times,
                "velocity": self.dynamics.spliced_velocity(u, s)}
