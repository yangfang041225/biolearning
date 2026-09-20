"""修改调控系统再积分；对应 Methods 的 Simulation of perturbation effects。

输入 dynamics、TF 列索引和时间 (N,)；输出 WT/KO 状态和速度 (N,G)。
不重新拟合时间或其他参数，比较相同时间位置；不实现 CellRank。
"""
import copy
import torch
from model import solve_at_times


def knockout_dynamics(dynamics, tf_index):
    if not 0 <= tf_index < dynamics.num_genes:
        raise ValueError("TF 索引越界。")
    ko = copy.deepcopy(dynamics)
    # 只改副本，保留原 WT 模型。删列代表删除 TF 对所有靶基因的影响。
    with torch.no_grad():
        ko.transcription.W_raw[:, tf_index] = 0.0
    return ko


@torch.no_grad()
def compare_perturbation(dynamics, times, tf_index=0, steps=96):
    ko = knockout_dynamics(dynamics, tf_index)
    result = {}
    G = dynamics.num_genes
    for label, system in [("WT", dynamics), ("KO", ko)]:
        x = solve_at_times(system, times, steps)
        u, s = x[:, :G], x[:, G:]
        result[label] = {"U": u, "S": s,
                         "velocity": system.spliced_velocity(u, s)}
    result["delta_S"] = result["KO"]["S"] - result["WT"]["S"]
    result["delta_velocity"] = result["KO"]["velocity"] - result["WT"]["velocity"]
    return result
