"""运行合成数据教学实验，并保存全部数值结果与图。

输入命令行参数；输出 results/ 下的 JSON、NPZ、checkpoint 和 PNG。
这不是论文复现，也不把拟合良好等同于真实 GRN 的唯一恢复。
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from data import make_synthetic_data
from model import MiniRegVelo, solve_at_times
from train import fit
from perturb import compare_perturbation
from visualize import save_plots


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    data, truth = make_synthetic_data(args.cells, args.t_max, seed=args.seed)
    model = MiniRegVelo(data, args.t_max, args.time_mode, args.steps)
    # 训练器只获得观测与先验，无法读取 truth。
    history = fit(model, data, args.epochs, args.lr, args.regularization)
    with torch.no_grad():
        pred = model(data["U_obs"], data["S_obs"])
        W = model.dynamics.transcription.effective_weights()
        beta, gamma = model.dynamics.rates()
        grid = torch.linspace(0, args.t_max, 120)
        curves = compare_perturbation(model.dynamics, grid, args.ko, steps=2 * args.steps)
        cells_ko = compare_perturbation(model.dynamics, pred["time"], args.ko,
                                       steps=2 * args.steps)
        # 更细步长与另一算法用于检查积分精度，不用于偷偷改变训练结果。
        x = torch.cat([pred["U_pred"], pred["S_pred"]], dim=1)
        refined = solve_at_times(model.dynamics, pred["time"], steps=2 * args.steps)
        adaptive = solve_at_times(model.dynamics, pred["time"], method="dopri5")
        mse = ((pred["U_pred"] - data["U_obs"]) ** 2).mean()
        mse += ((pred["S_pred"] - data["S_obs"]) ** 2).mean()

    def arr(value):
        return value.detach().cpu().numpy()

    metrics = {
        "initial_reconstruction_mse_sum": history[0]["mse_u"] + history[0]["mse_s"],
        "final_reconstruction_mse_sum": mse.item(),
        "time_pearson_r": float(np.corrcoef(arr(truth["time"]), arr(pred["time"]))[0, 1]),
        "W_rmse_all_entries": float(torch.sqrt((W - truth["W"]).square().mean())),
        "max_state_error_step_doubling": float((x - refined).abs().max()),
        "max_state_error_vs_dopri5": float((x - adaptive).abs().max()),
        "note": "Training reconstruction only. No posterior uncertainty; parameters not uniquely identifiable.",
    }
    numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
    if not np.isfinite(numeric).all():
        raise RuntimeError("评估出现非有限指标。")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    with (folder / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez(folder / "results.npz", U_obs=arr(data["U_obs"]), S_obs=arr(data["S_obs"]),
             G_prior=arr(data["G_prior"]), gene_names=np.array(data["gene_names"]),
             U_pred=arr(pred["U_pred"]), S_pred=arr(pred["S_pred"]),
             time=arr(pred["time"]), true_time=arr(truth["time"]), W=arr(W),
             true_W=arr(truth["W"]), b=arr(model.dynamics.transcription.b),
             beta=arr(beta), gamma=arr(gamma), velocity=arr(pred["velocity"]),
             true_velocity=arr(truth["velocity"]),
             KO_U=arr(cells_ko["KO"]["U"]), KO_S=arr(cells_ko["KO"]["S"]),
             KO_velocity=arr(cells_ko["KO"]["velocity"]), grid_time=arr(grid),
             WT_grid_S=arr(curves["WT"]["S"]), KO_grid_S=arr(curves["KO"]["S"]))
    torch.save({"state_dict": model.state_dict(), "config": vars(args),
                "U_obs": data["U_obs"], "S_obs": data["S_obs"],
                "G_prior": data["G_prior"], "gene_names": data["gene_names"]}, folder / "model.pt")
    save_plots(folder, data, truth, pred, history, W, grid, curves, data["gene_names"][args.ko])
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved to: {folder.resolve()}", flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Educational Mini-RegVelo, not a paper reproduction")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--t-max", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-mode", choices=["free", "network"], default="free")
    parser.add_argument("--ko", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--output", default=str(Path(__file__).parent / "results"))
    run(parser.parse_args())
