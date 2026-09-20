"""将训练结果画成静态诊断图，所有输入均为当前运行实际结果。

输入 data、truth、prediction、训练历史与 WT/KO 曲线；输出 PNG。
目的：检查拟合、时间、速度、网络和扰动；不是论文 Figure 的复现。
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后台保存文件，不依赖图形窗口。
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, INK = "#2874A6", "#D48628", "#30343B"
SIGNED = LinearSegmentedColormap.from_list("signed", [BLUE, "#FFFFFF", ORANGE])


def array(tensor):
    return tensor.detach().cpu().numpy()


def save_plots(folder, data, truth, prediction, history, W, grid_times, curves, tf_name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK})
    N = data["U_obs"].shape[0]
    names = data["gene_names"]

    def finish(fig, name, title):
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(folder / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["epoch"] + 1 for h in history], [h["loss"] for h in history], color=BLUE,
            label="Total training objective")
    ax.plot([h["epoch"] + 1 for h in history],
            [h["mse_u"] + h["mse_s"] for h in history], color=INK,
            linestyle="--", label="Reconstruction MSE sum")
    ax.set(xlabel="Epoch", ylabel="Loss", yscale="log")
    ax.legend()
    finish(fig, "loss.png", f"Training loss | synthetic data, N={N}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(array(truth["time"]), array(prediction["time"]), color=BLUE, s=20)
    maximum = float(grid_times[-1])
    ax.plot([0, maximum], [0, maximum], color=INK, linestyle="--", label="Equal time")
    ax.set(xlabel="Known synthetic time (arbitrary units)",
           ylabel="Learned shared time (arbitrary units)", xlim=(0, maximum), ylim=(0, maximum))
    ax.legend()
    finish(fig, "latent_time.png", f"Latent time comparison | N={N}; scale is not identified")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    matrices = [array(truth["W"]), array(W)]
    limit = max(1e-8, max(float(np.abs(m).max()) for m in matrices))
    for ax, matrix, title in zip(axes, matrices, ["Generating W", "Learned W"]):
        ax.imshow(matrix, cmap=SIGNED, vmin=-limit, vmax=limit)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
               xlabel="Regulator (column)", ylabel="Target (row)", title=title)
        for g in range(3):
            for j in range(3):
                ax.text(j, g, f"{matrix[g,j]:+.2f}", ha="center", va="center",
                        color="white" if abs(matrix[g,j]) > 0.65 * limit else INK)
    finish(fig, "grn.png", "Regulatory weights | signed values; identical color scale")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, (obs_key, pred_key, label) in enumerate([
        ("U_obs", "U_pred", "Unspliced"), ("S_obs", "S_pred", "Spliced")
    ]):
        order = np.argsort(array(prediction["time"]))
        for g in range(3):
            ax = axes[row, g]
            ax.scatter(array(prediction["time"]), array(data[obs_key])[:, g],
                       facecolors="none", edgecolors=BLUE, s=19, label="Observed")
            ax.plot(array(prediction["time"])[order], array(prediction[pred_key])[order, g],
                    color=INK, label="ODE fit")
            ax.set(title=f"{names[g]}: {label}", xlabel="Learned time (a.u.)",
                   ylabel="Abundance (a.u.)")
            if row == 0 and g == 0:
                ax.legend()
    finish(fig, "reconstruction.png", f"Observed and reconstructed abundance | N={N}; training cells")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for g, ax in enumerate(axes):
        true_v, pred_v = array(truth["velocity"])[:, g], array(prediction["velocity"])[:, g]
        lo, hi = min(true_v.min(), pred_v.min()), max(true_v.max(), pred_v.max())
        ax.scatter(true_v, pred_v, color=BLUE, s=16)
        ax.plot([lo, hi], [lo, hi], "--", color=INK)
        ax.set(title=names[g], xlabel="Generating ds/dt (a.u.)", ylabel="Learned ds/dt (a.u.)")
    finish(fig, "velocity.png", f"Spliced velocity comparison | N={N}; time-scale dependent")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, (key, label) in enumerate([("S", "Spliced abundance (a.u.)"),
                                         ("velocity", "Spliced velocity (a.u.)")]):
        for g, ax in enumerate(axes[row]):
            ax.plot(array(grid_times), array(curves["WT"][key])[:, g], color=BLUE,
                    label="WT fitted system")
            ax.plot(array(grid_times), array(curves["KO"][key])[:, g], color=ORANGE,
                    linestyle="--", label=f"{tf_name} regulon KO")
            ax.set(title=names[g], xlabel="Same simulation time (a.u.)", ylabel=label)
            if key == "velocity":
                ax.axhline(0, color="#B0B0B0", linewidth=0.7)
            if row == 0 and g == 0:
                ax.legend()
    finish(fig, "perturbation.png", f"WT and {tf_name} regulon KO | learned system, zero initial states")
