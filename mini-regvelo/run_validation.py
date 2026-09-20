"""依次运行数值检查和端到端实验，失败时返回非零退出码。"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    report = {"status": "running", "checks": [], "python": sys.version}

    def save():
        (ROOT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def command(name, arguments):
        print(f"\nValidation: {name}", flush=True)
        subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)
        report["checks"].append(name)
        save()

    save()
    try:
        command("core_numerical_checks", ["-m", "unittest", "-v", "test_core"])
        command("free_time_end_to_end", ["main.py"])
        metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
        if metrics["final_reconstruction_mse_sum"] >= 0.7 * metrics["initial_reconstruction_mse_sum"]:
            raise RuntimeError("默认训练未达到至少 30% 的重建 MSE 下降，请检查模型和优化。")
        if max(metrics["max_state_error_step_doubling"], metrics["max_state_error_vs_dopri5"]) > 0.005:
            raise RuntimeError("拟合后积分误差超过教学检查阈值 0.005。")
        report["checks"].append("training_improves_and_solver_agrees")
        # network 仅做端到端冒烟检查；30 轮不保证拟合收敛。
        command("network_time_smoke_30_epochs", ["main.py", "--time-mode", "network",
                "--epochs", "30", "--output", str(ROOT / "results_network_smoke")])
        from PIL import Image
        for folder in [ROOT / "results", ROOT / "results_network_smoke"]:
            images = list(folder.glob("*.png"))
            if len(images) != 6:
                raise RuntimeError(f"预期六张图：{folder}")
            for path in images:
                with Image.open(path) as image:
                    image.verify()
        report["checks"].append("plot_files_decode")
        report["visual_review"] = "Pending human/agent visual inspection; decoding alone is not visual QA."
        report["metrics"] = metrics
        versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (ROOT / "environment.txt").write_text(versions, encoding="utf-8")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise
    save()
    print("Numerical and end-to-end validation passed; see validation.json.", flush=True)


if __name__ == "__main__":
    main()
