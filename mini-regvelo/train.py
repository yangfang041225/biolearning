"""训练闭环：观测 -> 时间 -> ODE -> 重建损失 -> 参数更新。

输入模型与 data 字典；输出标量训练历史。
教学目标 MSE_U+MSE_S+lambda*sum(W**2)，不是 Equation 5 的 ELBO。
不含论文 Equations 6--8 的完整动力学正则。
"""
import copy
import torch
from torch.nn import functional as F


def fit(model, data, epochs=400, learning_rate=0.02, weight_decay=1e-4):
    if epochs < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("epochs 和 learning_rate 应为正，weight_decay 非负。")
    # 这里用 Adam；论文使用 AdamW。本例惩罚项显式写在 loss 中。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_loss = float("inf")
    best_state = None
    model.train()
    for epoch in range(epochs):
        # 1. 清除上一次梯度。PyTorch 默认累加梯度，所以不能省略。
        optimizer.zero_grad()
        # 2. 正向计算；这里包含数值积分。
        prediction = model(data["U_obs"], data["S_obs"])
        mse_u = F.mse_loss(prediction["U_pred"], data["U_obs"])
        mse_s = F.mse_loss(prediction["S_pred"], data["S_obs"])
        W = model.dynamics.transcription.effective_weights()
        penalty = weight_decay * W.square().sum()
        loss = mse_u + mse_s + penalty
        if not torch.isfinite(loss).item():
            raise RuntimeError("损失非有限；请降低学习率或增大积分步数。")
        # 保存的是本次计算 loss 时的参数，不是下一次更新后的参数。
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "loss": loss.item(),
                        "mse_u": mse_u.item(), "mse_s": mse_s.item(),
                        "penalty": penalty.item()})
        # 3. 反向传播：通过求解器计算每个可学习参数的导数。
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
               for p in model.parameters()):
            raise RuntimeError("发现非有限梯度。")
        # 教学数值保护：限制一次更新的梯度范数，不改变 forward 方程。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # 4. 用梯度更新 W_raw、b、速率 raw 参数和时间参数/网络。
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch + 1:4d} loss={loss.item():.6f} "
                  f"reconstruction={(mse_u + mse_s).item():.6f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return history
