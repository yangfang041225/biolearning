"""有物理/数值意义的检查：解析解、有限差分梯度、硬约束和 KO。"""
import unittest
import torch
from data import make_format_example, make_synthetic_data
from dynamics import RNADynamics
from model import MiniRegVelo, solve_at_times
from perturb import knockout_dynamics, compare_perturbation


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_against_analytic_solution(self):
        # W=0,b=0,beta=gamma=1: u=a(1-e^-t), s=a(1-(1+t)e^-t)。
        system = RNADynamics(torch.zeros(1, 1)).double()
        times = torch.tensor([0., 0.3, 1., 3.], dtype=torch.float64)
        result = solve_at_times(system, times, steps=96)
        a = torch.log(torch.tensor(2., dtype=torch.float64))
        expected = torch.stack([a * (1 - torch.exp(-times)),
                                a * (1 - (1 + times) * torch.exp(-times))], dim=1)
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)

    def test_time_gradient_matches_finite_difference(self):
        system = RNADynamics(torch.zeros(1, 1)).double()
        times = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
        value = solve_at_times(system, times, steps=64).sum()
        analytic = torch.autograd.grad(value, times)[0]
        eps = 1e-5
        with torch.no_grad():
            finite = (solve_at_times(system, times + eps, 64).sum()
                      - solve_at_times(system, times - eps, 64).sum()) / (2 * eps)
        torch.testing.assert_close(analytic[0], finite, atol=1e-6, rtol=1e-5)

    def test_weight_gradient_and_mask(self):
        system = RNADynamics(torch.tensor([[0., 0.], [1., 0.]])).double()
        times = torch.tensor([0.7, 1.3], dtype=torch.float64)
        weight = system.transcription.W_raw
        with torch.no_grad():
            weight[1, 0] = 0.4
        solve_at_times(system, times, 64).sum().backward()
        gradient = weight.grad[1, 0].item()
        self.assertGreater(abs(gradient), 1e-6)
        self.assertTrue((weight.grad[system.transcription.G_prior == 0] == 0).all().item())
        eps = 1e-5
        with torch.no_grad():
            weight[1, 0] = 0.4 + eps
            plus = solve_at_times(system, times, 64).sum()
            weight[1, 0] = 0.4 - eps
            minus = solve_at_times(system, times, 64).sum()
        self.assertAlmostEqual(gradient, ((plus - minus) / (2 * eps)).item(), places=5)

    def test_ko_changes_trajectory_preserves_wt(self):
        system = RNADynamics(torch.tensor([[0., 0.], [1., 0.]]))
        with torch.no_grad():
            system.transcription.W_raw[1, 0] = 1.
        original = system.transcription.effective_weights().detach().clone()
        ko = knockout_dynamics(system, 0)
        self.assertTrue((ko.transcription.effective_weights()[:, 0] == 0).all().item())
        torch.testing.assert_close(system.transcription.effective_weights(), original)
        times = torch.tensor([0., 1., 2.])
        result = compare_perturbation(system, times, 0)
        torch.testing.assert_close(result["WT"]["S"][:, 0], result["KO"]["S"][:, 0])
        self.assertGreater(result["WT"]["S"][-1, 1].item(), result["KO"]["S"][-1, 1].item())
        # 在相同状态上 spliced velocity 不直接依赖 W。
        u, s = torch.ones(2), torch.ones(2)
        torch.testing.assert_close(system.spliced_velocity(u, s), ko.spliced_velocity(u, s))

    def test_both_time_modes_have_gradients(self):
        data = make_format_example()
        for mode in ("free", "network"):
            model = MiniRegVelo(data, time_mode=mode, steps=16)
            pred = model(data["U_obs"], data["S_obs"])
            loss = (pred["U_pred"] - data["U_obs"]).square().mean()
            loss += (pred["S_pred"] - data["S_obs"]).square().mean()
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            self.assertGreater(sum(p.grad.abs().sum().item()
                                   for p in model.latent_time.parameters()), 1e-8)

    def test_synthetic_reproducibility_and_row_alignment(self):
        data, truth = make_synthetic_data(8, noise_std=0.)
        data2, truth2 = make_synthetic_data(8, noise_std=0.)
        torch.testing.assert_close(data["U_obs"], truth["U_clean"])
        torch.testing.assert_close(data["S_obs"], truth["S_clean"])
        torch.testing.assert_close(data["U_obs"], data2["U_obs"])
        torch.testing.assert_close(truth["time"], truth2["time"])


if __name__ == "__main__":
    unittest.main()
