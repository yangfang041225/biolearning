# Environment setup and validation record

This file records the environment the code was run in and the outcome of that run. It is
not part of the upstream teaching material.

## 1. Environment used

- Python 3.13.14 (Windows x64)
- torch 2.14.0+cpu
- torchdiffeq 0.2.5
- numpy 2.5.3
- matplotlib 3.11.2
- pillow 12.3.0

Command to reproduce:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pillow
```

`pillow` is required by `run_validation.py` (it decodes the exported PNGs) but is not
listed in `requirements.txt`.

## 2. How to run

```powershell
.\.venv\Scripts\python.exe data.py
.\.venv\Scripts\python.exe transcription.py
.\.venv\Scripts\python.exe -m unittest -v test_core
.\.venv\Scripts\python.exe main.py --epochs 20 --output results_smoke   # quick smoke
.\.venv\Scripts\python.exe run_validation.py                            # full validation
```

## 3. Validation result

`validation.json` reports `status: passed` on this environment, with checks
`core_numerical_checks`, `free_time_end_to_end`, `training_improves_and_solver_agrees`,
`network_time_smoke_30_epochs`, `plot_files_decode`.

All six unit tests in `test_core.py` pass, including the analytic-solution check and
the finite-difference checks of the time and weight gradients.

Free-time mode, 400 epochs (the run stored in `results/`):

| quantity | value |
|---|---|
| reconstruction MSE sum, epoch 1 | 0.081504 |
| reconstruction MSE sum, epoch 400 | 0.000201 |
| learned vs. generating time, Pearson r | 0.9914 |
| W RMSE over all entries | 0.4278 |
| max state error, 48 vs. 96 RK4 steps | 2.4e-07 |
| max state error, RK4 vs. dopri5 | 6.0e-07 |

Network-time mode, 30-epoch smoke run: MSE 0.1753 to 0.0067, time Pearson r 0.9916.

## 4. What the figures show, including a negative result

Visual inspection of `results/*.png` (recorded in `validation.json`):

- `loss.png` decreases monotonically on a log scale; the total objective and the
  reconstruction MSE are almost identical, so the penalty term is negligible here.
- `reconstruction.png` shows the ODE curves passing through the observed points in all
  six panels: the fitted dynamics explain the observations very well.
- `latent_time.png` shows the learned time is monotone in the generating time but not
  equal to it: the mapping is nonlinear and saturates near `t_max`. The time axis is
  therefore only identified up to a monotone reparameterisation.
- `grn.png` is the important negative result. Only one edge is recovered:

  | edge | generating W | learned W |
  |---|---|---|
  | A -> B | +1.20 | +0.46 |
  | A -> C | +0.60 | -0.04 |
  | B -> C | -0.80 | +0.03 |

  Reconstruction MSE reaches 2e-4 while two of the three true edges stay near zero.
  A low reconstruction error does **not** imply recovery of the generating GRN, which
  is exactly the identifiability problem discussed in section 7 of `README.md`. In the
  full RegVelo model this is addressed by variational inference plus the dynamics
  regularisation of Equations 6-8, which this teaching version replaces with plain MSE.
- `velocity.png` correlates with the generating velocity but deviates in a
  time-scale-dependent way, as expected given the time reparameterisation above.
- `perturbation.png` shows that deleting the A regulon lowers B (the one recovered
  edge) and leaves A and C unchanged; C is unchanged because the learned weights into C
  are near zero. The quality of a simulated knockout therefore depends on the learned
  GRN, not on how well the model reconstructs the data.

The `results_network_smoke/` folder produced by `run_validation.py` is excluded from
this repository by `.gitignore`; it is a 30-epoch smoke run, not a converged result.
