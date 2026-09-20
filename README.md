# biolearning

Learning notes and study projects on computational biology.

## Contents

### [mini-regvelo](mini-regvelo/)

A teaching-oriented mini implementation of **RegVelo: Gene-regulatory-informed dynamics
of single cells** (Cell 189, 3773-3800, 2026), used to study how a dynamical system can
be recovered from static single-cell snapshots.

What is in it:

- The full mini-RegVelo pipeline: observations `(U_obs, S_obs)` and a prior graph
  `G_prior` in, latent time, a coupled RNA ODE, reconstruction loss and parameter
  updates out.
- Six unit tests covering the analytic solution, finite-difference checks of the time
  and weight gradients, the prior mask, and the knockout logic.
- A 400-epoch validation run with its metrics, training history, six diagnostic figures
  and the trained model checkpoint.
- [`mini-regvelo/ENVIRONMENT_NOTES.md`](mini-regvelo/ENVIRONMENT_NOTES.md): how the
  environment was set up, a Windows runtime-DLL problem that prevented `import torch`,
  and a summary of what the figures do and do not show.
- [`mini-regvelo/README.md`](mini-regvelo/README.md): module-by-module description of
  the code, the mathematics behind each file, and an explicit list of the places where
  this teaching version simplifies the published model.

Key result of the recorded run: the model reconstructs the observed data almost exactly
(reconstruction MSE 1.6e-4 per matrix) while recovering only one of the three true
regulatory edges. Low reconstruction error is not evidence that the underlying gene
regulatory network has been identified.
