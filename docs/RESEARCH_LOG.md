# Research Log

A running, reverse-chronological log of the research process behind HunYen3D: papers read,
methods tried, the hypothesis behind each, and what the results actually showed.

The point is not a tidy changelog — it's to capture *why* something was tried and *what was
learned*, including dead ends. Results are interpreted as evidence about methods, not as
leaderboard numbers (see the README's scope note: resources are limited and SOTA is not the
goal).

## Experiments

| Date | Paper / Idea | Hypothesis | Change | Dataset | Result / Observation | Next step |
|------|--------------|------------|--------|---------|----------------------|-----------|
| 2026-06 | Hunyuan3D-2 ShapeVAE (baseline) | The hand-written VAE can fit a single mesh, confirming the encode→KL→decode→SDF path and loss are wired correctly | Overfit one mesh: `+experiment=overfit` (lr 1e-4, kl_weight 0, fixed_seed, bf16) | 1 mesh | Train loss → ~0.003 by ~epoch 69. Reproduction path validated. | Move to a small multi-mesh set (`+experiment=first_train`); turn KL back on (γ≈1e-4) and watch recon vs. KL. |

## Papers to read / backlog

- _(add candidate papers and the specific idea each might contribute)_

## Method ideas / open questions

- KL weight schedule: fixed small γ vs. warmup — effect on reconstruction sharpness.
- Augmentation: rotation/jitter at the light preprocessing stage (SDF is equivariant to rigid
  rotation) — does it help generalization on a small dataset?
- _(running list of inspirations to validate once the baseline is solid)_
