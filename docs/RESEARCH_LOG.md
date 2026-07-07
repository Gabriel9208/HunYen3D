# Research Log

A running, reverse-chronological log of the research process behind HunYen3D: papers read,
methods tried, the hypothesis behind each, and what the results actually showed.

The point is not a tidy changelog — it's to capture *why* something was tried and *what was
learned*, including dead ends. Results are interpreted as evidence about methods, not as
leaderboard numbers (see the README's scope note: resources are limited and SOTA is not the
goal).

## Papers

| Paper / Idea | Content |
|------|--------------|
| 3DShape2VecSet | VecSet representation | 
| Dora | Sharp Edge Sampling |
| Hunyuan3D-2 | A VAE-DiT 3D shape foundation model structure |
| XCube | Sparse Voxel |
| DeepSDF (CVPR 2019) | Clamped-L1 SDF loss, δ=0.1 truncation; **auto-decoder** (per-shape latent, no encoder/KL) |
| Occupancy Networks (CVPR 2019) | Geometry as occupancy (inside/outside) via **BCE**; uniform sampling works best |

## Experiments

| Date | Paper / Idea | Hypothesis | Change | Dataset | Result / Observation | Next step |
|------|--------------|------------|--------|---------|----------------------|-----------|
| 2026-06 | Hunyuan3D-2 ShapeVAE (baseline) | The hand-written VAE can fit a single mesh, confirming the encode→KL→decode→SDF path and loss are wired correctly | Overfit one mesh: `+experiment=overfit` (lr 1e-4, kl_weight 0, fixed_seed, bf16) | 1 mesh | Train loss → ~0.003 by ~epoch 69. Reproduction path validated. | Move to a small multi-mesh set (`+experiment=first_train`); turn KL back on (γ≈1e-4) and watch recon vs. KL. |
| 2026-07-02 | DeepSDF (TSDF clamp) | Clamping the far field to ±δ stops it dominating the MSE and focuses capacity near the surface | Clamp pred & gt to [−0.1, 0.1] before MSE (`clamp_val=0.1`) | small (~45k) | **Collapse from-scratch**: KL→1e-5, val loss flat 0.0083 ×27 ep, output RMS≈16, IoU 0%, marching cubes empty 4/4. See detailed §1 | Abandon clamp for from-scratch; keep far-field anchor |
| 2026-07-06 | Near-surface weighting (own construction) | Up-weight near surface *without* removing the far-field anchor that clamp destroyed | `w = 1 + λ·exp(−β·|gt|)`, loss Σ(w·se)/Σw; β∈{30,60,100}, λ=4; 3 epochs | small, 16 val | IoU 5.65%→**25–30%** (β=100 best); **near sign-acc stuck ~52%** for all β. 10-ep follow-up: loss 5× lower, IoU 46%, but near sign-acc only 53.5% → **structural, not under-training**. See §2 | Sign-decoupled loss (§3) |
| 2026-07-06 | Occupancy Nets / 3DShape2VecSet (sign decouple) | A BCE sign term supplies the non-vanishing surface gradient MSE lacks | `SignAwareSDFLoss = weighted-MSE + α·BCE(−k·pred, gt<0)`, same head; α,k Hydra-tunable | small (staged) | Self-check: **~380× stronger gradient** at a surface sign-error vs weighted-MSE; not yet trained. See §3 | Run `4_sign_aware`, sweep α∈{.03,.1,.3}, k∈{10,30,100} |
| 2026-07-07 | Sign-hinge (own) + isolation diagnostics | Is the stuck near sign-acc the loss, the architecture, sampling/KL, or capacity? | `SignHingeSDFLoss` α-sweep {0.1,0.5}; then single-mesh overfit (deterministic & real VAE path) | small + 1-mesh | hinge near sign flat ~52% across α → **not loss**. Single-mesh overfit: deterministic **92%**, real VAE path (σ≈1) **95%** → **not architecture, not sampling/KL**. Isolated to **capacity/generalization to 45k**. See §4 | Capacity experiment: num_latents 2048 vs 4096 (`6_cap2048` / `7_cap4096`) |

## Detailed entries — SDF loss design (2026-07)

The line of work on *why* the small model has low MSE but broken geometry. Format per step:
**Motivation / Decision / Result / Why / References** (which paper, in which scenario, used which
method to solve what — and, honestly, where we have no citation).

### §1. Small baseline → TSDF clamp (2026-07-02/03)

**Motivation.** The small unclamped baseline (ckpt `2026-07-02/23-03-26`) had low MSE but broken
geometry: near-band (|gt|<0.02) sign-accuracy ≈50% (coin flip), occupancy IoU 5.65%, fragmented
marching cubes. A swap test ruled out posterior collapse (the latent *was* being used). Diagnosis:
the MSE gradient is `2·(pred−gt)`, so near the surface where gt≈0 the residual — and thus the
gradient — is weakest exactly where geometry matters. Capacity flows to the easy far field and the
surface is neglected ("loss low, quality bad").

**Decision.** Adopt DeepSDF-style TSDF clamping: clamp both prediction and target to [−0.1, 0.1]
before the MSE, so the far field can no longer dominate the loss.

**Result.** Disaster from-scratch: KL→~1e-5 (posterior collapse), val loss dead-flat at 0.0083 for
27 epochs, output RMS blew up to ~16, IoU 0%, marching cubes empty on 4/4 shapes.

**Why.** Clamping removes the far-field *directional anchor*. With both pred and gt clamped to δ in
the far field, the loss there is flat (zero gradient) for any pred ≥ δ — so an unbounded constant
field minimises the loss and the latent becomes unnecessary → collapse.

**References.**
- **DeepSDF** (CVPR 2019) — introduces the clamped-L1 SDF loss with truncation δ=0.1
  (unit-sphere-normalised) to focus capacity near the surface. *Crucial scenario difference*:
  DeepSDF is an **auto-decoder** (a per-shape latent optimised directly, no encoder, no KL), so it
  never faces posterior collapse. The same truncation that is safe there is fatal in our
  encoder+KL VAE. **Same method, different regime, opposite outcome.**
- **TSDF** truncation originates in volumetric range-image fusion (Curless & Levoy, SIGGRAPH 1996;
  KinectFusion, 2011) — a fusion/denoising trick, never intended as a from-scratch training loss.

### §2. Clamp → near-surface weighting (2026-07-06)

**Motivation.** Keep the near-surface emphasis clamp aimed for, but *without* removing the
far-field anchor whose flat loss caused collapse.

**Decision.** Additive-floor weighting `w = 1 + λ·exp(−β·|gt|)`, loss = Σ(w·(pred−gt)²)/Σw. The far
field keeps w→1 (a normal MSE anchor); the near surface is up-weighted to 1+λ. Normalisation makes
β=0 collapse to plain MSE (clean toggle). λ=4, β swept {30,60,100}. Refactored the loss into
swappable Hydra classes (Weighted / Clamped).

**Result (3-epoch sweep, 16 val shapes).** IoU 5.65% → **25–30%** (β=100 best); the mid band
(0.02–0.1) MSE crossed the all-zeros baseline for the first time (a band actually "learned"). **But
near-band sign-accuracy stayed at ~52% (coin flip) for all three β**, and near-band MSE was still
worse than outputting zero.

**10-epoch follow-up (β=100, cosine re-matched to 10 ep).** Rules out under-training: val loss
driven **5× lower** (0.0038→0.0008) and IoU up to **46%**, yet **near sign-accuracy moved only
52.4%→53.5% — still coin-flip**, and near-band MSE stayed *worse* than outputting zero (0.00061 vs
0.00010). **Verdict: not under-training — structural.** Minimising this loss further does not
resolve the surface sign; the loss and the target metric (near sign-acc) are decoupled, exactly as
the vanishing-gradient argument predicts. This is the green light for §3.

**Why.** Weighting helped the **mid** band, where residuals are non-zero so a gradient exists →
better global inside/outside → higher IoU. It cannot help the **near shell**: the gradient is
`2·w·(pred−gt)`, and at the surface (pred−gt)→0, so `w × (≈0) = ≈0`. Weighting *multiplies* a
vanishing gradient; it cannot create one. Sweeping β 3.3× with zero movement in near sign-accuracy
is the empirical confirmation.

**References.**
- **Honest status: none of the SDF/occupancy papers surveyed use per-point distance weighting.**
  DeepSDF, **Occupancy Networks** (CVPR 2019), and **3DShape2VecSet** (SIGGRAPH 2023) all balance
  near/far via **sampling density**, not loss reweighting (OccNet reports uniform sampling best;
  DeepSDF samples aggressively near the surface).
- `exp(−β·|·|)` is a generic RBF/Gaussian-kernel falloff, not from a specific 3D paper. β was
  chosen by matching the weight's effective reach (~3/β) to DeepSDF's δ=0.1 → β≈30, up to 100 to
  match our data's near-surface scale (median |sdf|≈0.016, 57% of points within |sdf|<0.02).
- **Correction:** an earlier note mis-attributed exponential distance weighting to *Pixel2ISDF*;
  the original uses truncated L2 + image/eikonal supervision and has no such term. No citation
  supports this weighting — it is our own construction.

### §3. Weighting → sign-decoupled loss `SignAwareSDFLoss` (2026-07-06, staged)

**Motivation.** Since weighting cannot manufacture a surface gradient under MSE, attack the root
cause directly. Precise framing: MSE does not "ignore" sign, but its sign penalty ∝ magnitude, so
it vanishes at the surface (where magnitudes →0) — exactly where sign matters. Decouple the sign
into a classification whose gradient is *maximal* at the boundary.

**Decision.** `SignAwareSDFLoss = weighted-MSE + α·BCE(logit = −k·pred, target = [gt<0])`. Reuse
the same SDF head: reinterpret its output as an occupancy logit (inside = sdf<0). α (`sign_weight`)
balances sign vs distance; k (`sign_temp`) is the sigmoid temperature. Both Hydra-tunable; α=0
reduces to the weighted loss. Single head — the minimal decoupling, no second head yet.

**Result.** Not yet trained. Self-check: at a surface sign-error point (gt=+1e-3, pred=−1e-3),
SignAware's gradient is ~**380× larger** than weighted-MSE's — the non-vanishing surface gradient
is restored. Awaiting `4_sign_aware`; success criterion = near sign-accuracy leaving ~52%
(target >65%).

**Why.** A logistic-classification loss has its steepest gradient at the decision boundary (the
surface) — the complement of MSE's weakness. k sets the band where the sign gradient is
non-saturated (~3/k, the same length-scale role β played; too-large k saturates the sigmoid
off-surface and starves it). α sets its magnitude relative to the distance term. Taken to the limit
(α→∞, drop distance) this *is* occupancy + BCE — so the decoupling **re-derives 3DShape2VecSet's
original objective**, here added onto distance regression as the smallest single-head version.

**References.**
- **Occupancy Networks** (CVPR 2019) — models geometry as occupancy trained with **BCE**; the
  direct precedent for a "sign head" with a non-vanishing boundary gradient.
- **3DShape2VecSet** (SIGGRAPH 2023) — the representation this project reproduces; its original
  supervision is **occupancy + BCE**. The decoupling, at its limit, recovers exactly this —
  evidence the instinct converges on the field's established choice, not a novel gamble.
- **DeepSDF** (CVPR 2019) — used **L1** (not L2) for SDF regression; L1's gradient is sign(residual),
  constant magnitude, i.e. non-vanishing at the surface. Precedent that keeping a live near-surface
  gradient matters. (An L1 variant is the cheaper fallback if BCE proves unstable.)
- The weighted-MSE + BCE **hybrid itself is our own construction**; only its BCE component is the
  OccNet / 3DShape2VecSet objective.

### §4. Isolating the near-sign-acc bottleneck — a narrowing funnel (2026-07-07)

After §1–3, near-band sign-accuracy was stuck at ~52% (coin flip) no matter the loss. This day
was spent **eliminating hypotheses one at a time**, each test with a control. Two diagnostics were
themselves confounded and only caught by adding a control — logged honestly below.

**The funnel** (each row kills one hypothesis):

| # | Test | Question | Result | Eliminated |
|---|------|----------|--------|-----------|
| 1 | weighted β-sweep {30,60,100}, 3 ep | Is it β? | near sign ~52% for all | not β |
| 2 | weighted 10-epoch | Under-training? | loss 5× lower, IoU 46%, near sign 53.5% | not under-training |
| 3 | *diag (batch 2, no accum)* | Why is BCE unstable? | "BCE inflates pred" | ⚠️ confounded |
| 4 | **control: weighted in same diag** | Really BCE? | weighted bounces identically → diag invalid | retract "BCE unstable" |
| 5 | hinge 1-ep real runner vs weighted first 30 steps | Is hinge unstable? | step-for-step identical → early spike is a normal fresh-model MSE transient | retract the whole "instability" story |
| 6 | hinge α-sweep {0.1, 0.5} 3 ep | Does more sign weight help? | near sign 52.0 / 52.3, flat | not α, not the sign-loss form |
| — | synthesis | | clamp/weighted/BCE/hinge × β × α × epoch all ~52% | **not the loss** |
| 7 | *single-mesh overfit, KL=0 sampled* | Can the architecture do it? | σ exploded to 120 (no KL to regularise) | ⚠️ confounded |
| 8 | **single-mesh overfit, deterministic (z=μ)** | Pure architectural capacity? | near sign climbs to **92%** as near-RMS→0.0033 (< |gt| median 0.0071) | not architecture, not loss |
| 9 | **single-mesh, real VAE path (sampled z + kl=1e-4), 8k steps** | Sampling/KL vs generalisation? | σ≈1 (KL healthy), near sign → **95%** | not sampling/KL |

**Conclusion.** The architecture + weighted-MSE **can** resolve near-surface sign — 92% deterministic,
95% with the full stochastic VAE path — but only when memorising **one** mesh. On the 45k-mesh train
set the same model sits at 53%. The bottleneck is therefore **capacity / generalisation to 45k
shapes through a small latent set**, not the loss, not the architecture's expressive power, not the
VAE sampling. The entire §1–3 loss line was treating the wrong disease.

**Method notes (honest).**
- **RMS, not MSE, is the yardstick.** RMS = √(mean err²) is in SDF units, so it compares directly to
  |gt|. Near-surface signs are recoverable exactly when near-RMS < |gt| (0.0033 < 0.0071 → 92%).
- **The σ explosion (test 7) was self-inflicted:** with `kl_weight=0` there is no term pulling σ
  toward the prior, and on a single mesh the decoder memorises f(query)→sdf and ignores the latent,
  so σ drifts unbounded (→120). With KL on (test 9) σ stays ≈1. KL works; it was switched off.
- **Two confounded diagnostics (3, 7)** were only exposed by adding a control (4, 8/9). Lesson: every
  quick diagnostic needs a control run, or its small-batch / no-KL setup can fake the effect.

**Next.** Capacity experiment: hold the small model fixed, vary only `num_latents` (2048 vs 4096, with
the coupled `random+important` sampling) — `6_cap2048` vs `7_cap4096`. Batch size per config found via
`smoke_test.py` first. Expectation: if near sign-acc rises with 4096, capacity is confirmed as the
lever; then map how many shapes each capacity can hold before near sign degrades.

## Papers to read / backlog

- _(add candidate papers and the specific idea each might contribute)_

## Method ideas / open questions

- KL weight schedule: fixed small γ vs. warmup — effect on reconstruction sharpness.
- Augmentation: rotation/jitter at the light preprocessing stage (SDF is equivariant to rigid
  rotation) — does it help generalization on a small dataset?
- _(running list of inspirations to validate once the baseline is solid)_
