# Research Log

Reproducing & improving the **Hunyuan3D-2 ShapeVAE** — a VecSet VAE: encode surface points → latent
set → decode SDF. Single RTX 5080, ~45k ShapeNet watertight meshes (3DShape2VecSet dataset).

## Primer — what this project is (read first)

- **Task & I/O.** Given a mesh's surface points, the encoder maps them to a latent set; a decoder then
  predicts, for any 3D query point, its **SDF** (signed distance to the surface: <0 inside, >0 outside,
  0 on it). The model **regresses a continuous SDF** — it does *not* output occupancy. Occupancy / IoU
  numbers are **derived** from the SDF by its sign (inside = SDF<0). *(The reference papers
  3DShape2VecSet / OccNet supervise occupancy with BCE; we regress SDF with MSE — "occupancy+BCE" below
  is background, not our objective.)*
- **Architecture.** VecSet VAE: encoder cross-attention → **a set of N latent tokens, each of width 64**
  (`num_latents` N ∈ {2048, 4096}; `latent_dim` = 64) → KL → decoder cross-attention → per-query SDF.
  N (token count) and 64 (per-token width) are *separate* axes; **total latent dim d = N×64 ≈ 1.3e5** at
  N=2048. Mesh is extracted from the predicted SDF field by **marching cubes (MC)**.
- **Objective.** Plain or near-surface-weighted **MSE on the SDF**, plus a **KL** regulariser weighted by
  `kl_weight`.
- **Experiment layout — the "grid" and "cells".** Capacity experiments form a **2×2 grid** over
  `num_latents {2048, 4096} × depth {enc4/dec8, enc6/dec12}`, labelled **cells A–D** (table in Stage-2
  below). **Cell A** = the smallest (2048, enc4/dec8); capacity is monotone, so A is the floor.
- **Scale anchors (is a number good?).** IoU / F-score / NC are 0–1, higher is better; Chamfer and MSE
  are errors, lower is better. Reference points: overfitting *one* mesh is the ceiling (F-score ~1.0,
  IoU ~90%, Chamfer ~0.009); the real 45k-generalisation task currently reaches **IoU ~45%**
  (current-best, **not** a solved target — real headroom remains).

## Glossary & notation

**Acronyms.** VAE (variational autoencoder) · SDF (signed distance field) · TSDF (truncated SDF) ·
MSE (mean-squared error) · BCE (binary cross-entropy) · KL (Kullback–Leibler divergence) · ELBO
(evidence lower bound) · MC (marching cubes — mesh extraction from the SDF grid) · IoU
(intersection-over-union) · NC (normal consistency) · FPS (farthest-point sampling) · DiT (diffusion
transformer) · VecSet (the latent-set representation from 3DShape2VecSet).

**Notation.** `encN/decM` = N encoder / M decoder layers · `klXeY` = `kl_weight` = X·10⁻ʸ (e.g.
`kl1e-4`) · `cell A–D` = the four grid configs · `rand+imp` = counts of random + importance
(near-surface) query points · `sample-avg (K=32)` = decode K latent samples z∼N(μ,σ), average the SDF
fields · `μ-path` = decode the posterior mean μ deterministically.

**Symbols.** μ = posterior mean · σ = posterior std · d = total latent dim (≈1.3e5) · gt = ground-truth
SDF · `|gt|<0.02` = query points within 0.02 of the surface (coords normalised to ~[−1,1]) · σ√d =
radius of the Gaussian sample shell in latent space.

## TL;DR

- **The loss is not the bottleneck.** Clamp, near-surface weighting, sign-BCE and sign-hinge all left
  near-surface sign-acc ~52% (coin flip); a one-hypothesis-at-a-time funnel ruled out the loss term.
- **Training BUDGET was badly underestimated — the old "~45%" was undertrained.** Cell A (2048,
  enc4/dec8) on 45k: 5 ep → V-IoU 43% / F 0.27, but the SAME cell at **20 ep → V-IoU 63% / F 0.69 /
  Chamfer 0.040 / NC 0.84** (2026-07-16, sample-avg K=32, res-256). Real current-best is **~63% V-IoU**,
  not 45%. S-IoU plateaus ~ep 15-17. Not collapse (swap ratio 8.66×, active units 94%). Budget is a
  first-class lever, alongside depth.
- **Depth helps; latent width does not — and "2048 beats 4096" was a measurement artifact.**
  enc4/dec8→enc6/dec12 lifted IoU 27→38%. But on re-eval with the unified metric suite, **2048 vs 4096
  (matched, enc4/dec8, 3 ep) is indistinguishable under seed noise**: 4096 swings V-IoU 28–38% /
  Chamfer 0.10–0.39 across two seeds, straddling the single 2048 run on every metric. Verdict: **4096 is
  unstable and no better — don't spend GPU on it**, but the old "~7 pt real gap" claim is **withdrawn**.
- **Reference (Hunyuan3D-2) reconstructs at 1024 tokens** — fewer than our 2048/4096 — with a DEEP net +
  massive data. So token count is not our bottleneck; the reference is "few tokens, deep". This points
  the plan at **1024 + depth**, not more tokens. (Confirm which config the paper's 1024 refers to; our
  log had recorded the ShapeVAE as 4096.)
- **near-sign-acc was the wrong target** — a thresholded proxy that saturates below the RMS floor and MC
  resolution. Judge geometry by **IoU / Chamfer / NC / F-score / eyeballing the mesh**.
- **decode(μ) is not guaranteed by the ELBO.** The "μ-eval broke" on high-KL / long-training runs was an
  **eval artifact, not a model break** — sample-averaged eval recovers the geometry. Eval a *sample*
  (field standard) or stay in a healthy-posterior regime. Not a KL problem; KL is a usable knob.
- **Capacity gate PASSED (2026-07-15):** the smallest cell memorises one mesh to F-score 1.00, NC 0.94,
  Chamfer 0.009, S-IoU 90%. Architecture is sufficient → the 2×2 grid is unblocked.

## Hard-won process lessons (the expensive ones)

- **Overfit ONE mesh FIRST, before any generalisation run.** The isolation funnel (07-07, timeline
  below) burned days on loss engineering while the real bottleneck was capacity; the single-mesh overfit
  is what finally isolated it. Capacity is **monotone** → overfitting the *smallest* cell clears the
  whole grid. This is now a mandatory Stage-1 gate, not an afterthought.
- **Every quick diagnostic needs a matched control.** Two diagnostics faked effects purely from small-
  batch / no-KL setups ("BCE is unstable", "σ explodes") — only a control run in the same setup exposed
  them as artifacts.
- **Don't judge geometry by a thresholded metric.** near-sign saturates; it drove the diagnosis but was
  never a legitimate target.
- **Don't confuse "decode(μ) broke" with "model broke".** Check a sample-path eval before blaming KL or
  training length.
- **Ops:** Blackwell (RTX 5080) throws random `cudaErrorLaunchTimeout` on long runs even at `-pl 300` →
  save checkpoints so a late crash still leaves a usable one.

## Stage-1 — Capacity gate: overfit one mesh (2026-07-15)

**Purpose.** Prove the architecture *can* memorise a single shape (rule out "insufficient capacity")
before spending GPU on the grid. Capacity is monotone → test only the **smallest** cell (A); if it
passes, every larger cell trivially has enough capacity.

**Controlled conditions — FIXED parameters (and why):**

| Param | Value | Why fixed |
|---|---|---|
| train mesh | 1 fixed (`sorted(paths)[0]` = an airplane), `max_meshes=1` | single-shape memorisation; deterministic pick |
| model | cell A: 2048 latents, enc4/dec8, width 1024, 16 heads, latent_dim 64 | smallest cell = the monotone capacity floor |
| `kl_weight` | 0 | pure capacity test, no regulariser |
| `deterministic` | true (z=μ) | no sampling → σ never sampled → no σ-explosion even at kl=0 |
| recon loss | plain MSE (`near_beta=0`) | no loss-engineering confound |
| lr | 1e-4 | 1e-3 diverges on overfit; 1e-4 descends stably |
| eval | μ-path (trained at z=μ), MC **res-256** | 256 exposes surface roughness that res-128 smooths over |
| PASS bar | train RMS < 0.02 | = the near-band width; a pre-set principled bar, not a post-hoc threshold |

**Varied parameters — what I OPENED, and what each revealed:**

| Param | Swept | Finding |
|---|---|---|
| `fixed_seed` | 0 (frozen) → null (resample every epoch) | **Frozen = only 2048 query points are ever supervised → the SDF field wiggles freely between them → rough / ugly surface.** Resampling injects fresh near-surface points each epoch → densely pins the field → smooth. |
| LR schedule | constant → cosine decay (1e-4→1e-6) | Constant-LR + resampling floors the loss at the SGD noise ball (the target now moves each epoch). Cosine decay settles it: loss 2.3e-4 → 1.4e-5, recovering the fidelity that resampling alone had cost. |
| epochs | 3000 → 6000 → 12000 | More epochs only help *once the seed is unfrozen and LR decays*; more frozen-seed epochs only over-fit the 2048 points harder. |

**Results (same ckpt lineage, eval res-256, μ-path):**

| version | NC (smooth) | Chamfer | F-score@.02 | V-IoU | S-IoU | RMS | train loss |
|---|---|---|---|---|---|---|---|
| frozen seed, const-LR, 3k ep | 0.712 | 0.0124 | 0.993 | 87.0% | 83.6% | 0.0036 | ~0 |
| unfrozen, const-LR, 6k ep | 0.860 | 0.0247 | 0.947 | 64.0% | 58.1% | 0.0119 | 2.3e-4 |
| **unfrozen, cosine-LR, 12k ep** | **0.943** | **0.0087** | **1.000** | **92.3%** | **90.2%** | 0.0035 | **1.4e-5** |

**Verdict: PASS.** The architecture memorises one mesh to F-score 1.00 / NC 0.94 / Chamfer 0.009 /
S-IoU 90%, and the render matches by eye. Two levers set render quality: **(1) resample every epoch**
(kills high-frequency roughness), **(2) cosine LR decay** (settles the resampled moving-target). Both
are already ON by default in real 45k training (`train.fixed_seed:null`, cosine scheduler) — so the
"ugly frozen overfit" is a protocol artifact, **not** what real training produces. **By monotonicity,
capacity is sufficient for the whole 2×2 grid → Stage-2 is unblocked.**

*S-IoU tops out ~90% (not higher) because it is thresholded sign in the near band — it saturates against
MC resolution, exactly why near-sign was retired. The continuous arbiters (F-score 1.00, Chamfer 0.009,
NC 0.94) are essentially maxed; they are the fidelity answer.*

## Stage-2 plan — 2×2 capacity grid (num_latents × depth)

Full 45k. One fixed protocol; vary only latent width × depth.

| cell | num_latents | enc/dec | rand+imp pts | batch × accum |
|---|---|---|---|---|
| A | 2048 | 4/8 | 1024+1024 | 4 × 8 |
| B | 2048 | 6/12 | 1024+1024 | 2 × 16 |
| C | 4096 | 4/8 | 2048+2048 | 2 × 16 |
| D | 4096 | 6/12 | 2048+2048 | 1 × 32 |

**Fixed across the grid:** eff-batch 32, `kl_weight=1e-4` (**per-dim mean-KL, kept deliberately** — 4096
thus gets ~2× total budget, the intended "more dims = more capacity at a fixed per-dim rate"; stated, not
a bug), plain MSE, cosine LR, fixed val set. **Eval:** sample-avg (K=32) primary **+** μ-path diagnostic
(no `max` — the winner flips across cells). F-score τ=0.02 (> MC cell 2/128≈0.0156). Start 5 ep; check
cell B 5-vs-7, then fix one epoch count for the whole grid.

## Stage-2 results (2026-07-16)

All eval: sample-avg (K=32), res-256, 16 val shapes, same val meshes.

**Cell A budget sweep (2048, enc4/dec8, plain MSE, kl1e-4, cosine 3e-4→1e-6).** Same cell, only epochs
change — this is the decisive "is 45% undertraining or a ceiling" experiment:

| epochs | V-IoU | S-IoU | Chamfer | F@.02 | NC | RMS |
|---|---|---|---|---|---|---|
| 5 (`grid_train`) | 43.5% | 43.6% | 0.109 | 0.270 | 0.598 | 0.040 |
| **20 (`grid_train_A20`)** | **63.1%** | **57.5%** | **0.040** | **0.685** | **0.837** | **0.0126** |

→ **Budget was the dominant lever.** F-score 0.27→0.69, V-IoU 43→63, Chamfer halved. The old "~45%" was
badly undertrained. val-loss flattened ~ep 10-14 and S-IoU plateaued ~ep 15-17 (LR ~2e-5 there → the
late settle is the anneal, not new learning) → **budget for the depth ladder ≈ 15-20 ep**. Still short of
the single-mesh ceiling (90% / F 1.0), so headroom vs capacity/depth/data remains. The 5-ep cell A is
**not** posterior collapse: swap test ratio 8.66×, sign-flip 36%, active units 93.7%, mu-spread 0.47 →
the decoder genuinely uses the latent.

**2048 vs 4096 re-eval (matched: enc4/dec8, kl1e-4, near_beta=30, 3 ep — the only CLEAN existing pair).**
Re-scored the old `6_cap2048` / `7_cap4096` / `7_cap4096_rerun` ckpts with the unified metric suite:

| metric | 2048 (1 seed) | 4096 seed-1 | 4096 seed-2 |
|---|---|---|---|
| V-IoU | 37.3% | 38.3% | 28.2% |
| S-IoU | 38.2% | 39.5% | 28.5% |
| Chamfer | 0.117 | 0.391 | 0.100 |
| F@.02 | 0.271 | 0.102 | 0.321 |
| NC | 0.561 | 0.514 | 0.521 |

→ **The two 4096 seeds swing more (V-IoU 28–38, Chamfer 0.10–0.39, F 0.10–0.32) than any 2048-vs-4096
gap; the single 2048 run sits between them on every metric.** So 2048 and 4096 are **indistinguishable
under seed noise here, and 4096 is markedly less stable** (one seed also lost 1/16 shapes to empty MC).
The prior "2048 beats 4096 (~7 pt real)" headline came from a **confounded** deep comparison
(`10_deep_mse` 2048 = plain MSE vs `11_deep_4096` 4096 = weighted loss, and `cap_deep4096_kl1e3` at
kl1e-3) — no clean deep 2048-vs-4096 pair exists. **Caveat:** this clean pair is shallow/3-ep/weighted
(undertrained), 1 seed vs 2; the only robust conclusion is "4096 unstable, not better".

**Note — `num_latents` is one knob, not two.** It IS the encoder FPS-query length =
`random_sample_count + important_sample_count`; you cannot hold query sampling fixed while changing the
latent count (they are the same number). Eval each cell in its native config.

## Metrics (decided)

Chamfer-L1 + V-IoU + S-IoU + Normal-Consistency + F-score@0.02, all in `src/metrics/` with per-metric
N/A fallback when a mesh is empty/uncomputable. Range / direction / scale anchor for each:

| metric | what it measures | range | good | anchor |
|---|---|---|---|---|
| **V-IoU** | occupancy IoU over **all** query points (global inside/outside, occ = sign(SDF)); a bare **"IoU"** in the timeline means V-IoU | 0–1 | higher | overfit ~0.92; 45k ~0.45 |
| **S-IoU** | V-IoU restricted to the **near band** `\|gt\|<0.02` (surface region) | 0–1 | higher | overfit ~0.90 |
| **Chamfer-L1** | mean nearest-neighbour surface distance (normalised coords ~[−1,1]) | ≥0 | lower | overfit ~0.009 |
| **NC** | normal consistency = mean \|cos∠\| of matched surface normals | 0–1 | higher | overfit ~0.94 |
| **F-score@0.02** | precision·recall of surface points within τ=0.02 | 0–1 | higher | overfit ~1.0 |

- **S-IoU is a comparability number ONLY** — being thresholded sign, it saturates like near-sign; do NOT
  gate on it.
- **Chamfer / NC / F-score** are the *continuous* fine-surface arbiters (no sign floor) — they carry the
  "is the surface actually learned" question.
- **τ must exceed the MC cell size** `2/res` (≈0.0156 at res 128), else F-score scores agreement finer
  than a vertex can be placed.
- **Retired metrics** used earlier in the timeline: **near-sign-acc** = fraction of near-band points with
  the correct *predicted sign* (saturates below the resolvable precision → dropped as a target); **RMS** =
  √(mean SDF-error²), same units as `gt` (used as a training-health read; the PASS bar `RMS<0.02` = the
  near-band width).

## Timeline (condensed)

| Date | What | Verdict |
|---|---|---|
| 2026-06 | Overfit 1 mesh, baseline VAE path | train loss→0.003; encode→KL→decode→SDF path wired correctly |
| 07-03 | first_train OOM → halved heads, FPS query 4096→2048 | trains, but near-surface awful |
| 07-05 | DeepSDF TSDF clamp ±0.1 | **collapse** — clamp removes the far-field anchor (the un-clamped far-field MSE that keeps the loss non-degenerate) → **posterior collapse** (decoder ignores the latent, KL→0). Keep the anchor. |
| 07-06 | Near-surface weighting `1+λ·exp(−β|gt|)` (λ = near-surface up-weight, β = falloff rate) | IoU 5.6→25-30%; near-sign stuck ~52% for all β → structural, not under-training |
| 07-06 | Sign-BCE `SignAwareSDFLoss` | self-check: ~380× stronger surface gradient (BCE steepest at the boundary, where MSE vanishes) |
| 07-07 | Sign-hinge α-sweep + **isolation funnel** (= eliminate one hypothesis at a time — loss form, β, epochs, architecture, sampling — each with a matched control) | all losses ~52% → **not the loss**. Single-mesh overfit (near-sign-acc): deterministic z=μ **92%** / sampled VAE path **95%** → **not architecture, not sampling**. Isolated to **capacity / 45k generalisation**. |
| 07-07/09 | Latent 2048 vs 4096 (3 ep) | 4096 helps when posterior healthy (32.5 vs 27.3%) but fragile at kl1e-4 |
| 07-08 | Depth enc6/dec12 + metric reframe | IoU 27→38%; **depth is a real lever**. near-sign is a **saturating proxy** → switch target to IoU/Chamfer/mesh |
| 07-09 | Deep + plain MSE, cosine vs const-LR | const-LR stalls (needs decay to settle); cosine 7ep IoU 45% |
| 07-10/11 | Depth×latent (7 ep): 2048 vs 4096 @ kl{1e-4,1e-3} | **2048 (45.3%) beats 4096 (31.6 / 36.9%)** → latent width is not a lever |
| 07-11/12 | 2048 long (20 ep) | μ-eval worsened (45→36) while sampled-val improved → *looked* like posterior drift… |
| 07-12/13 | 2048 kl5e-4 ×2 seeds | μ-eval "broke" (45→5-7 on both seeds)… |
| 07-13/14 | **Sample-avg eval probe (K=32)** | …both of the above were **decode(μ) artifacts, not model breaks** — sample-avg recovers 44/43. **4096<2048 survives** (~7 pt real). Root cause: decode(μ) has no ELBO term; in a ~1.3e5-dim latent the samples live on a shell of radius σ√d and μ is a near-zero-*mass* center the decoder never trains on (concentration of measure). Hunyuan/3DShape2VecSet decode a *sample*, not μ. |
| **07-15** | **Capacity gate rebuilt (Stage-1 above)** | **PASS** — clean, logged, new metric suite. Grid unblocked. |
| **07-16** | **Cell A budget sweep 5→20 ep (Stage-2 results above)** | **Budget was the dominant lever**: V-IoU 43→63%, F 0.27→0.69. Old "45%" was undertrained. Plateau ~ep 15-17 → budget ≈15-20 ep. |
| **07-16** | **2048-vs-4096 re-eval (clean pair)** | **"2048 beats 4096" withdrawn** — indistinguishable under seed noise; 4096 swings V-IoU 28-38% and is unstable. Headline's deep pair was loss-confounded. |
| **07-16** | **Paper check: Hunyuan3D-2 recon uses 1024 tokens** | Reference is "few tokens, deep" → token count is not our bottleneck. Plan pivots to **1024 + depth**. |

## Papers

| Paper | Contribution used |
|---|---|
| 3DShape2VecSet (SIGGRAPH 23) | VecSet representation; occupancy+BCE supervision; KL=1e-3, stated as being for the generative stage |
| Hunyuan3D-2 / 2.1 | VAE-DiT structure; ShapeVAE latent×64, enc8/dec16, width 1024; `encode` defaults to `sample_posterior=True` (decodes a sample, not μ). **Token count unresolved: log had 4096, but the paper's reconstruction experiments use 1024 — confirm which is the recon VAE config (07-16 timeline).** |
| Dora | Sharp-edge sampling |
| DeepSDF (CVPR 19) | Clamped-L1 SDF loss δ=0.1; **auto-decoder** (per-shape latent, no encoder/KL → no collapse — why clamp is safe there but fatal in our encoder+KL VAE) |
| Occupancy Networks (CVPR 19) | Occupancy via BCE; eval metrics IoU / Chamfer-L1 / NC / F-score |
| IGR / SIREN | Eikonal regulariser — precedent for a redundant auxiliary loss helping (and sometimes destabilising) |

## Backlog / open questions

- **RUNNING (2026-07-16): the single VAE baseline — 1024 tokens + enc8/dec16** (`vae_baseline_1024_l8_16`,
  cosine 3e-4→1e-6, kl1e-4, plain MSE, 20 ep, 327M params, batch 2×16). This **replaces the whole 2×2
  grid**: matching the reference depth (8/16) directly means no shallow-vs-deep sweep is needed, and 1024
  tokens matches the Hunyuan3D-2 reconstruction config. Drop 4096 (unstable, no better) and the depth
  ladder. Watch val/S-IoU for the plateau. Open: confirm which config the paper's 1024 is; does 1024 hurt
  on our 45k (small data may prefer fewer tokens).
- Re-test auxiliary/redundant losses (sign-hinge, eikonal) by **IoU/mesh**, not near-sign — their real
  effect was never measured (only the wrong metric was).
- KL knob: free-bits vs warmup vs fixed `kl_weight` — effect on μ-usability and recon sharpness (KL is reopened as
  a usable knob, not a break-cause).
- Augmentation (rotation/jitter; SDF is rigid-equivariant) — does it help small-data generalisation?
