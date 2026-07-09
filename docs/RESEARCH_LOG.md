# Research Log

A running, reverse-chronological log of the research process behind HunYen3D: papers read,
methods tried, the hypothesis behind each, and what the results actually showed.

The point is not a tidy changelog — it's to capture *why* something was tried and *what was
learned*, including dead ends. Results are interpreted as evidence about methods, not as
leaderboard numbers (see the README's scope note: resources are limited and SOTA is not the
goal).

> **Guiding lesson (retrospective).** Much of §1–4 chased near-surface **sign-accuracy** as if it
> were the objective. It isn't. Good sign-acc is neither sufficient nor necessary for good
> reconstruction — it's a **saturating proxy** that tops out below marching-cubes resolution (full
> argument in §5b). Its proper role is an **auxiliary diagnostic**: a quick read on whether surface
> detail is being learned at all. So the loss-engineering line that tried to *manufacture* sign
> (BCE §3, hinge §5d) was optimising a proxy — the target metric is mesh geometry (IoU / Chamfer /
> `viz_recon`), and losses/capacity should be judged by that.

## Papers

| Paper / Idea | Content |
|------|--------------|
| 3DShape2VecSet | VecSet representation | 
| Dora | Sharp Edge Sampling |
| Hunyuan3D-2 | A VAE-DiT 3D shape foundation model structure |
| XCube | Sparse Voxel |
| DeepSDF (CVPR 2019) | Clamped-L1 SDF loss, δ=0.1 truncation; **auto-decoder** (per-shape latent, no encoder/KL) |
| Occupancy Networks (CVPR 2019) | Geometry as occupancy (inside/outside) via **BCE**; uniform sampling works best. Eval metrics: IoU / Chamfer-L1 / normal-consistency / F-score |
| IGR / SIREN (2020) | **Eikonal** regularizer (‖∇f‖=1): a constraint a true SDF satisfies automatically, yet empirically crucial — the 3D precedent for a *redundant* auxiliary loss helping (see §5d) |

## Experiments

| Date | Paper / Idea | Hypothesis | Change | Dataset | Result / Observation | Next step |
|------|--------------|------------|--------|---------|----------------------|-----------|
| 2026-06 | Hunyuan3D-2 ShapeVAE (baseline) | The hand-written VAE can fit a single mesh, confirming the encode→KL→decode→SDF path and loss are wired correctly | Overfit one mesh: `+experiment=overfit` (lr 1e-4, kl_weight 0, fixed_seed, bf16) | 1 mesh | Train loss → ~0.003 by ~epoch 69. Reproduction path validated. | Move to a small multi-mesh set (`+experiment=first_train`); turn KL back on (γ≈1e-4) and watch recon vs. KL. |
| 2026-07-03 | None | None | first_train caue OOM. Lower the self-attention layer head num to half of the original and reduce FPS query points from 4096 to 2048 (uniform 1024 + sharp edge 1024) | 3DShape2VecSet Watertight mesh dataset | Training reconstruction loss → 0.00? and kl loss → 0.2. However, the result looks good but its awful near| |
| 2026-07-05 | DeepSDF (TSDF clamp) | Clamping the far field to ±δ stops it dominating the MSE and focuses capacity near the surface | Clamp pred & gt to [−0.1, 0.1] before MSE (`clamp_val=0.1`) | small (~45k) | **Collapse from-scratch**: KL→1e-5, val loss flat 0.0083 ×27 ep, output RMS≈16, IoU 0%, marching cubes empty 4/4. See detailed §1 | Abandon clamp for from-scratch; keep far-field anchor |
| 2026-07-06 | Near-surface weighting (own construction) | Up-weight near surface *without* removing the far-field anchor that clamp destroyed | `w = 1 + λ·exp(−β·|gt|)`, loss Σ(w·se)/Σw; β∈{30,60,100}, λ=4; 3 epochs | small, 16 val | IoU 5.65%→**25–30%** (β=100 best); **near sign-acc stuck ~52%** for all β. 10-ep follow-up: loss 5× lower, IoU 46%, but near sign-acc only 53.5% → **structural, not under-training**. See §2 | Sign-decoupled loss (§3) |
| 2026-07-06 | Occupancy Nets / 3DShape2VecSet (sign decouple) | A BCE sign term supplies the non-vanishing surface gradient MSE lacks | `SignAwareSDFLoss = weighted-MSE + α·BCE(−k·pred, gt<0)`, same head; α,k Hydra-tunable | small (staged) | Self-check: **~380× stronger gradient** at a surface sign-error vs weighted-MSE; not yet trained. See §3 | Run `4_sign_aware`, sweep α∈{.03,.1,.3}, k∈{10,30,100} |
| 2026-07-07 | Sign-hinge (own) + isolation diagnostics | Is the stuck near sign-acc the loss, the architecture, sampling/KL, or capacity? | `SignHingeSDFLoss` α-sweep {0.1,0.5}; then single-mesh overfit (deterministic & real VAE path) | small + 1-mesh | hinge near sign flat ~52% across α → **not loss**. Single-mesh overfit: deterministic **92%**, real VAE path (σ≈1) **95%** → **not architecture, not sampling/KL**. Isolated to **capacity/generalization to 45k**. See §4 | Capacity experiment: num_latents 2048 vs 4096 (`6_cap2048` / `7_cap4096`) |
| 2026-07-07/08 | Capacity — latent width | Does doubling `num_latents` help? | `6_cap2048` vs `7_cap4096`, 3 ep, eff-batch 32 | small 45k | 2048: IoU 27%. 4096 **run 1**: μ-path eval broken (σ→5, posterior pathological, MSE 0.11, IoU 14). **Rerun (identical config, 07-09)**: posterior healthy, μ-MSE 0.00227, **IoU 32% > 2048's 27%**. ⚠️ Same config → one pathological, one healthy: 4096 helps *when stable* but is fragile at kl=1e-4. See §5a | Free-bits to stabilise 4096; depth |
| 2026-07-08 | Depth (enc6/dec12) + metric reframe | Is it network depth? Is near-sign even the right target? | `8_cap_6_12_layers`; 3 ep (stalled) then 7 ep | small 45k | 3 ep stalled at val 0.013 (short cosine killed LR); 7 ep trains fine (val→0.0016). **IoU 27%→38%** (depth is a real lever) while near-sign flat ~54% → near-sign is a **saturating proxy**, IoU/mesh is the target. viz: both meshes coarse, depth smoother/less fragmented. See §5 | Deeper+longer; judge by IoU/Chamfer/viz |
| 2026-07-09 | Deep + plain MSE, scheduler on/off | Does constant LR (no scheduler) help the deep net with the simplest loss? | `9_deep_mse_nosched` (const LR) vs `10_deep_mse` (cosine, 7 ep) | small 45k | const-LR **stalls** (val ~0.03 flat/rising — can't settle without decay). `10_deep_mse` (cosine) running. See §5 | (pending) |

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

**Result.** Collapse from-scratch — posterior collapse, dead-flat val, exploded output, empty mesh
(numbers in the table row).

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

**Result (table has the IoU/near-sign numbers).** New signal beyond the table: the **mid** band
(0.02–0.1) crossed the all-zeros baseline for the first time (a band actually "learned"), but the
**near** band stayed *worse* than outputting zero even after the 10-epoch follow-up drove val loss
5× lower. **Verdict: not under-training — structural.** Minimising this loss further does not resolve
the surface sign; loss and target metric (near sign-acc) are decoupled, exactly as the
vanishing-gradient argument predicts. Green light for §3.

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

**Result.** Not yet trained; self-check confirms the restored surface gradient (~380×, see table).
Success criterion for `4_sign_aware` = near sign-accuracy leaving ~52% (target >65%).

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

**Next.** Capacity experiment (`6_cap2048` vs `7_cap4096`) — vary only `num_latents`; see §5.

### §5. Capacity, the metric reframe, and depth (2026-07-07 → 07-09)

§4 isolated the bottleneck to "capacity / generalisation to 45k". This section acts on that — and
in doing so corrects the *target metric* we had been optimising.

**5a. Latent width (2048 vs 4096) — and a μ-eval pitfall.**
Doubling `num_latents` (with coupled sampling; eff-batch held at 32) barely moved near-sign
(54.8% → 56.6%, sample-path). **Run 1** of `7_cap4096` looked terrible in eval_recon (MSE 0.11,
IoU 14%) while its *training* val loss was fine (0.0015). Cause: eval_recon uses the posterior mean
**μ** (deterministic, reproducible — the correct choice), but that run's posterior was
**pathological** (σ up to 5, vs 2048's 1.5), so μ fell off the learned manifold (μ-path MSE 0.16 vs
sample-path 0.0009). **μ-eval didn't fail — it correctly exposed a sick posterior.**

**Rerun overturns the "weak lever" verdict (2026-07-09).** Re-ran `7_cap4096` with the *identical*
config (kl=1e-4, eff-batch 32, 3 ep) → this time the posterior was **healthy**: μ-path MSE 0.00227
(not 0.11), val 0.0085→0.0026→0.0023, and **IoU 32.5%, clearly above 2048's 27.3%** (lower MSE too,
0.00227 vs 0.00323). So doubling latents **does help** — the earlier "barely moves / weak lever"
call was an artefact of run 1's broken posterior, not the real 4096 capacity. **The catch is
stability**: the *same* config produced a pathological posterior once and a healthy one once, so
4096 at kl=1e-4 is **fragile** (pathology is a stochastic risk, not an inevitability). Fix is
free-bits / stronger KL so μ reliably reconstructs — then 4096's capacity gain is dependable.

**5b. The metric reframe — near-sign-acc is a saturating proxy.**
Key realisation (user-driven): **high near-surface sign-acc is not a necessary condition for good
reconstruction.** Sign-acc is a thresholded view of near-RMS, and it saturates at the surface:
- For points with |gt| < achievable RMS, the sign is a coin flip *regardless of model quality*
  (you cannot get the sign of a point whose true value is below your precision floor). With median
  near |gt|≈0.007, even a good fit (RMS 0.007) tops out around ~75% near-sign, never 100%.
- Marching cubes at res 128 has cell size 2/128 ≈ 0.016, so any surface displacement below that is
  **not even extractable** — pushing near-sign from 75%→95% (RMS 0.007→0.003) optimises something
  below the extraction resolution.
So near-sign was a *useful diagnostic* (it drove the whole §1–4 narrowing) but the **wrong target**.
The right target is actual mesh geometry: **IoU / Chamfer / normal-consistency / eyeballing the
mesh** (`viz_recon`). Built `src/metrics/` for this — `OccupancyIoU`, `ChamferDistance` (scipy
cKDTree, points→scalar), and `diagnose/BandedSDFMetrics` (the near/mid/far sign-acc kept as an
internal diagnostic). Standard metrics for reporting, banded sign-acc for debugging.

**Consequence for loss design.** Sign-acc's proper role is an *auxiliary read* on whether surface
detail is being learned — not a quantity to build a loss around. The whole line that tried to
*manufacture* the sign (BCE §3, hinge §5d, and the sign-motivated weighting §2) was engineering a
proxy. This does **not** mean those losses are worthless — only that near-sign was the wrong lens to
judge them; their real question (do they improve **IoU/mesh**?) was never actually measured (§5d).
Going forward, judge any loss/capacity change by mesh geometry, and read sign-acc only as a sanity
check that the near band is being touched at all.

**5c. Depth (enc6/dec12) — the real lever, but the LR schedule matters.**
Deeper net (enc6/dec12, up from small's 4/8), num_latents fixed 2048. 3-epoch run **stalled** at
val 0.013 (flat from epoch 0) — but this was **not** a depth failure: the 3-epoch cosine decayed
the LR to ~0 before the deeper net got going. The 7-epoch run (cosine re-matched) **trains fine**
(val 0.0134→0.0016, comparable to the shallow model). By the *right* metric it clearly helps:
**IoU 27% → 38%**, while near-sign stays flat ~54% — exactly the reframe of 5b (IoU moves, the
saturating near-sign doesn't). `viz_recon` on a fine-grid shelf: both models lose the grid detail
(coarse blobby slab), but the deeper one is smoother/more complete, the shallow one noisier and
fragmented — visually consistent with IoU 38% vs 27%. Both are still poor, so there is real
headroom; the direction is more capacity + longer training (not more loss engineering). *(Const-LR
variant `9_deep_mse_nosched` stalls — without decay the model can't settle; the scheduler helps.)*

**5d. Redundant/auxiliary losses — open question with precedent.**
The sign hinge is *informationally* redundant (sign is derivable from SDF), but a redundant signal
can still reshape the gradient field / reprioritise scarce capacity toward the surface. Precedent:
deep supervision / auxiliary losses (Deeply-Supervised Nets), and — directly in 3D — the **Eikonal
loss** (IGR/SIREN), a constraint a true SDF satisfies automatically yet which is empirically crucial
for neural-SDF quality (and which can also *destabilise*, cf. NeurIPS'23). Honest gap: we dismissed
hinge by **near-sign** (the wrong, saturating metric); its effect on **IoU/mesh** was never
measured. A clean re-test would be hinge vs plain MSE on the same deep net, judged by IoU + viz.

**Ops notes (learned the hard way).**
- **Long training runs need `setsid nohup` detachment**, not the Bash tool's `run_in_background` —
  the latter killed 7-epoch runs at ~2 min / epoch-0 twice (clean external SIGKILL, no error).
- **Blackwell (RTX 5080) GPU hangs** (`cudaErrorLaunchTimeout`) crash long runs randomly even at
  `-pl 300`; save per-epoch checkpoints so a late crash still leaves a usable ckpt.

## Papers to read / backlog

- _(add candidate papers and the specific idea each might contribute)_

## Method ideas / open questions

- KL weight schedule: fixed small γ vs. warmup — effect on reconstruction sharpness.
- Free-bits for larger latents (§5a).
- Re-test redundant/auxiliary losses by IoU/mesh, not near-sign (§5d).
- Push the confirmed lever: deeper + longer training (§5c).
- Augmentation: rotation/jitter at the light preprocessing stage (SDF is equivariant to rigid
  rotation) — does it help generalization on a small dataset?
- _(running list of inspirations to validate once the baseline is solid)_
