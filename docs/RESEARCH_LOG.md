# Research Log

Trained on a single RTX 5080 with ~45k ShapeNet watertight meshes (the 3DShape2VecSet dataset),
this work reproduces and tries to improve the **Hunyuan3D-2 ShapeVAE** modules.

> **Note (2026-07-25).** Every earlier 45k-generalisation experiment used the wrong train/val split
> (cross-category, with the train and val categories fully disjoint — zero-shot rather than
> in-distribution), so those numbers are untrustworthy and have been removed along with the conclusions
> and tables drawn from them. The only results kept are the split-independent Stage-1 single-mesh
> overfit and the method/design descriptions. Everything below is now retrained and re-evaluated on the
> paper's official in-distribution three-way split (train/val/test), and refilled run by run.

## Primer — the basic setup of this project

This work builds on the open-source Hunyuan3D 2.0 project and its technical report, so the base model
architecture is inherited from it.

### Reconstruction side: the VAE encoder, decoder, and related modules

- **Task and I/O.** Given a mesh's surface points, the encoder maps them into a latent set; the decoder
  then predicts, for any 3D query point, its **SDF** (signed distance: <0 inside, >0 outside, 0 on the
  surface). Occupancy / IoU numbers are read off from the sign of the SDF (inside = SDF<0).
- **Architecture.** A VecSet VAE: encoder → **a set of N latent tokens, each of width 64**
  (`num_latents` N ∈ {1024, 3072}; `latent_dim` = 64) → KL → decoder → a per-query SDF. The mesh is
  extracted from the predicted SDF field by **marching cubes (MC)**.
- **Objective.** An **SDF MSE** term plus a **KL loss**.
- **Metrics.** V-IoU, S-IoU, F-score, NC, Chamfer distance.

## Stage-1 — Capacity test: overfit one mesh

*(This section overfits a single mesh with no train/val split at all, so it is unaffected by the split
bug and its numbers remain trustworthy — which is why it is kept.)*

**Purpose.** First prove the architecture *can* memorise a single shape, so that "insufficient capacity"
can be ruled out.

**Controlled conditions:**

| Param | Value | Why fixed |
|---|---|---|
| train mesh | 1 fixed (`sorted(paths)[0]` = an airplane), `max_meshes=1` | learn on the same single datum |
| model | cell A: 2048 latents, enc4/dec8, width 1024, 16 heads, latent_dim 64 | smallest cell = the monotone capacity floor |
| `kl_weight` | 0 | a pure capacity test, so ignore the regularisation term for now |
| `deterministic` | true (z=μ) | only μ is taken, so σ can be anything → no σ-explosion even at kl=0 |
| lr | 1e-4 | 1e-3 diverges on an overfit; 1e-4 descends stably |
| eval | μ-path (trained at z=μ), MC **res-256** | 256 exposes the surface roughness that res-128 smooths over |
| PASS bar | train RMS < 0.02 | = the near-band width |

**Varied parameters:**

| Param | Swept | Finding |
|---|---|---|
| `fixed_seed` | 0 (frozen) → null (resample every epoch) | **With a frozen seed the model only ever sees the same set of query points supervised, so even if it memorises all of them, nothing constrains the field between the points.** |
| LR schedule | constant → cosine decay (1e-4→1e-6) | Constant-LR plus resampling floors the loss at the SGD noise ball (the target moves every epoch). Cosine decay settles it: loss 2.3e-4 → 1.4e-5, recovering the fidelity that resampling alone had cost. |
| epochs | 3000 → 6000 → 12000 | More epochs help only *after the seed is unfrozen and the LR decays*; more frozen-seed epochs merely over-fit those 2048 points harder. |

**Results (same ckpt lineage, eval res-256, μ-path):**

| version | NC (smooth) | Chamfer | F-score@.02 | V-IoU | S-IoU | RMS | train loss |
|---|---|---|---|---|---|---|---|
| frozen seed, const-LR, 3k ep | 0.712 | 0.0124 | 0.993 | 87.0% | 83.6% | 0.0036 | ~0 |
| unfrozen, const-LR, 6k ep | 0.860 | 0.0247 | 0.947 | 64.0% | 58.1% | 0.0119 | 2.3e-4 |
| **unfrozen, cosine-LR, 12k ep** | **0.943** | **0.0087** | **1.000** | **92.3%** | **90.2%** | 0.0035 | **1.4e-5** |

**Verdict: PASS.** The architecture memorises one mesh to F-score 1.00 / NC 0.94 / Chamfer 0.009 /
S-IoU 90%, and the render matches by eye.

## Stage-2 — Baseline VAE, 20 epochs

`vae_baseline_1024_l8_16` (1024 tokens, enc8/dec16, plain MSE, kl1e-3, cosine 3.5e-5→1e-6, 20 ep;
the paper's official in-distribution split, trained from scratch). This is the non-anchor reference
baseline.

**A μ-broken posterior.** This baseline has a small μ-spread (0.14–0.20) and a σ_rms glued to the prior
(≈0.97), which places decode(μ) off the sample shell → the μ-path evaluation simply collapses. Because
the downstream diffusion fits a *sample* rather than μ (and the field standard is to decode a sample
too), the trustworthy read is **sample-avg**.

| eval | V-IoU | S-IoU | RMS | σ_rms | μ-spread |
|---|---|---|---|---|---|
| μ-path (256) | 9.32% | 7.52% | 0.216 | 0.969 | 0.202 |
| **sample-avg K=32 (64 shapes)** | **51.03%** | **44.36%** | 0.0203 | 0.972 | 0.144 |

*Caveat: sample-avg has so far only been run on 64 shapes (256×32 times out), and there are no mesh
metrics yet (Chamfer/F/NC). A full 256-shape SDF + 16-shape mesh eval is still to be run so this lines
up with, and is comparable to, the anchor version.*

**Decoder-depth variant — `vae_baseline_1024_l8_22` (2026-08-01).** Same recipe but **decoder deepened to 22
layers** (enc8/dec22 vs dec16 above), 20 epochs, from scratch. Same protocol (sample-avg K=16, mesh 16 res-128):

| eval | V-IoU (all) | V-IoU (uniform) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | μ-spread |
|---|---|---|---|---|---|---|---|
| l8_22 mesh 16 res-128 | 50.21% | 65.30% | 42.84% | 0.0402 | 0.705 | 0.814 | 0.134 |

→ dec22 at 20 epochs (uniform 65.3% / F 0.705) lands roughly where dec16 vanilla **converge** does (69.3% / 0.779) —
deepening the decoder gives no clear jump. *Caveat: l8_22 ran only 20 epochs while the vanilla it's compared to is
a converge run (~ep57), so budgets aren't matched; and the 16 shapes are all airplanes. Judging the depth benefit
still needs a budget-matched run.*

## Anchor VAE — per-token surface anchors as decoder conditioning

### Background

In LATTICE, Tencent's own improvement to the shape-generation branch of Hunyuan3D 2.0, they propose a
two-stage generation scheme. Because the shapes generated by Hunyuan3D 2.0 were not detailed enough,
they voxelise the output, extract the active voxels lying on the object's surface, and feed those as
input to a second Hunyuan3D-2.5 generation model. This guides the otherwise unstructured VecSet
generation, mimicking the success of grid-based 2D image generation.

### My approach

Rather than running the full generation pipeline twice, I wanted to run it once. I kept Hunyuan3D 2.0's
original architecture (VAE + DiT) and, on top of it, designed an **Anchor** module that tries to extract
absolute coordinates from the latent and inject them into the decoder as reference points, so that both
the generative model and the decoder can refer directly to the object's absolute coordinates while
generating. This decouples *where* from *what* — the same goal the paper pursues, reached by a different
route.

### Questions

**Q1. The latent is produced by diffusion, so it seemingly cannot make use of anchor information —
after all, the anchor takes the latent as its input?**

A1. Studies have found that 2D image and video diffusion models sketch the rough layout early and only
carve details in the later steps, and testing shows the same holds in 3D. So I take the intermediate
z from every time step in the second half of diffusion generation, compute the ẑ it predicts at that
moment, and use that as the anchor model's input.

**Q2. Can the anchor extract coordinates effectively?**

A2. Yes — provided it is trained jointly with the VAE, not bolted on after the VAE has been trained
alone. During training the anchor's gradient must also flow back into the VAE encoder: experiments show
that detaching the two makes training very hard (a moving target). Whether an anchor can instead be
trained on an already-trained VAE encoder is not yet established, and is left for future work.

### Design

**Anchor VAE (`AnchorVAE`):**
1. An **`Anchor` module** that runs 6 self-attn layers over the sampled z and, after a linear layer
   `proj_anchor`, produces per-token raw coordinates `(B, num_latents, 3)`.
2. A `FourierEmbedder` (the same PE used for the SDF query points) encodes the anchor coordinates.
3. An **`AnchorDecoder`** that, just before the SDF-query cross-attention, adds one more cross-attention
   (`anchor_cross_attention`, with the anchors as KV and the latent as Query), so the latent can lean on
   the anchor information to reconstruct a finer 3D model.

**Supervision:**
1. The anchors are pulled toward the **raw xyz of the FPS query points**, using **chamfer-L1 (the main
   supervision — a differentiable bidirectional `torch.cdist` NN, weight 1.0) plus index-aligned MSE (an
   auxiliary term, weight 0.1)**.
2. The config knobs `anchor_chamfer_weight` / `anchor_mse_weight` default to 0, so the base VAE and every
   prior run are unaffected.
3. The raw `anchor_cd` / `anchor_mse` are logged to wandb (train + val).

### Results

`anchor_baseline_1024_l8_16` (AnchorVAE, same recipe as the baseline above: kl1e-3, lr 3.5e-5 cosine,
20 ep, new split). Unlike the plain baseline, the anchor supervision spreads μ apart and keeps a
**healthy posterior** (μ-spread 0.42, σ_rms 0.73) — though sample-avg still reads clearly higher than
μ-path.

| eval | V-IoU | S-IoU | RMS | σ_rms | μ-spread |
|---|---|---|---|---|---|
| μ-path (256) | 48.46% | 39.81% | 0.0189 | 0.733 | 0.419 |
| **sample-avg K=32 (256 shapes)** | **66.42%** | **59.87%** | 0.0135 | 0.733 | 0.419 |

*Caveat: the mesh metrics (Chamfer/F/NC) were cut short this round and are still to be run. The baseline
is currently 64-shape while the anchor is 256-shape, so the shape counts do not match — both need one
full eval under the same protocol (256 SDF + 16 mesh, sample-avg) before they can be compared properly.
The old-split "anchor beats baseline" relationship still has to be re-verified on the new split.*

### Converge run — DONE + eval (2026-07-28)

`anchor_converge_1024_l8_16` (kl1e-3 anchor recipe, constant LR 3.5e-5 to a plateau, then resumed from
best.pt at **lr÷10 = 3.5e-6** to settle; new in-distribution split, from scratch). Evaluated with
**sample-avg** (K=16) — μ-path understates this posterior. This round also **adds a volume-IoU**: an
occupancy IoU over uniform-in-volume points only, matching the paper protocol (OccNet / 3DShape2VecSet /
Hunyuan all sample IoU points this way), as opposed to our default V-IoU whose bank is ~80% near-surface.

| eval | V-IoU (all) | V-IoU (uniform, paper) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | RMS | μ-spread |
|---|---|---|---|---|---|---|---|---|
| SDF 256 shapes | 76.23% | **87.34%** | 71.04% | — | — | — | 0.0107 | 0.327 |
| mesh 16 shapes res-128 | 81.23% | **88.79%** | 77.27% | **0.0117** | **0.979** | **0.940** | 0.0103 | 0.265 |

*(Both rows sample-avg K=16; the 256-shape SDF metrics are the more representative, the 16-shape row is
the mesh-viz convention and is the one carrying Chamfer/F/NC — all 16/16 extracted a surface.)*

→ **Surface fidelity is essentially SOTA-level**: F-score **0.979**, Chamfer **0.0117**, NC **0.94** — on
par with or better than 3DShape2VecSet's reported F-score (~0.97). **Volume-IoU ~88%** sits in the paper
range; the only low number is **S-IoU (near-surface sign) ~71-77%**, which is the structural weakness of
SDF-MSE (vanishing gradient at the surface), separate from surface quality. **Key lesson: the earlier
"stuck at 0.72" was measuring the wrong metric** — IoU over the ~80%-near-surface bank rather than the
paper's uniform volume points; switching to volume-IoU immediately lands back in the paper range. Renders
(GT↔recon, 16/16) are in `results/converge_meshsample_viz/`.

**Vanilla control (anchor-vs-vanilla A/B, 2026-07-30).** Ran the vanilla arm `vae_converge_lr3.5e-6_1024_l8_16`
under the same recipe (anchor removed, everything else identical: kl1e-3, constant LR 3.5e-5 to plateau then
lr÷10 to settle; same in-distribution split), evaluated the same way (sample-avg K=16, mesh 16 shapes res-128):

| eval | V-IoU (all) | V-IoU (uniform) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | μ-spread |
|---|---|---|---|---|---|---|---|
| **anchor** mesh 16 res-128 | **81.23%** | **88.79%** | **77.27%** | **0.0117** | **0.979** | **0.940** | 0.265 |
| vanilla mesh 16 res-128 | 58.16% | 69.25% | 52.27% | 0.0315 | 0.779 | 0.882 | 0.148 |

→ **Same codebase, same protocol: anchor beats vanilla across the board** (volume-IoU +19.5pt, S-IoU +25pt,
Chamfer nearly halved, F-score +0.20). This is the core A/B of the project — the contribution is established
here, without needing to match paper absolutes. *Caveat: the vanilla run only reached epoch ~57 with lr just
dropped 10×, and the 16 shapes are all one category (02691156 airplanes); a fairer comparison still needs both
arms evaluated cross-category on more shapes.*

#### Ablation — architecture vs supervision (archonly, 2026-08-01)

Does anchor's gain come from the **extra architecture** (the decoder's anchor cross-attention params) or from the
**anchor supervision** (chamfer+mse pulling tokens onto the surface)? Ran `anchor_archonly_1024_l8_16`: **anchor
architecture untouched, but `anchor_chamfer_weight`/`anchor_mse_weight` set to 0** (supervision off), kl1e-3,
lr 3.5e-5, 20 epochs, from scratch. Same protocol (sample-avg K=16, mesh 16 res-128). **For a budget-matched
comparison the anchor row uses the 20-epoch `anchor_baseline_1024_l8_16` (not the converge version)** — same
architecture, same 20 epochs as archonly, differing only in the supervision switch:

| run | arch | anchor sup. | budget | V-IoU (uniform) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | μ-spread |
|---|---|---|---|---|---|---|---|---|---|
| **anchor_baseline** | anchor | ✅ | 20 ep | **83.66%** | **64.19%** | **0.0169** | **0.937** | **0.908** | 0.322 |
| archonly | anchor | ❌ (w=0) | 20 ep | 47.63% | 31.52% | 0.0718 | 0.515 | 0.734 | **0.814** |
| vanilla (ref) | plain | — | converge ~ep57 | 69.25% | 52.27% | 0.0315 | 0.779 | 0.882 | 0.148 |

→ **Clean comparison (top two rows, same arch, same 20 ep): supervision on vs off = uniform +36pt, S-IoU +32.7pt,
F-score +0.42. Supervision is decisive.** The architecture itself is not the source of the gain — it's a liability:
archonly (arch, no supervision) loses even to the converged plain vanilla (F-score 0.515 vs 0.779). And **archonly's
μ-spread 0.814 sits near the prior (1.0)** — the posterior barely departs from the prior, meaning the extra decoder
cross-attention **hardly uses the latent when there's no anchor signal to ground it** (posterior near-collapse),
versus anchor_baseline's μ-spread 0.322 (the latent is actually used). **Conclusion: the gain comes from anchor
supervision, not architectural capacity.** *Caveat: the vanilla row is converge (~ep57), not 20 ep, so it serves only
as a reference that archonly underperforms even a plain baseline — it is not the budget-matched arm; aligning it needs
a 20-epoch `vae_baseline_1024_l8_16` mesh-16 eval. The 16 shapes are all airplanes.*

**Generation probe (latent health).** `scripts/sample_shapes.py` samples the latent and decodes directly:
posterior (z~N(μ,σ)) and aggpost (aggregate posterior) both decode full shapes (~12k+ verts), but the
plain prior (z~N(0,I)) degenerates into fragments (~1.2k verts) — the decoder is healthy, but the latent
is not aligned to N(0,1). **That is exactly the gap the DiT is meant to fill.**

## Metrics

Chamfer-L1 + V-IoU + S-IoU + Normal-Consistency + F-score@0.02, all in `src/metrics/`, each with an N/A
fallback when a mesh is empty or uncomputable. Range / direction / scale anchor for each:

| metric | what it measures | range | good | anchor |
|---|---|---|---|---|
| **V-IoU** | occupancy IoU over **all** query points (global inside/outside, occ = sign(SDF)); a bare **"IoU"** means V-IoU | 0–1 | higher | overfit ~0.92 |
| **S-IoU** | V-IoU restricted to the **near band** `\|gt\|<0.02` (surface region) | 0–1 | higher | overfit ~0.90 |
| **Chamfer-L1** | mean nearest-neighbour surface distance (normalised coords ~[−1,1]) | ≥0 | lower | overfit ~0.009 |
| **NC** | normal consistency = mean \|cos∠\| of matched surface normals | 0–1 | higher | overfit ~0.94 |
| **F-score@0.02** | precision·recall of surface points within τ=0.02 | 0–1 | higher | overfit ~1.0 |

- **V-IoU has two point sets:** the default `V-IoU (all)` is over the whole query bank (~80% near-surface,
  harder); `V-IoU (uniform)` uses only the uniform-in-volume points, **matching the paper's volume-IoU
  protocol** (OccNet / 3DShape2VecSet / Hunyuan) — this is the one to compare against papers (evaluate.py
  prints both). The ~80%-near-surface version systematically understates, so don't compare it to papers.
- **S-IoU is a comparability number only** — being a thresholded sign, it saturates just like near-sign,
  so **do not** gate on it.
- **Chamfer / NC / F-score** are the *continuous* fine-surface arbiters (no sign floor) — they carry the
  question of whether the surface was actually learned.
- **τ must exceed the MC cell size**; otherwise F-score would grade agreement finer than a single
  vertex can be placed. Measured ceilings and the choice of τ=0.02 are in
  [F-score τ and the marching-cubes ceiling](#f-score-τ-and-the-marching-cubes-ceiling) below.
- **μ-path vs sample-avg.** decode(μ) is not guaranteed by the ELBO; in a high-dim latent the samples
  live on a shell of radius σ√d, while μ can be a near-zero-mass centre the decoder never trains on.
  When the posterior is μ-broken, μ-path badly understates or even collapses → **always read sample-avg
  (K=32)** and treat μ-path as a diagnostic only.
- **Retired metrics:** **near-sign-acc** = the fraction of near-band points with the correct predicted
  sign (it saturates below the resolvable precision, so it was dropped as a target); **RMS** =
  √(mean SDF-error²), in the same units as `gt` (used as a training-health read; the PASS bar
  `RMS<0.02` = the near-band width).

### F-score τ and the marching-cubes ceiling

*Moved here from `scripts/paper_eval.py` on 2026-09-28 so the script keeps only the decision, not
the derivation.*

**The cell size is `2/(res-1)`, not `2/res`.** `Postprocess._extract` maps a voxel index back to
world coordinates as `v/(r-1)*2 - 1` (`src/model/shape/VAE/postprocess.py`), so at resolution 128
neighbouring vertices are 2/127 = **0.01575** apart, not 2/128 = 0.0156. Small difference, but it is
exactly where the ceiling breaks, so it matters for picking τ.

**Measured ceiling.** Marching-cubes the *true* SDF on the same grid and score the result against
the GT mesh — this is the best F-score any model could possibly achieve at that resolution. Measured
on 8 test shapes, 100k surface samples each:

| τ | 0.005 | 0.01 | 0.01575 | 0.02 |
|---|---|---|---|---|
| mean F | 0.59 | 0.967 | 0.9993 | 0.9997 |
| worst shape F | 0.45 | 0.916 | 0.9980 | 0.9987 |

**The break is at the cell size.** Below it the score collapses; at or above it the ceiling is
essentially 1.0. This is why **τ = 0.02 is the default**: the literature-standard 0.01 throws away
up to 8 F-score points to pure discretisation, and does so *unevenly across shapes* (worst shape
0.916 vs mean 0.967), which would bias any comparison toward whichever model happens to produce
smoother meshes rather than more accurate ones.

**A wrong inference that was later refuted — worth keeping.** The same measurement put the
Chamfer-L1 floor at 0.0095 (range 0.0052–0.0115) against reported values of 0.0127 / 0.0136 / 0.0143.
I concluded from this that most of each Chamfer number was discretisation rather than model error,
and that the models were being separated near an artificial floor.

**That conclusion was wrong.** Re-running at resolution 256 on 600 shapes (`docs/eval_result_256.md`)
moved Chamfer by at most 1.8%, and *upward* rather than downward (lambda 0.01270 → 0.01333). If the
number were dominated by a discretisation floor, quartering the cell size would have cut it
substantially. The ordering and the gaps between models were unchanged.

The lesson: a *ceiling* measured on the true field bounds what F-score can reach, but it does not
license the parallel claim about Chamfer. F-score is a thresholded count and so reacts sharply to
the cell size; Chamfer is a mean distance whose error is dominated by where the surface actually is,
not by how finely it is tessellated. Verify a floor by changing the resolution, not by reasoning
from the ceiling.

## Papers

### 3D Shape VAE
| Paper | Contribution used |
|---|---|
| 3DShape2VecSet (SIGGRAPH 23) | VecSet representation; occupancy+BCE supervision; KL=1e-3, explicitly stated as being for the generative stage |
| Hunyuan3D-2 / 2.1 | The VAE-DiT structure; ShapeVAE latent×64, enc8/dec16, width 1024; `encode` defaults to `sample_posterior=True` (decodes a sample, not μ). **Token count unresolved: the log recorded 4096, but the paper's reconstruction experiments use 1024 — confirm which is the recon VAE config.** |
| Dora | Sharp-edge sampling |
| DeepSDF (CVPR 19) | Clamped-L1 SDF loss δ=0.1; **auto-decoder** (per-shape latent, no encoder/KL → no collapse — which is why the clamp is safe there but fatal in our encoder+KL VAE) |

### DiT / Diffusion
Premise: diffusion models carve the shape early and the details late.

| Paper | arXiv | Contribution |
|---|---|---|
| DP-DMD (Diversity-Preserved DMD) | 2602.03139 | Directly observes the denoising trajectory on SD3.5-M (rectified flow), Appendix B + Figure A |
| TIDE | 2503.07050 | Uses a sparse autoencoder to analyse the DiT and pinpoint the timestep where the silhouette forms |
| SuperEdit | 2505.02370 | A four-stage decomposition: early global layout / mid local object attributes / late detail / style spanning throughout |
| UniTransfer | 2509.21086 | Chain-of-Prompt, injecting the prompt in coarse/medium/fine stages |
| SCoPE (Progressive Prompt Detailing) | CVPR-25 workshop | Progressive prompt refinement, designed on the coarse-to-fine premise |
| SPARE | 2602.07058 | Cites this property in its Timestep Targeting section |
| DiT-BlockSkip | 2603.20755 | Adjusts patch size by timestep, on the basis that high timesteps learn the global picture and low timesteps the detail |
| Mixture-of-Diffusers | OpenReview lcmd2Qdrsv | In the time-series domain, a division of labour between early/late-stage diffusers |

*Most entries cite this property to design a method rather than set out to verify it. The ones doing
systematic measurement are TIDE and the DP-DMD appendix. All are image/video, none 3D.*

## Backlog / open questions

**Re-run list (all on the paper's official in-distribution split; old-split numbers removed as
untrustworthy).**
- ✅ `vae_baseline_1024_l8_16` (plain, 20 ep) — trained + partial eval (see Stage-2); full sample-avg
  (256 SDF + 16 mesh) still to run.
- ✅ `anchor_baseline_1024_l8_16` (anchor, 20 ep) — trained + SDF eval (above); mesh still to run.
- ✅ `anchor_converge_1024_l8_16` (kl1e-3 trained to convergence, settled at lr÷10) — eval'd (see Converge run: volume-IoU 88.8%, F-score 0.979); `max_epochs=-1` so it can be resumed anytime.
- ⏳ Ablations to re-run: capacity control (`l8_22`, pure depth), detach (cut the anchor→encoder
  gradient), KL sweep, arch-only (anchor architecture but weight=0).

**Open questions.**
- Confirm which recon VAE config the paper's 1024 tokens corresponds to.
- Evaluate baseline / anchor with **sample-avg** throughout (μ-path is untrustworthy for a μ-broken
  posterior); fill in matched shape counts and mesh metrics before comparing.
- KL knob: free-bits vs warmup vs a fixed `kl_weight` — effect on μ-usability and reconstruction
  sharpness.
- Augmentation (rotation/jitter; SDF is rigid-equivariant) — does it help small-data generalisation?

## Final test-set evaluation (2026-08-02)

All seven experiments (each **best.pt**) evaluated under one protocol on the **paper in-distribution test split**.
**SDF metrics (V-IoU all/uniform, S-IoU, μ-spread) over all 2,592 test shapes**; **Chamfer / F-score@.02 / NC over
128 shapes stratified across categories** (marching cubes is too slow for the full set — ~12 days; 128 across 55
categories is representative). All **sample-avg K=16, res-128**. Script: `scripts/final_eval.py` (two `evaluate.py`
passes per model → `results/final_eval/*.json` + `summary.md`). Sorted by V-IoU (uniform):

| experiment | arch | anchor sup. | budget | V-IoU (all) | V-IoU (uniform) | S-IoU | Chamfer↓ | F@.02↑ | NC↑ | μ-spread |
|---|---|---|---|---|---|---|---|---|---|---|
| **anchor_converge_lr3.5e-6** | anchor | ✅ | converge | 77.54% | **89.48%** | **71.87%** | **0.0193** | **0.938** | **0.929** | 0.385 |
| **anchor_baseline** | anchor | ✅ | 20 ep | 67.16% | **83.01%** | 59.99% | 0.0241 | 0.872 | 0.889 | 0.465 |
| vae_converge_lr3.5e-6 | plain | — | converge | 62.28% | 80.92% | 55.04% | 0.0383 | 0.755 | 0.879 | 0.197 |
| vae_baseline (dec16) | plain | — | 20 ep | 55.53% | 75.34% | 48.21% | 0.0471 | 0.656 | 0.818 | 0.223 |
| vae_baseline_l8_22 (dec22) | plain | — | 20 ep | 54.66% | 75.11% | 47.02% | 0.0462 | 0.665 | 0.818 | 0.215 |
| anchor_detach | anchor | ⚠ detach | 20 ep | 42.84% | 58.88% | 37.10% | 0.0822 | 0.469 | 0.723 | 0.154 |
| anchor_archonly | anchor | ❌ (w=0) | 20 ep | 42.20% | 58.13% | 36.52% | 0.0834 | 0.427 | 0.721 | **1.056** |

**This table collapses the whole project into one sentence: quality is driven by anchor supervision that reaches the
encoder — not the architecture, not decoder depth, not merely training longer.**

1. **Anchor supervision beats vanilla at every budget.** 20 ep: uniform 83.01 vs 75.34 (+7.7pt), F 0.872 vs 0.656;
   converge: 89.48 vs 80.92 (+8.6pt), F 0.938 vs 0.755. Consistent and large.
2. **The decisive evidence — detach vs anchor_baseline (same architecture, same anchor loss; the only difference is
   whether the gradient flows back to the encoder): uniform 58.88 vs 83.01 (−24pt), F 0.469 vs 0.872.** Detaching the
   anchor gradient makes the entire anchor branch a liability — worse even than plain vanilla (75%). **This directly
   validates the core design claim: the anchor supervision must flow back into the VAE encoder** (otherwise it's a
   moving target, cf. Q2/A2).
3. **The architecture alone, unsupervised, is a liability.** archonly (w=0, no supervision) and detach (loss on but
   gradient cut) both drop to ~58% uniform / F ~0.43–0.47, far below plain vanilla — two independent controls, same
   conclusion. μ-spread agrees: detach **0.154** (near posterior collapse, encoder μ barely varies with shape),
   archonly **1.056** (μ scatters but decodes to nothing); healthy anchor sits at 0.39–0.47.
4. **Deepening the decoder does nothing.** l8_22 (dec22) ≈ l8_16 (dec16) (uniform 75.11 vs 75.34) — not a
   capacity/depth story.
5. **Training longer helps but doesn't change the ordering.** converge beats 20 ep on both arms (anchor 83→89,
   vanilla 75→81), yet anchor > vanilla holds at both budgets.

**Surface fidelity (anchor_converge): F-score 0.938, Chamfer 0.0193, NC 0.929** — on-par with 3DShape2VecSet even
over the full cross-category test set. This is the project's first same-protocol, full-test final comparison, and it
supersedes the scattered 16-shape / val numbers above. Renders are in `results/final_<experiment>/`.

## Latent-regulariser and loss-shape experiments (2026-08 → 2026-09)

*Migrated here from the experiment-config headers on 2026-09-26. Those headers were the only record
of these measurements; the configs now carry a three-line summary and point back to this section.*

### The measured problem these all attack

On the converged lambda-disjoint checkpoint (epoch 67, 24 test shapes × 50k queries) the SDF error
decomposes as **deterministic 0.0035 + posterior noise 0.0017** — 81% of the MSE is a fit the decoder
cannot improve. Train MAE 0.00375 vs test 0.00388 is a 3.5% gap, so this is pure underfitting, not a
generalisation failure, and no amount of regularisation helps. The near-surface band (|gt| < 0.002)
carries an error of 0.0030, **larger than the band is wide**, which is why the sign is near-random
there and V-IoU stalls around 83.

Baseline latent pathology, measured on `vae_disjoint_converge_kl1e-4_1024_l8_16`: sigma 0.87 against
mu.std 0.45, i.e. **81% of z's variance is noise**; the aggregate posterior is far from N(0,I) with
covariance condition number 590 and excess kurtosis +8.2 along random directions.

### Cross-check: is the noise-dominated posterior ours, or the architecture's?

Everything above rests on one measurement: most of the latent's variance is posterior noise rather
than shape information. That is only interesting if it is a property of the architecture and the
unweighted SDF loss, not of how this particular model was trained. Hunyuan3D-2.1 ships a released
ShapeVAE checkpoint, so the same quantities can be measured on it directly.

Run: `uv run --with einops python -m scripts.hunyuan_posterior_check --shapes 5`. Takes under a
minute once the weights are cached locally (7.5 GB under `~/.cache/hy3dgen/`; the ShapeVAE part is
a 625 MiB fp16 checkpoint pulled from HuggingFace on first use).

| | mu.std | sigma | SNR | signal fraction |
|---|---|---|---|---|
| ours (`vae_converge_kl1e-4_lr3.5e-6`, epoch 64) | 0.3172 | 0.9176 | 0.346 | **10.3%** |
| hunyuan3d-2.1, released checkpoint | 0.4576 | 0.8321 | 0.550 | **21.4%** |

Signal fraction is `Var(mu) / (Var(mu) + E[sigma^2])` — the share of the latent's total variance that
actually moves with the shape. Both models are noise-dominated: roughly 90% of our latent's variance
and 79% of theirs is posterior noise re-drawn on every sample.

**So the pathology is not a local training bug.** A released, production-scale model trained by the
authors of the architecture shows the same qualitative behaviour, which is what justifies treating it
as a property of the design rather than something to fix by tuning this repo's hyperparameters.

**But it is not identical, and the difference is not small.** Their signal fraction is about twice
ours (21.4% vs 10.3%). Whatever they do differently — vastly more data, a different KL weight, a
longer schedule — it buys a meaningfully cleaner posterior. Claiming the two are "the same" would
overstate the result; the honest claim is that both sit in the same regime.

**Two caveats that limit how far this can be pushed.**

1. *The field-disagreement numbers are not comparable across the two rows.* The script decodes each
   model twice, once from mu and once from a posterior sample, and reports how much the two output
   fields differ (ours 6.2%, theirs 9.5%). But our half evaluates on the cached SDF query points,
   which are 80% near-surface, while the Hunyuan half evaluates on points drawn uniformly in the
   cube. Near-surface points are where a field disagrees most, so the two numbers are measured on
   different question sets and should not be read as "theirs is worse".
2. *Their encoder was run with sharp-edge sampling switched off* (`pc_sharpedge_size: 0`), because
   that is what their shipped config and their own demo use. Given that this project's entire
   subject is the sharp branch, it is worth being explicit that the released default does not use
   one — their latent has no sharp/uniform split to collapse in the first place.

Also note the sample is 5 shapes, and `torch_cluster.fps` is replaced by a pure-Python fallback
(`scripts/hunyuan_posterior_check.py` explains why). Both are fine for a statistic aggregated over
the whole latent, but this is a sanity check, not a benchmark.

### lambda-VAE (arXiv:2607.05531, "Variance Equalization for Posterior Collapse")

Sampling becomes `z = mu + sigma^lam * eps` (Eq.9) with per-channel `lam_i = max(1, log(1-1/delta) /
(2 log sigma_i))` (Eq.16), while the KL term keeps charging the **original** sigma² (Eq.10). That
asymmetry is the whole method: the decoder sees a far cleaner latent than the KL penalty implies,
closing the information gap without weakening the regulariser. Substituting Eq.16 back gives
`sigma^lam = sqrt(1 - 1/delta)` for every channel above the floor — one shared effective noise level.

delta is the only hyperparameter and sets that level directly:

| delta | 1.001 | 1.01 | 1.1 | 2.0 |
|---|---|---|---|---|
| noise `sqrt(1-1/delta)` | 0.0316 | 0.0995 | 0.302 | 0.707 |

**delta = 1.1 was chosen** (both lambda runs use it) because it keeps the latent genuinely stochastic
— noise 0.302 against mu.std ~0.54, i.e. ~76% signal — rather than the near-deterministic 0.032 the
paper's RGB setting gives. It also costs the decoder far less: on a converged model, moving the
injected noise to 0.307 loses **3.3 V-IoU**, where moving it to 0.032 loses **9.0**.

**Ramp sizing.** `lam_ramp_steps: 225000` ≈ 10 epochs at ~22.7k `encode()` calls/epoch. The paper's
stated reason for its 150-epoch ramp — waiting for sigma to settle — does **not** apply here: sigma is
flat from epoch 1 (0.87 at ep1; 0.83–0.94 across ep9–60 of five runs), and at init sigma > 1 makes
Eq.16 return lam = 1 anyway, so lambda-VAE self-disables until sigma falls below 1. The ramp is sized
for the **decoder**: dropping the injected noise 0.89 → 0.032 in one step costs 9 V-IoU (73.3 → 64.2)
because z.std halves and `proj_latent` has never seen that input scale. Any lambda run must last
**≥ 20 epochs** — lam* is only fully on after epoch 10 and needs as long again to converge.

**Result on the non-MRL partner:** active units 55.9% → **91.9%**, channel std(mu) spread 10.7× →
**3.2×**. Final λ eval: 65,536/65,536 dimensions pass the Burda AU 0.01 threshold, with 24% *less*
total Var(mu) than the baseline.

Runs: `vae_lambda_disjoint_converge_kl1e-4_1024_l8_16` (lr 3.5e-5) and
`vae_lambda_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16` (lr 3.5e-6), both A/B'd against
`vae_disjoint_converge_kl1e-4_1024_l8_16` with only `model.lam_delta` differing (0.0 → 1.1).
*Caveat: both headers' prose said delta = 1.001 and lam* ≈ 24.8 while the YAML body set
`lam_delta: 1.1`; the body is what ran. The 1.001 figures are from an earlier plan, not from these runs.*

### lambda-VAE on the masked-attention encoder

`masked_vae_lambda_converge_kl1e-4_1024_l8_16`, against `masked_vae_converge_kl1e-4_1024_l8_16`.

`MaskedCrossAttentionEncoder`: the 1024 query tokens are uniform-first, sharp-second and the 20,480 KV
points are ordered the same way, so the masked blocks split both at the halfway point — uniform tokens
attend to everything, sharp tokens attend to sharp only. Two unmasked "mix" layers (mid-stack and
last) are the sole path by which uniform context reaches the sharp tokens. Block count matches the
plain encoder exactly (1 cross + 3 + mix + 3 + mix = 9 vs 1 cross + 8 self = 9), so **nothing here is
added capacity**.

Why lambda on this encoder: its problem is *not* dead channels (dead(<0.01) = 0/64 at epoch 18) but
uneven energy — per-channel mu.std max/median 2.18×, and per-channel KL spread 23× because KL goes as
mu². Variance equalization targets exactly that axis.

**Attribution warning:** two changes are stacked (masked attention + lambda-VAE). The clean read on
lambda-VAE alone is `vae_lambda_disjoint_converge_kl1e-4_1024_l8_16`.

### SIGReg (LeJEPA) as a latent regulariser — two attempts

**Attempt 1, SIGReg replacing KL on the sampled z** (`vae_sigreg_disjoint_converge_1024_l8_16`,
`sig_weight: 1e-2`). Motivation: the KL term charges `E_x[KL(q(z|x)‖p(z))] = I(x;z) + KL(q(z)‖p(z))`,
i.e. it pays for mutual information (which costs reconstruction) as well as for aggregate-posterior
shape (which is what the downstream DiT actually needs); SIGReg pays only the second term.
`sig_weight` was calibrated against recon **at convergence**, not at init — recon runs 0.0028 at epoch
0 down to 0.0002 by epoch ~40, so a fixed weight grows ~15× in relative terms over the run.

**This failed, and the failure is informative.** At epoch 4: SIGReg(z) = 0.00128 against a
finite-sample floor of 0.00050 — effectively converged — while the informative part went the *wrong*
way: SIGReg(mu) 0.366 vs the KL baseline's 0.241, mu.std 0.17 vs 0.45, signal share of z 3.0% vs
19.7%. **81% of z's variance is posterior noise, and Gaussian noise already satisfies the test on its
own**, so SIGReg-on-z is satisfied by the noise and lets mu do whatever it likes. Collapse risk was
anticipated: with KL gone, mu=0 / sigma=1 is a valid minimiser (SIGReg 1.04 at collapse vs 390 on a
real latent) and only reconstruction opposes it — weakest exactly once it has converged.

**Attempt 2, SIGReg on mu, added on top of an unchanged KL term**
(`vae_sigreg_mu_disjoint_converge_1024_l8_16`, `sig_weight: 1e-4`, `kl_weight: 1e-4`).

Why the KL term **stays**: SIGReg cannot hold the latent's scale. The Epps-Pulley statistic saturates
once the distribution is wider than the Gaussian window — measured on exact Gaussians, std 5 → 0.810,
std 10 → 0.824, std 200 → 0.822, with d/d(scale) of +0.334 at std 2, +0.025 at std 5 and −0.0002 at
std 20. Reconstruction pushes mu to **grow** (bigger mu = better SNR against a fixed sigma), which is
exactly the direction SIGReg guards badly, so dropping KL diverges: z_std reached **7.3 at sig 1e-4
and 181 at sig 1e-2**. LeJEPA never hits this because a JEPA has no reconstruction term — its failure
mode is collapse (shrinking), the side where SIGReg's gradient is strong. So KL anchors the scale and
SIGReg shapes skew/kurtosis/isotropy, which KL alone does badly.

`sig_weight = 1e-4` verified against the baseline over 240 steps from the same seed: viou 24.55 vs
23.44, kl 2.42 vs 2.46, z_std 2.52 vs 2.54 — indistinguishable trajectory, no divergence, while the
statistic itself fell 1.96 → 1.33.

### Recon-loss exponent: sqrt instead of MSE

`vae_disjoint_sqrt_kl1e-4_lr3.5e-6_1024_l8_16`, a single-axis A/B that **inherits** its partner
`vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16` (deliberately the opposite of the lambda runs: here
the whole point is that exactly one thing differs, so if the partner's LR or kl_weight ever moves, this
run must move with it). The entire diff is `task.recon_loss.exp: 0.5` and `eps: 1e-4`.

MSE's optimum is the conditional **mean**, and hedging toward the mean is precisely how small features
get averaged into a smooth surface — visible in the rendered comparison, where both models keep the
airframe but lose the small underbody protrusions. `exp = 0.5` moves the optimum toward the mode.

What it actually changes, measured on this loss at eps=1e-4 (per-point gradient):

| \|e\| | exp=2 | exp=0.5 | |
|---|---|---|---|
| 1e-4 | 3.06e-9 | 2.47e-7 | **81× more** pull on the nearly-correct points |
| 4e-3 | 2.67e-7 | 1.49e-7 | the crossover — and today's mean error |
| 1e-1 | 3.51e-6 | 1.51e-8 | **233× less** pull on the worst points |

So this is a reallocation of effort from the worst-fitted points to the well-fitted majority, with the
break-even sitting right at the current error level.

**The risk.** Everything past |e| > 0.02 gets largely abandoned — about 1% of query points. Part of
that 1% is broken GT (one test shape's labels claim 31.9% of the cube is inside a mesh whose true
volume is 1.0%, and that single shape carries 86% of the test MSE) and dropping those is a gain; the
rest is genuinely hard geometry, and dropping that is a loss.

**Scale.** `WeightedSDFLoss` raises the aggregate to `^(2/exp)`, so exp=0.5 lands on the same magnitude
as exp=2 rather than the 3700× below it that raw `mean(|e|^0.5)` would give. The match is exact only
for a homogeneous residual set: measured 9.4e-6 vs 1.6e-5 at the converged error scale (1.7×) and
5.4e-2 vs 1.7e-1 at init (3×). `kl_weight` therefore keeps its meaning and was deliberately **not**
retuned. Note that because the two runs optimise a different functional, `val/recon` is **not**
comparable across them — reading it as "the sqrt run has lower loss" is meaningless. The deciding
metric is `val/viou` against the partner's **81.35** (its best.pt's stored `best_metric`, epoch 65).

**eps = 1e-4 is required, not cosmetic.** Below exp=1 the gradient of |e|^0.5 diverges as e→0 and NaNs.
Smoothing only acts where |e| < eps, and a batch of 50k residuals ~N(0, 0.004) puts its smallest |e|
near 1e-7 every step, so anything much below 1e-5 is inert. Measured max/typical per-point gradient
ratio at exp=0.5: **436 for eps=1e-9 against 11.5 for eps=1e-4**, while plain MSE's own ratio is 9.7 —
1e-4 is the value that leaves this no more lopsided than the MSE training already trusted, and it
inflates the reported loss by 0.07%.

**Warm start, and why `init_from` not `resume`.** The partner's checkpoints load unchanged — verified
with `load_state_dict(strict=True)` on both best.pt (epoch 65) and last.pt (epoch 68), and the model,
preprocess and data configs compare equal, because the only key differing anywhere in the two composed
configs is `task.recon_loss.exp`. Use `trainer.init_from`, **not** `trainer.resume`: resume also
restores the Adam moments, which were estimated under e² — the very gradient profile this run changes
— giving every parameter a wrong effective step until the beta2=0.999 EMA turns over (~1000 steps).
Resume would also inherit `best_metric=81.35`, so no checkpoint gets written until the new loss beats
the old run, and a temporary dip during re-adaptation is expected.

**Result:** the sqrt run reached val/viou **79.71** against the partner's 75.28. *(Run died at epoch 22
on 2026-09-20 with no traceback — external kill / SIGHUP, not a numerical failure.)*

**If it loses,** try exp=1.0 (MAE, conditional median) before abandoning the idea; 0.5 is the
aggressive end of the range.

### MRL + lambda + per-m exponents, stacked

`mrl_lambda_exp_disjoint_kl1e-4_1024_l8_16` — three levers on the disjoint VAE, all aimed at the same
measured problem above.

1. **MRL** (`task.m_values: [8, 16, 32, 64]`, `m_weights: [1, 2, 3, 4]`) forces every prefix of the code
   to reconstruct on its own, so channels must be ordered by importance instead of each carrying an
   arbitrary share. The weights are a *sampling distribution* over m, so one decode per step rather
   than four: same expected gradient, more variance.
2. **lambda-VAE** (`lam_delta: 1.1`) attacks the 19% posterior-noise term directly.
3. **Per-m recon exponents** (`m_exponents: [2.0, 1.5, 1.0, 0.5]`) make the loss less mean-seeking as m
   grows. Short prefixes keep exp=2 because gross shape *is* what they are for; only the full-width
   code, the one actually asked to resolve detail, gets exp=0.5.

**The risk, stated up front.** exp < 1 makes the influence function `p·|e|^(p-1)` *decrease* with |e|:
at |e| = 0.1 the gradient is 135 against MSE's 1e5, a **740× reduction**. The same change that stops
the blur also tells the model to give up on its worst points — and the near-surface failures above
*are* large residuals. If `val/viou` drops while `val/recon` looks fine, this is why, and the first
thing to try is flattening `m_exponents` to `[2, 2, 1.5, 1]`.

**Confounding, acknowledged:** three axes move at once, so a win does not attribute to any one of
them. Deliberate given the schedule — this run answers "does the stack beat 83 V-IoU at all". The
partner isolating the exponent is the same config with `m_exponents` removed. Validation always runs
at m = max(m_values) = 64, so `val/viou` stays directly comparable to the non-MRL runs.

### lambda-VAE implementation: two NaN incidents

Both killed a real run; both are why `Gaussian.sample` compares in **log space** rather than the
obvious way.

1. `lam` comes from a channel-mean sigma but multiplies every element's sigma, and `sigma^lam` only
   shrinks while sigma < 1 — above 1 it explodes. 5.2% of elements have sigma > 1 (max 8.4); at
   lam ≈ 12 that sent the noise to **1089** against an SDF target range of ±1.6, and NaN'd at epoch 8.
   Fix: Eq.16's `max(1, ·)` must be applied **elementwise**, not to the channel statistic.
2. Writing that clamp as `minimum(sigma, sigma**lam)` NaN'd even earlier, at epoch 6: logvar clamps at
   +20 so sigma can reach exp(10) = 22026, and `22026**10.7` overflows to inf. `minimum()` picks sigma
   in the forward, but autograd still evaluates pow's local gradient and multiplies it by the zero
   routed to that branch — **0 × inf = NaN**. Comparing `log(sigma)` against `lam·log(sigma)` makes
   exactly the same selection, never evaluates the overflowing power, and leaves a finite gradient on
   the winning branch.

`kl_divergence()` is deliberately left alone: charging the KL for the **original** sigma while the
decoder only ever sees the reduced noise is the entire mechanism (Eq.10).
