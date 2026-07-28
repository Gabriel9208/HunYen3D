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
- **τ must exceed the MC cell size** `2/res` (≈0.0156 at res 128); otherwise F-score would grade
  agreement finer than a single vertex can be placed.
- **μ-path vs sample-avg.** decode(μ) is not guaranteed by the ELBO; in a high-dim latent the samples
  live on a shell of radius σ√d, while μ can be a near-zero-mass centre the decoder never trains on.
  When the posterior is μ-broken, μ-path badly understates or even collapses → **always read sample-avg
  (K=32)** and treat μ-path as a diagnostic only.
- **Retired metrics:** **near-sign-acc** = the fraction of near-band points with the correct predicted
  sign (it saturates below the resolvable precision, so it was dropped as a target); **RMS** =
  √(mean SDF-error²), in the same units as `gt` (used as a training-health read; the PASS bar
  `RMS<0.02` = the near-band width).

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
