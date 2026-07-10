# HunYen3D

A from-scratch study re-implementation of the **Hunyuan3D-2 ShapeVAE**, taking
Hunyuan3D-2.1 as the starting point.

## Motivation

This is a learning/research project, not a product. Its aims, in order:

- **Reproduce** the Hunyuan3D-2 ShapeVAE from scratch (architecture and training loop
  written by hand, starting from Hunyuan3D-2.1 as the reference), to build a solid mental
  model of how the method actually works.
- **Explore** techniques that improve the results — progressively reading other papers and
  applying their ideas — to cultivate research method and perspective.
- **Long-term: become an experiment playground.** The goal is a sandbox where new
  inspirations, formed by colliding different mindsets with existing methods, can be put to
  the test and validated against real runs.

**Where the project is today:** still in the *perception-building and reproduction* phase.
The baseline must be solid before it becomes a playground for idea validation.

### Scope & expectations

Compute and memory are limited, so experiments deliberately target datasets that the
available hardware can actually support. The emphasis is on the **essence of the methods and
on the observed results** — **not on chasing SOTA benchmark numbers**, which would require
large-scale data this project does not pursue. Read the numbers here as evidence about
*methods*, not as leaderboard entries.

## What's implemented

- **ShapeVAE** — encoder, decoder, attention, and mesh preprocessing, hand-written
  (`src/model/`).
- **Training scaffold** — a generic, model-agnostic plain-PyTorch `Trainer` driven by
  **Hydra** configs, logged with **Weights & Biases**, with **tqdm** progress bars
  (`src/engine/`). The trainer depends only on a small `Task` contract, not on the VAE.

## Architecture overview

```
mesh (.obj)
   │  preprocess: normalize → surface/important sampling → Fourier(xyz) + raw normals
   ▼
encoder:  CrossAttn (point cloud → latents) → SelfAttn × 8 ─┐
                                                            ▼
                                                      KL bottleneck
                                                            │
decoder:  SelfAttn × 16 → CrossAttn (grid query points → features) → SDF
   │
   ▼
loss = MSE(pred SDF, gt SDF) + γ · KL
```

- Encoder/decoder/VAE: `src/model/shape/VAE/{encoder,decoder,vae}.py`
- Attention block: `src/model/general/attention.py`
- Preprocessing (FourierEmbedder + Mesh2Query + Preprocessor): `src/model/shape/preprocess.py`
- VAE loss (`VAETask`): `src/engine/task.py`

Default model dims (`configs/model/shape_vae.yaml`): `num_latents=6144`, `width=1024`,
`num_head=16`, 8 encoder / 16 decoder layers, `latent_dim=64`. Input widths:
`encoder_pe_dim=42` (Fourier(xyz)=39 + 3 normals), `decoder_pe_dim=39` (Fourier(xyz) only).

## Repository layout

```
main.py                     Hydra entry point → src.engine.runner.run
configs/                    Hydra config groups (see Configuration)
  config.yaml               top-level defaults list
  model/ task/ data/        model, loss/task, dataset configs
  optimizer/ scheduler/     adamw, cosine
  trainer/ wandb/ preprocess/
  experiment/               base_overfit.yaml, base_first_train.yaml (used via +experiment=)
scripts/
  build_cache.py            offline preprocessing-cache builder (run before training)
  evaluate.py               checkpoint geometry eval (SDF metrics; +viz adds Chamfer + PNG)
src/
  model/                    ShapeVAE (encoder/decoder/attention/preprocess)
  engine/                   Trainer + run (runner.py), builder, task, data,
                            checkpoint, logging, utils
docs/RESEARCH_LOG.md        running log of papers read, methods tried, observations
```

## Installation

Managed with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13):

```bash
uv sync
```

This installs PyTorch, Hydra, wandb, tqdm, trimesh, mesh-to-sdf, scipy, and datasets. A CUDA
GPU is recommended; the trainer supports bf16 autocast to fit the model into limited memory.

## Usage

### 1. Data layout

Place `.obj` meshes under (both directories are gitignored):

```
data/train/*.obj
data/val/*.obj
```

### 2. Build the preprocessing cache (run first)

Training defaults to **light** mode, which reads a precomputed cache. Build it once per
experiment before training:

```bash
uv run python scripts/build_cache.py +experiment=base_first_train
```

This materializes the expensive, deterministic preprocessing (mesh→SDF, surface sampling)
into `cache/<mesh_stem>.pt`. See [Preprocessing & caching](#preprocessing--caching).

### 3. Train

```bash
# Full training run
uv run python main.py +experiment=base_first_train

# Overfit a single mesh — the standard sanity check before any real run
uv run python main.py +experiment=base_overfit
```

### Common overrides

Any config field can be overridden on the command line:

```bash
uv run python main.py +experiment=base_first_train trainer.max_epochs=200 optimizer.lr=1e-4
uv run python main.py +experiment=base_first_train scheduler.T_max=5000   # pin the cosine period
uv run python main.py +experiment=base_first_train scheduler=null         # disable LR scheduling
uv run python main.py +experiment=base_first_train trainer.resume=/abs/path/to/last.pt
```

By default `scheduler.T_max` is left unset (`???`) and the runner computes it automatically
as `len(train_loader) × max_epochs` (one scheduler step per optimizer step); set it
explicitly only to pin the cosine period.

## Configuration (Hydra)

`configs/config.yaml` composes the run from config groups:

```yaml
defaults:
  - model: shape_vae
  - task: vae
  - data: shape_vae
  - optimizer: adamw
  - scheduler: cosine
  - trainer: default
  - wandb: default
  - preprocess: default
```

- **Experiment configs** (`configs/experiment/*.yaml`, `# @package _global_`) bundle all the
  overrides for a run and are activated with `+experiment=<name>`.
- **Override syntax**: `group=value` swaps a group, `field=value` sets a value, `+field=value`
  adds one, `~field` removes one.
- `hydra.job.chdir: true` — each run executes from its own output directory, so
  `checkpoints/` and logs are isolated per run.

## Preprocessing & caching

Per-mesh preprocessing has two stages, split by cost and determinism:

- **heavy** — expensive and deterministic (mesh→SDF via `mesh-to-sdf`, surface sampling).
  This is the only real bottleneck (~seconds per mesh) and is materialized **offline** by
  `scripts/build_cache.py` into `cache/<mesh_stem>.pt`.
- **light** — cheap and stochastic (subsample / FPS / SDF subset / Fourier), run **per epoch**
  inside the dataset so each epoch still sees fresh sampling.

`ObjMeshDataset` (`src/engine/data.py`) defaults to **light** mode: it loads the cache and
applies the light stage. Each cache file stores a **parameter signature**; if you change the
heavy preprocessing parameters (e.g. `num_surface_samples`, `n_query_*`), loading a stale
cache raises an error telling you to rebuild (`force=true`). For ad-hoc runs without a cache,
set `data.train.dataset.mode=heavy` (slower; recomputes everything on the fly).

For reproducibility, set `fixed_seed` on a dataset to freeze the light-stage sampling (used by
the overfit experiment so the data stops changing and the loss can converge cleanly).

## Research log

Papers read, methods tried, hypotheses, and result observations are tracked in
[`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

## References

- **Hunyuan3D 2.0** — *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured
  3D Assets Generation* (Tencent Hunyuan3D Team). The ShapeVAE reproduced here corresponds to
  §3.1 / Fig. 3. Paper PDF: `Hunyuan2.pdf` (repo root).
- **Hunyuan3D-2.1** — the open-source release used as the implementation starting point.

Further references will be added here as more papers are read and applied.
