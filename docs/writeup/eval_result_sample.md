# Reconstruction evaluation

- generated: 2026-09-13 23:50
- split: `data/test`, 2592 shapes, seed 0
- marching cubes resolution 128, F-score tau 0.02
- V-IoU/S-IoU on the full 250k-point SDF bank per shape
- latent: z ~ q(z|x), one draw per shape (matches training)

| metric | disjoint | vanilla |
|---|---|---|
| epoch | 65 | 64 |
| shapes evaluated | 2592 | 2592 |
| **mixed bank (near+uniform)** |  |  |
| V-IoU  pooled | 81.43 | 80.63 |
| V-IoU  per-shape | 81.42 | 80.60 |
| S-IoU  pooled | 75.82 | 74.93 |
| S-IoU  per-shape | 76.21 | 75.30 |
| **uniform block only (3DShape2VecSet)** |  |  |
| V-IoU  pooled | 93.67 | 93.25 |
| V-IoU  per-shape | 90.39 | 89.80 |
| **marching cubes @ 128** |  |  |
| Chamfer-L1 (lower better) | 0.01364 | 0.01428 |
| F-score@0.02 | 0.9749 | 0.9712 |
| Normal consistency | 0.9495 | 0.9477 |
| shapes with no surface | 0 | 0 |

## checkpoints

- **disjoint** — `outputs/vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16/2026-09-12_15-43-29/checkpoints/best.pt` (epoch 65)
- **vanilla** — `outputs/vae_converge_kl1e-4_lr3.5e-6_1024_l8_16/2026-09-12_10-25-03/checkpoints/best.pt` (epoch 64)
