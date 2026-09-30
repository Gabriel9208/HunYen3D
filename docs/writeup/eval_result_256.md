# Reconstruction evaluation

- generated: 2026-09-22 02:53
- split: `data/test`, 600 shapes, seed 0
- marching cubes resolution 256, F-score tau 0.02
- V-IoU/S-IoU on the full 250k-point SDF bank per shape
- latent: z ~ q(z|x), one draw per shape (matches training)

| metric | lambda | disjoint | vanilla |
|---|---|---|---|
| epoch | 67 | 65 | 64 |
| shapes evaluated | 600 | 600 | 600 |
| **mixed bank (near+uniform)** |  |  |  |
| V-IoU  pooled | 83.44 | 81.15 | 80.31 |
| V-IoU  per-shape | 83.35 | 81.09 | 80.23 |
| S-IoU  pooled | 78.32 | 75.51 | 74.58 |
| S-IoU  per-shape | 78.62 | 75.89 | 74.94 |
| **uniform block only (3DShape2VecSet)** |  |  |  |
| V-IoU  pooled | 94.37 | 93.53 | 93.09 |
| V-IoU  per-shape | 91.23 | 89.91 | 89.35 |
| **marching cubes @ 256** |  |  |  |
| Chamfer-L1 (lower better) | 0.01333 | 0.01389 | 0.01444 |
| F-score@0.02 | 0.9830 | 0.9744 | 0.9708 |
| Normal consistency | 0.9545 | 0.9491 | 0.9458 |
| shapes with no surface | 0 | 0 | 0 |

## checkpoints

- **lambda** — `/data/outputs/vae_lambda_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16/2026-09-14_00-23-59/checkpoints/best.pt` (epoch 67)
- **disjoint** — `/data/outputs/vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16/2026-09-12_15-43-29/checkpoints/best.pt` (epoch 65)
- **vanilla** — `/data/outputs/vae_converge_kl1e-4_lr3.5e-6_1024_l8_16/2026-09-12_10-25-03/checkpoints/best.pt` (epoch 64)
