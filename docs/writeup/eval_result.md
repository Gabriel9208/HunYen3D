# Reconstruction evaluation

- generated: 2026-09-13 07:05
- split: `data/test`, 2592 shapes, seed 0
- marching cubes resolution 128, F-score tau 0.02
- V-IoU/S-IoU on the full 250k-point SDF bank per shape

| metric | disjoint | vanilla |
|---|---|---|
| epoch | 65 | 64 |
| shapes evaluated | 2592 | 2592 |
| **mixed bank (near+uniform)** |  |  |
| V-IoU  pooled | 64.07 | 54.96 |
| V-IoU  per-shape | 62.73 | 52.80 |
| S-IoU  pooled | 54.30 | 42.65 |
| S-IoU  per-shape | 53.43 | 41.58 |
| **uniform block only (3DShape2VecSet)** |  |  |
| V-IoU  pooled | 87.69 | 85.57 |
| V-IoU  per-shape | 77.91 | 70.89 |
| **marching cubes @ 128** |  |  |
| Chamfer-L1 (lower better) | 0.03097 | 0.04591 |
| F-score@0.02 | 0.8612 | 0.7609 |
| Normal consistency | 0.9298 | 0.9133 |
| shapes with no surface | 5 | 5 |

## checkpoints

- **disjoint** — `outputs/vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16/2026-09-12_15-43-29/checkpoints/best.pt` (epoch 65)
- **vanilla** — `outputs/vae_converge_kl1e-4_lr3.5e-6_1024_l8_16/2026-09-12_10-25-03/checkpoints/best.pt` (epoch 64)
