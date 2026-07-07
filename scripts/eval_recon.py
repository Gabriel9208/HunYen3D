"""幾何重建評估:回答「MSE 0.0x 到底是好是壞」。

MSE 會被遠場好猜的點拉低而失真,所以這裡按 |gt_sdf| 距離分帶看 MSE,並算幾何上有意義的:
  - sign accuracy(內/外判對比例)——尤其近表面那帶才是關鍵
  - occupancy IoU(sdf<0 視為內部)
用固定 mu(不取樣)在每個形狀自己的 query points 上重建。

用法:
  uv run python scripts/eval_recon.py +experiment=small \
      +ckpt=/abs/path/last.pt +shapes=16
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.engine.utils import get_device, set_seed
from src.metrics import BandedSDFMetrics, OccupancyIoU


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(0)
    device = get_device(cfg.trainer.get("device", "auto"))
    model = instantiate(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(cfg.ckpt, map_location=device, weights_only=False)["model"])

    ds = instantiate(cfg.data.val.dataset)
    n = min(int(cfg.get("shapes", 16)), len(ds))

    preds, gts = [], []
    with torch.no_grad():
        for i in range(n):
            it = ds[i]
            mu, _ = model.encoder(it["query"].unsqueeze(0).to(device), it["data"].unsqueeze(0).to(device))
            pred = model.decode(mu, it["query_points"].unsqueeze(0).to(device)).squeeze(0).cpu()
            preds.append(pred)
            gts.append(it["gt_sdf"])
    pred = torch.cat(preds).flatten()
    gt = torch.cat(gts).flatten()

    diag = BandedSDFMetrics()(pred, gt)
    iou = OccupancyIoU()(pred, gt)

    print(f"\nshapes={n}  points={gt.numel():,}  ckpt={cfg.ckpt}")
    print(f"overall MSE      : {(pred - gt).pow(2).mean():.5f}")
    print(f"overall RMS      : {(pred - gt).pow(2).mean().sqrt():.4f}  (SDF 尺度 ~[-1,1])")
    print(f"sign accuracy    : {diag['overall_sign_acc']:.2f}%  (整體;遠場好猜會灌高)")
    print(f"occupancy IoU    : {iou * 100:.2f}%")

    print("按距離分帶(pred0/pred1 = 全吐 0 / 全吐 1 的對照基準;model MSE ≈ pred0 = 該帶等於沒學):")
    for name, b in diag["bands"].items():
        print(f"  {name:<16} 佔比{b['frac']:5.1f}%  MSE={b['mse']:.5f}  "
              f"(pred0={b['pred0']:.5f} pred1={b['pred1']:.5f})  sign-acc={b['sign_acc']:5.1f}%")


if __name__ == "__main__":
    main()
