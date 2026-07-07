from __future__ import annotations

import torch

# Bands by |gt_sdf| distance (custom); the far upper bound is large to cover everything.
DEFAULT_BANDS = [(0.0, 0.02, "near |gt|<0.02"), (0.02, 0.1, "mid  0.02-0.1"), (0.1, 9.9, "far  >0.1")]


class BandedSDFMetrics:
    """Custom diagnostic metric (not a literature standard, for internal debugging):

    overall sign-acc + per-band (near/mid/far by |gt|) sign-acc / MSE / pred0-pred1
    baselines. Designed because the overall average is inflated by easy far-field
    points and hides that the near surface is a coin flip.
    pred0 = MSE of predicting all 0, pred1 = MSE of predicting all 1: model MSE ≈
    pred0 means that band learned nothing.
    """

    def __init__(self, bands=DEFAULT_BANDS) -> None:
        self.bands = bands

    def __call__(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> dict:
        pred, gt = pred_sdf.flatten(), gt_sdf.flatten()
        a = gt.abs()
        out = {"overall_sign_acc": (pred.sign() == gt.sign()).float().mean().item() * 100, "bands": {}}
        for lo, hi, name in self.bands:
            m = (a >= lo) & (a < hi)
            if not m.any():
                continue
            out["bands"][name] = {
                "frac": m.float().mean().item() * 100,
                "mse": (pred[m] - gt[m]).pow(2).mean().item(),
                "pred0": gt[m].pow(2).mean().item(),           # predict all 0
                "pred1": (1.0 - gt[m]).pow(2).mean().item(),   # predict all 1
                "sign_acc": (pred[m].sign() == gt[m].sign()).float().mean().item() * 100,
            }
        return out


if __name__ == "__main__":
    # near band all correct, far band all wrong -> per-band sign-acc should be 100 / 0
    gt = torch.tensor([0.005, -0.01, 0.5, -0.5])       # two near, two far
    pred = torch.tensor([0.005, -0.01, -0.5, 0.5])     # near correct, far flipped
    r = BandedSDFMetrics()(pred, gt)
    assert abs(r["bands"]["near |gt|<0.02"]["sign_acc"] - 100.0) < 1e-6, r
    assert abs(r["bands"]["far  >0.1"]["sign_acc"] - 0.0) < 1e-6, r
    assert abs(r["bands"]["near |gt|<0.02"]["frac"] - 50.0) < 1e-6, r
    assert abs(r["overall_sign_acc"] - 50.0) < 1e-6, r  # 2/4 correct
    print("OK: BandedSDFMetrics")
