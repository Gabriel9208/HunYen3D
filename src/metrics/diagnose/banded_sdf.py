from __future__ import annotations

import torch

# Bands by |gt_sdf| distance (custom); the far upper bound is large to cover everything.
DEFAULT_BANDS = [(0.0, 0.02, "near |gt|<0.02"), (0.02, 0.1, "mid  0.02-0.1"), (0.1, 9.9, "far  >0.1")]


class BandedSDFMetrics:
    """Custom diagnostic metric (not a literature standard, for internal debugging):

    per-band (near/mid/far by |gt|) RMS, normalised against the predict-all-0 baseline.
    Designed because the overall average is dominated by easy far-field points and hides
    which band is actually failing.

    Per band:
      rms        = sqrt(mean err²), in SDF units so it compares directly to |gt|.
      pred0/pred1 = MSE of predicting all 0 / all 1 (context baselines).
      rms_ratio  = rms / rms(predict-all-0). <1 = beats predicting zero (band learned
                   something); ≈1 = learned nothing; >1 = worse than predicting zero.

    (Sign-accuracy was dropped: it saturates below the RMS floor / marching-cubes
    resolution, so it is not a defensible quality read — RMS is the continuous replacement.
    The actual reconstruction target stays IoU / Chamfer / mesh.)
    """

    def __init__(self, bands=DEFAULT_BANDS) -> None:
        self.bands = bands

    def __call__(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> dict:
        pred, gt = pred_sdf.flatten(), gt_sdf.flatten()
        a = gt.abs()
        out: dict = {"bands": {}}
        for lo, hi, name in self.bands:
            m = (a >= lo) & (a < hi)
            if not m.any():
                continue
            mse = (pred[m] - gt[m]).pow(2).mean().item()
            pred0 = gt[m].pow(2).mean().item()             # predict all 0 (== the RMS baseline)
            out["bands"][name] = {
                "frac": m.float().mean().item() * 100,
                "mse": mse,
                "rms": mse ** 0.5,
                "pred0": pred0,
                "pred1": (1.0 - gt[m]).pow(2).mean().item(),  # predict all 1
                "rms_ratio": (mse ** 0.5) / max(pred0 ** 0.5, 1e-9),
            }
        return out


if __name__ == "__main__":
    # near band fit perfectly (rms 0, ratio 0); far band predicted 0 (rms == baseline, ratio 1).
    gt = torch.tensor([0.005, -0.01, 0.5, -0.5])       # two near, two far
    pred = torch.tensor([0.005, -0.01, 0.0, 0.0])      # near exact, far = predict-zero
    r = BandedSDFMetrics()(pred, gt)
    near, far = r["bands"]["near |gt|<0.02"], r["bands"]["far  >0.1"]
    assert abs(near["rms"] - 0.0) < 1e-6, r
    assert abs(near["rms_ratio"] - 0.0) < 1e-6, r
    assert abs(far["rms"] - 0.5) < 1e-6, r
    assert abs(far["rms_ratio"] - 1.0) < 1e-6, r         # our pred==0 equals the baseline
    assert abs(near["frac"] - 50.0) < 1e-6, r
    print("OK: BandedSDFMetrics")
