from __future__ import annotations

import torch

# 按 |gt_sdf| 距離分帶(自製);far 上界取大值涵蓋全部
DEFAULT_BANDS = [(0.0, 0.02, "near |gt|<0.02"), (0.02, 0.1, "mid  0.02-0.1"), (0.1, 9.9, "far  >0.1")]


class BandedSDFMetrics:
    """自製診斷指標(非文獻標準,內部除錯用):

    整體 sign-acc + 按 |gt| 分帶(近/中/遠)的 sign-acc / MSE / pred0-pred1 基準。
    針對「overall 平均會被遠場好猜的點灌高、藏住近表面其實是擲硬幣」而設計。
    pred0 = 全吐 0 的 MSE、pred1 = 全吐 1 的 MSE:model MSE ≈ pred0 代表該帶等於沒學。
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
                "pred0": gt[m].pow(2).mean().item(),           # 全 predict 0
                "pred1": (1.0 - gt[m]).pow(2).mean().item(),   # 全 predict 1
                "sign_acc": (pred[m].sign() == gt[m].sign()).float().mean().item() * 100,
            }
        return out


if __name__ == "__main__":
    # near 帶全對、far 帶全錯 → 各帶 sign-acc 應為 100 / 0
    gt = torch.tensor([0.005, -0.01, 0.5, -0.5])       # 兩近、兩遠
    pred = torch.tensor([0.005, -0.01, -0.5, 0.5])     # 近對、遠反號
    r = BandedSDFMetrics()(pred, gt)
    assert abs(r["bands"]["near |gt|<0.02"]["sign_acc"] - 100.0) < 1e-6, r
    assert abs(r["bands"]["far  >0.1"]["sign_acc"] - 0.0) < 1e-6, r
    assert abs(r["bands"]["near |gt|<0.02"]["frac"] - 50.0) < 1e-6, r
    assert abs(r["overall_sign_acc"] - 50.0) < 1e-6, r  # 2/4 對
    print("OK: BandedSDFMetrics")
