from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics._pointcloud import sample_points, unit_ball_norm

# Vendored upstream: third_party/ULIP (salesforce/ULIP). See third_party/README.md.
_ULIP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ULIP")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between two (B, D) embeddings (L2-normalized first)."""
    return float((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1).mean())


class ULIPMetric:
    """ULIP-I: cosine similarity between a shape's ULIP point embedding and the input
    image's ULIP image embedding, in ULIP's joint CLIP space (↑ better, range [-1, 1]).

    `model` must expose ULIP's `encode_pc(pc)` and `encode_image(img)` (the vendored
    `ULIP2_WITH_OPENCLIP` / `ULIP_WITH_IMAGE` objects do). Pass `model=None` to build it
    from `third_party/ULIP` via `build_ulip` (needs open_clip + a checkpoint + a large
    open_clip download — offline callers inject a `model` instead).

    ULIP-2 consumes 10k COLORED points; colourless meshes get a constant grey so the
    channel exists — set `default_color` to match your checkpoint's expectation.
    """

    def __init__(self, ckpt: str | None = None, model=None, image_preprocess=None,
                 num_points: int = 10000, default_color: float = 0.4,
                 model_name: str = "ULIP2_PointBERT_Colored", device: str = "cpu") -> None:
        self.num_points = num_points
        self.default_color = default_color
        self.device = device
        if model is None:
            model, image_preprocess = build_ulip(model_name, ckpt, device)
        self.model = model.eval() if hasattr(model, "eval") else model
        self.image_preprocess = image_preprocess

    @torch.no_grad()
    def __call__(self, shape, image) -> float:
        pts = sample_points(shape, self.num_points)
        if pts is None:
            return float("nan")
        pc = self._to_pc(unit_ball_norm(pts))
        pf = self.model.encode_pc(pc.to(self.device))
        imf = self.model.encode_image(self._to_images(image).to(self.device))  # (V, D)
        imf = imf.mean(0, keepdim=True)                          # average over views -> (1, D)
        return _cosine(pf, imf)

    def _to_pc(self, pts: np.ndarray) -> torch.Tensor:
        xyz = torch.from_numpy(pts).float()                      # (N, 3)
        rgb = torch.full_like(xyz, self.default_color)           # colourless -> constant grey
        return torch.cat([xyz, rgb], dim=-1).unsqueeze(0)        # (1, N, 6)

    def _to_images(self, image) -> torch.Tensor:
        """One or many views -> (V, 3, H, W), batched through the image encoder in one pass."""
        if torch.is_tensor(image):                               # already preprocessed (…,3,H,W)
            return image if image.ndim == 4 else image.unsqueeze(0)
        from PIL import Image
        views = image if isinstance(image, (list, tuple)) else [image]
        tens = []
        for im in views:
            if isinstance(im, str):
                im = Image.open(im).convert("RGB")
            tens.append(self.image_preprocess(im))
        return torch.stack(tens)                                 # (V, 3, H, W)


@contextmanager
def _in_ulip_dir():
    """ULIP's factory imports `models.*` and reads `./models/pointbert/*.yaml` by relative
    path, so its own dir must be cwd + on sys.path while building."""
    cwd, path0 = os.getcwd(), list(sys.path)
    sys.path.insert(0, os.path.abspath(_ULIP_DIR))
    os.chdir(os.path.abspath(_ULIP_DIR))
    try:
        yield
    finally:
        os.chdir(cwd)
        sys.path[:] = path0


def build_ulip(model_name: str, ckpt: str | None, device: str):
    """Construct a vendored ULIP model + its open_clip image transform, optionally loading a
    checkpoint. NOT exercised in the offline self-check (needs open_clip weights); mirrors
    ULIP/main.py's `getattr(models, args.model)(args)` + `load_state_dict(ckpt["state_dict"])`.
    The `args` fields must match the released checkpoint's training config (see ULIP/scripts).
    """
    from types import SimpleNamespace

    import open_clip  # noqa: F401  (ULIP factory imports it)

    args = SimpleNamespace(npoints=10000)  # extend to match your checkpoint's training args
    with _in_ulip_dir():
        # models/__init__.py is empty, so `import models` does not expose ULIP_models' factory
        # functions (e.g. ULIP2_PointBERT_Colored) as package attributes -- import the submodule.
        from models import ULIP_models as ulip_models
        model = getattr(ulip_models, model_name)(args=args)
    if ckpt is not None:
        # weights_only=False: PyTorch >=2.6 defaults to True and rejects this checkpoint's numpy
        # scalars; the official ULIP-2 release from HF is a trusted source, ckpt= is caller-supplied.
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        # the released checkpoint was saved from a DDP-wrapped model (every key prefixed "module.");
        # strict=False was silently swallowing a 0/1202-key match without this and leaving the point
        # encoder at its random init. Stripping the prefix covers all 226 point_encoder.* keys; the
        # remaining unmatched keys are open_clip_model.* (CLIP backbone), which this checkpoint never
        # carries -- that tower is already loaded from its own pretrained='laion2b_s39b_b160k' below.
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    # image transform only (pretrained=None -> no weight download, just the preprocess)
    preprocess = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=None)[2]
    return model, preprocess


if __name__ == "__main__":
    # Weight-free self-check: inject a stub ULIP model to exercise sampling + cosine + guards.
    class _Stub:
        def __init__(self, pcv, imv):
            self.pcv, self.imv = pcv, imv
        def encode_pc(self, pc):
            return self.pcv.expand(pc.shape[0], -1)
        def encode_image(self, img):
            return self.imv.expand(img.shape[0], -1)

    v = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    stub = _Stub(v, v)
    m = ULIPMetric(model=stub, image_preprocess=lambda im: torch.zeros(3, 4, 4), num_points=256)
    pts = np.random.rand(2000, 3).astype(np.float32)
    s = m(pts, torch.zeros(3, 4, 4))
    assert abs(s - 1.0) < 1e-5, s                                   # aligned embeddings -> 1
    print(f"OK: ULIPMetric identical embeddings cos={s:.4f}")

    s3 = m(pts, torch.zeros(3, 3, 4, 4))                            # 3 views (V,3,H,W) -> averaged
    assert abs(s3 - s) < 1e-5 and -1.0 <= s3 <= 1.0, s3            # identical views -> same as 1 view
    print(f"OK: ULIPMetric 3-view average cos={s3:.4f}")

    m2 = ULIPMetric(model=_Stub(v, torch.tensor([[0.0, 1.0, 0.0, 0.0]])),
                    image_preprocess=lambda im: torch.zeros(3, 4, 4), num_points=256)
    s2 = m2(pts, torch.zeros(3, 4, 4))
    assert abs(s2) < 1e-5, s2                                       # orthogonal -> 0
    assert -1.0 <= s2 <= 1.0
    print(f"OK: ULIPMetric orthogonal cos={s2:.4f}")

    assert np.isnan(m(None, None)) and np.isnan(m(np.zeros((0, 3)), None))  # empty -> nan
    print("OK: ULIPMetric nan guard on empty geometry")
