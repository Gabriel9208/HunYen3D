from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics._pointcloud import sample_points, unit_ball_norm

# Vendored upstream: third_party/Uni3D (baaivision/Uni3D). See third_party/README.md.
_UNI3D_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "Uni3D")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1).mean())


class Uni3DMetric:
    """Uni3D-I: cosine similarity between a shape's Uni3D point embedding and the input
    image's EVA-CLIP image embedding, in Uni3D's joint space (↑ better, range [-1, 1]).

    Unlike ULIP, Uni3D's image tower is a SEPARATE open_clip (EVA-CLIP) model, so this
    holds two objects: `point_model.encode_pc(pc)` and `clip_model.encode_image(img)`.
    Pass both `None` to build from `third_party/Uni3D` via `build_uni3d` (needs timm +
    open_clip + downloads); offline callers inject stubs.

    Uni3D consumes 10k xyz+rgb points; colourless meshes get a constant grey channel.
    """

    def __init__(self, ckpt: str | None = None, point_model=None, clip_model=None,
                 image_preprocess=None, num_points: int = 10000, default_color: float = 0.4,
                 device: str = "cpu") -> None:
        self.num_points = num_points
        self.default_color = default_color
        self.device = device
        if point_model is None or clip_model is None:
            point_model, clip_model, image_preprocess = build_uni3d(ckpt, device)
        self.point_model = point_model.eval() if hasattr(point_model, "eval") else point_model
        self.clip_model = clip_model.eval() if hasattr(clip_model, "eval") else clip_model
        self.image_preprocess = image_preprocess
        # point_model stays fp32: the compiled pointnet2_ops CUDA kernel (furthest_point_sample)
        # hard-asserts float32 input, it's not a memory-driven choice. clip_model is fp16 -- that's
        # the actual size driver (EVA02-Enormous is ~10GB at fp32). Inputs must match each model or
        # encode_pc/encode_image hit "mat1 and mat2 must have the same dtype".
        self.point_dtype = next(self.point_model.parameters()).dtype if hasattr(point_model, "parameters") \
            else torch.float32
        self.clip_dtype = next(self.clip_model.parameters()).dtype if hasattr(clip_model, "parameters") \
            else torch.float32

    @torch.no_grad()
    def __call__(self, shape, image) -> float:
        pts = sample_points(shape, self.num_points)
        if pts is None:
            return float("nan")
        pc = self._to_pc(unit_ball_norm(pts))
        pf = self.point_model.encode_pc(pc.to(self.device, dtype=self.point_dtype))
        imf = self.clip_model.encode_image(self._to_images(image).to(self.device, dtype=self.clip_dtype))  # (V, D)
        imf = imf.mean(0, keepdim=True)                          # average over views -> (1, D)
        return _cosine(pf, imf)

    def _to_pc(self, pts: np.ndarray) -> torch.Tensor:
        xyz = torch.from_numpy(pts).float()
        rgb = torch.full_like(xyz, self.default_color)
        return torch.cat([xyz, rgb], dim=-1).unsqueeze(0)        # (1, N, 6) xyz+rgb

    def _to_images(self, image) -> torch.Tensor:
        """One or many views -> (V, 3, H, W), batched through the image encoder in one pass."""
        if torch.is_tensor(image):
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
def _in_uni3d_dir():
    cwd, path0 = os.getcwd(), list(sys.path)
    sys.path.insert(0, os.path.abspath(_UNI3D_DIR))
    os.chdir(os.path.abspath(_UNI3D_DIR))
    try:
        yield
    finally:
        os.chdir(cwd)
        sys.path[:] = path0


def build_uni3d(ckpt: str | None, device: str,
                clip_model_name: str = "EVA02-E-14-plus", clip_pretrained: str = "laion2b_s9b_b144k"):
    """Construct the vendored Uni3D point model + an open_clip EVA-CLIP image tower, optionally
    loading a checkpoint. NOT exercised offline (needs timm/open_clip weights); mirrors
    Uni3D/main.py: `create_uni3d(args)` + `open_clip.create_model_and_transforms(...)` +
    `load_state_dict(torch.load(ckpt)["module"])`. The `args` must match the released
    checkpoint's training config (Uni3D-giant defaults below; confirm against Uni3D/scripts).
    """
    from types import SimpleNamespace

    import open_clip

    args = SimpleNamespace(                       # Uni3D-giant defaults — confirm vs your ckpt
        pc_model="eva_giant_patch14_560", pretrained_pc=None, drop_path_rate=0.0,
        pc_feat_dim=1408, embed_dim=1024, group_size=32, num_group=512, pc_encoder_dim=512,
        patch_dropout=0.0,
    )
    with _in_uni3d_dir():
        from models.uni3d import create_uni3d
        point_model = create_uni3d(args)
    if ckpt is not None:
        # weights_only=False: PyTorch >=2.6 defaults to True and rejects this checkpoint's numpy
        # scalars; the official Uni3D-giant release from HF is a trusted source, ckpt= is caller-supplied.
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        point_model.load_state_dict(state.get("module", state), strict=False)
    # point_model: fp32 only (pointnet2_ops' compiled FPS kernel asserts float32 input, unrelated
    # to memory). clip_model: fp16 -- EVA02-Enormous is the actual ~10GB-at-fp32 memory driver, and
    # nothing here needs its own custom CUDA kernel, so halving it is a plain, safe memory saving.
    point_model = point_model.to(device=device)
    clip_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=clip_model_name, pretrained=clip_pretrained)
    return point_model, clip_model.to(device=device, dtype=clip_dtype), preprocess


if __name__ == "__main__":
    class _PointStub:
        def __init__(self, v):
            self.v = v
        def encode_pc(self, pc):
            return self.v.expand(pc.shape[0], -1)

    class _ClipStub:
        def __init__(self, v):
            self.v = v
        def encode_image(self, img):
            return self.v.expand(img.shape[0], -1)

    v = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    m = Uni3DMetric(point_model=_PointStub(v), clip_model=_ClipStub(v),
                    image_preprocess=lambda im: torch.zeros(3, 4, 4), num_points=256)
    pts = np.random.rand(2000, 3).astype(np.float32)
    s = m(pts, torch.zeros(3, 4, 4))
    assert abs(s - 1.0) < 1e-5, s
    print(f"OK: Uni3DMetric identical embeddings cos={s:.4f}")

    s3 = m(pts, torch.zeros(3, 3, 4, 4))                            # 3 views -> averaged
    assert abs(s3 - s) < 1e-5 and -1.0 <= s3 <= 1.0, s3
    print(f"OK: Uni3DMetric 3-view average cos={s3:.4f}")

    m2 = Uni3DMetric(point_model=_PointStub(v), clip_model=_ClipStub(torch.tensor([[0.0, 1.0, 0.0, 0.0]])),
                     image_preprocess=lambda im: torch.zeros(3, 4, 4), num_points=256)
    s2 = m2(pts, torch.zeros(3, 4, 4))
    assert abs(s2) < 1e-5 and -1.0 <= s2 <= 1.0, s2
    print(f"OK: Uni3DMetric orthogonal cos={s2:.4f}")

    assert np.isnan(m(None, None)) and np.isnan(m(np.zeros((0, 3)), None))
    print("OK: Uni3DMetric nan guard on empty geometry")
