import os

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes


class Postprocess:
    """latent -> mesh。在 [-1,1]³ 撒 grid、decode 出 SDF、marching cubes 取零等值面,存成 .off。

    fourier_embedder 必須是訓練時同一顆(頻率 / include_pi / include_input 要一致),故由外部傳入
    (呼叫端用 preprocessor.fourier_embedder)。

    輸出檔名由呼叫端組:results/{experiment_name}/{原始 folder}/{原始 filename}.off,
    例如 results/{exp}/02691156/1a04e3eab45ca15dd86060f189eb133.off。
    """

    def __init__(self, fourier_embedder, resolution: int = 128, chunk: int = 65536):
        self.fourier_embedder = fourier_embedder
        self.resolution = resolution
        self.chunk = chunk

    @torch.no_grad()
    def to_mesh(self, decode, z: torch.Tensor) -> trimesh.Trimesh | None:
        """decode: model.decode(z, query_pe)->(1,n,1);z: (1, L, latent_dim)。回傳 mesh,無表面則 None。"""
        r = self.resolution
        coords = torch.linspace(-1.0, 1.0, r)
        grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)

        sdf = []
        for i in range(0, grid.shape[0], self.chunk):
            pe = self.fourier_embedder(grid[i:i + self.chunk]).unsqueeze(0).to(z.device)  # embed 在 CPU 再移 device
            sdf.append(decode(z, pe).reshape(-1).cpu())
        return self._extract(torch.cat(sdf).reshape(r, r, r).numpy(), r)

    @staticmethod
    def _extract(sdf: np.ndarray, resolution: int) -> trimesh.Trimesh | None:
        if not (sdf.min() < 0.0 < sdf.max()):
            return None  # 沒有零穿越 = grid 內抓不到表面(全在內或全在外)
        verts, faces, _, _ = marching_cubes(sdf, level=0.0)
        verts = verts / (resolution - 1) * 2.0 - 1.0  # voxel index -> [-1, 1]
        return trimesh.Trimesh(vertices=verts, faces=faces)

    def __call__(self, decode, z: torch.Tensor, out_path: str) -> str | None:
        mesh = self.to_mesh(decode, z)
        if mesh is None:
            return None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mesh.export(out_path)
        return out_path


if __name__ == "__main__":
    # 自檢:餵解析球面 SDF(半徑 0.5,內<0),marching cubes 出來的頂點應落在半徑 0.5 上。
    r = 64
    c = np.linspace(-1.0, 1.0, r)
    x, y, zc = np.meshgrid(c, c, c, indexing="ij")
    sphere = np.sqrt(x**2 + y**2 + zc**2) - 0.5
    m = Postprocess._extract(sphere, r)
    assert m is not None
    radii = np.linalg.norm(m.vertices, axis=1)
    assert abs(radii.mean() - 0.5) < 0.03, radii.mean()          # 尺度/rescale 對
    assert Postprocess._extract(np.ones((r, r, r)), r) is None    # 無零穿越 -> None
    print(f"ok  {len(m.vertices)} verts  mean radius {radii.mean():.3f}")
