import os

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes


class Postprocess:
    def __init__(self, fourier_embedder, resolution: int = 128, chunk: int = 65536):
        self.fourier_embedder = fourier_embedder
        self.resolution = resolution
        self.chunk = chunk
        self._pe_cache = None      # (device, dtype) -> embedded grid, see to_mesh

    def _grid_pe(self, device, dtype):
        key = (str(device), dtype)
        if self._pe_cache is None or self._pe_cache[0] != key:
            r = self.resolution
            coords = torch.linspace(-1.0, 1.0, r)
            grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"),
                               dim=-1).reshape(-1, 3)
            pe = self.fourier_embedder(grid).to(device=device, dtype=dtype)
            self._pe_cache = (key, pe)
        return self._pe_cache[1]

    @torch.no_grad()
    def to_mesh(self, decode, z: torch.Tensor, decoder=None) -> trimesh.Trimesh | None:
        r = self.resolution
        pe_all = self._grid_pe(z.device, z.dtype)
        fast = decoder is not None and hasattr(decoder, "encode_latent")
        lat = decoder.encode_latent(z) if fast else None

        # Accumulate on the GPU: one transfer of r^3 floats at the end instead of one per chunk.
        sdf = torch.empty(pe_all.shape[0], device=z.device, dtype=z.dtype)
        for i in range(0, pe_all.shape[0], self.chunk):
            pe = pe_all[i:i + self.chunk].unsqueeze(0)
            out = decoder.query_sdf(pe, lat) if fast else decode(z, pe)[0]
            sdf[i:i + self.chunk] = out.reshape(-1)
        return self._extract(sdf.float().cpu().reshape(r, r, r).numpy(), r)

    @staticmethod
    def _extract(sdf: np.ndarray, resolution: int) -> trimesh.Trimesh | None:
        if not (sdf.min() < 0.0 < sdf.max()):
            return None  
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

