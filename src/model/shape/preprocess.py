import gzip
import os

import numpy as np
import torch
import trimesh
from torch import nn
from mesh_to_sdf import mesh_to_sdf


# Follows the implementation of original code
class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class Mesh2Query(nn.Module):
    """Mesh sampling primitives (the heavy-side parts). Turns a mesh into raw geometry
    samples: surface point cloud + normals, near-surface / uniform SDF supervision points.
    Does no Fourier and no IO. `Preprocessor` composes these primitives into
    build_cache (heavy) / sample (light).
    """

    def __init__(self):
        super().__init__()

    # From original Hunyuan2.1 code
    def normalize_mesh(self, mesh, scale=0.98):
        bbox = mesh.bounds
        center = (bbox[1] + bbox[0]) / 2
        scale_ = (bbox[1] - bbox[0]).max()

        mesh.apply_translation(-center)
        mesh.apply_scale(1 / scale_ * 2 * scale)

        return mesh

    def random_sample(self, mesh, num_samples):
        points, face_idx = trimesh.sample.sample_surface(mesh, count=num_samples)
        normals = mesh.face_normals[face_idx]
        points = torch.from_numpy(points.astype(np.float32))
        normals = torch.from_numpy(normals.astype(np.float32))

        return points, normals

    # From original Hunyuan2.1 code
    def important_sample(self, mesh, num_samples):
        """
        Sample points and normals preferentially from sharp edges of the mesh.

        Sharp edges are detected based on the angle between vertex normals and face normals.
        Points are sampled along these edges proportionally to edge length.

        Args:
            mesh (trimesh.Trimesh): Input mesh to sample from.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - samples: Sampled points along sharp edges, shape (num, 3).
                - normals: Corresponding interpolated normals, shape (num, 3).
        """
        vertices = mesh.vertices # (V, 3)
        face_normals = mesh.face_normals # (F, 3)
        vertex_normals = mesh.vertex_normals # (V, 3)
        faces = mesh.faces # (F, 3)

        min_cos_angles = np.ones(vertices.shape[0])
        for i in range(3):
            dot = np.stack((min_cos_angles[faces[:, i]], np.sum(vertex_normals[faces[:, i]] * face_normals, axis=-1)), axis=-1)
            min_cos_angles[faces[:, i]] = np.min(dot, axis=-1)

        sharp_mask = min_cos_angles < 0.985
        # collect edge
        edge_a = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
        edge_b = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
        sharp_edge_mask = ((sharp_mask[edge_a] * sharp_mask[edge_b]))

        edge_a = edge_a[sharp_edge_mask]
        edge_b = edge_b[sharp_edge_mask]

        sharp_verts_a = vertices[edge_a]
        sharp_verts_b = vertices[edge_b]
        sharp_verts_an = vertex_normals[edge_a]
        sharp_verts_bn = vertex_normals[edge_b]

        weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
        weights /= np.sum(weights)

        random_number = np.random.rand(num_samples)
        w = np.random.rand(num_samples, 1)
        index = np.searchsorted(weights.cumsum(), random_number)
        samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
        normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]

        samples = torch.from_numpy(samples.astype(np.float32))
        normals = torch.from_numpy(normals.astype(np.float32))

        return samples, normals

    def downsample(
        self,
        surface: torch.Tensor,
        down_sample_count: int
    ):
        ind = torch.randperm(surface.shape[0])[:down_sample_count]
        return surface[ind]

    def fps(
        self,
        surface: torch.Tensor,
        fps_count: int,
        seeds: torch.Tensor | None = None,
    ):
        n = surface.shape[0]
        fps_count = min(fps_count, n)
        surface_pc = surface[:, :3]

        if seeds is not None:
            min_dist = torch.cdist(surface_pc, seeds[:, :3]).min(dim=1).values
            current = int(torch.argmax(min_dist))
        else:
            min_dist = torch.full((n,), torch.inf, device=surface.device)
            current = 0

        selected = torch.empty(fps_count, dtype=torch.long, device=surface.device)

        for i in range(fps_count):
            selected[i] = current
            dist = torch.norm(surface_pc - surface_pc[current], dim=1)
            min_dist = torch.minimum(min_dist, dist)
            current = int(torch.argmax(min_dist))

        return surface[selected]

    def sample_query_points(
        self,
        mesh: trimesh.Trimesh,
        n_surface: int = 200_000,
        n_uniform: int = 50_000,
        sigma1: float = 0.01,
        sigma2: float = 0.05,
        ratio: float = 0.5,
        sample_point_count: int = 2_000_000,  # points mesh_to_sdf scatters on the surface (larger = more accurate, slower)
    ):
        n_s1 = int(n_surface * ratio)
        n_s2 = n_surface - n_s1

        surf_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
        near1 = surf_pts[:n_s1] + np.random.randn(n_s1, 3) * sigma1
        near2 = surf_pts[n_s1:] + np.random.randn(n_s2, 3) * sigma2
        near_surface = np.concatenate([near1, near2], axis=0)

        uniform = np.random.uniform(-1, 1, size=(n_uniform, 3))

        query_points = np.concatenate([near_surface, uniform], axis=0)
        query_points = np.clip(query_points, -1.0, 1.0).astype(np.float32)

        gt_sdf = mesh_to_sdf(
            mesh, query_points,
            surface_point_method='sample',  # pure trimesh sampling, no OpenGL needed (works headless)
            sign_method='normal',
            sample_point_count=sample_point_count,
        ).astype(np.float32)

        return torch.from_numpy(query_points), torch.from_numpy(gt_sdf).unsqueeze(-1)


class Preprocessor(nn.Module):
    """The single source of truth for mesh -> model-input tensors, supporting heavy / light paths:

    - build_cache(mesh_path): heavy. Produces the "big pool" (surface/sharp point clouds with
      normals) and the SDF bank (query_points + gt_sdf), all pre-Fourier raw geometry. Run once
      offline and saved.
    - sample(cache): light. Subsamples from the big pool + downsample/FPS, draws a subset from the
      SDF bank, then applies Fourier. Called every epoch, cheap (~1s), and keeps the subsampling randomness.
    - forward(mesh_path) = sample(build_cache(...)): heavy mode (on the fly, slow; a fallback for debug or no cache).

    Fourier only acts on xyz (3 dims); normals are concatenated as-is:
      encoder pe_dim = self.encoder_pe_dim = fourier(3) + (in_channels - 3)
      decoder pe_dim = self.decoder_pe_dim = fourier(3)
    """

    def __init__(
        self,
        pe_freqs: int = 6,
        in_channels: int = 6,          # channels per surface point: xyz(3) + normal(3)
        include_input: bool = True,
        include_pi: bool = True,       # whether Fourier frequencies are multiplied by π (official vae-v2-1 uses false)
        random_sample_count: int = 4096,
        important_sample_count: int = 2048,
        downsample_ratio: int = 20,    # KV points per branch = downsample_ratio * that branch's query count (Hunyuan derive)
        num_surface_samples: int = 249856,
        n_query_surface: int = 200_000,
        n_query_uniform: int = 50_000,
        sdf_sample_point_count: int = 2_000_000,  # mesh_to_sdf surface sample count (heavy; affects accuracy and speed)
        sdf_subset: int | None = None,  # SDF points supervised per sample (None = use the whole bank)
    ):
        super().__init__()

        self.pe_freqs = pe_freqs
        self.in_channels = in_channels
        self.include_input = include_input
        self.include_pi = include_pi
        self.random_sample_count = random_sample_count
        self.important_sample_count = important_sample_count
        self.downsample_ratio = downsample_ratio
        self.num_surface_samples = num_surface_samples
        self.n_query_surface = n_query_surface
        self.n_query_uniform = n_query_uniform
        self.sdf_sample_point_count = sdf_sample_point_count
        self.sdf_subset = sdf_subset

        self.mesh2query = Mesh2Query()
        # Fourier only on xyz (3 dims); normals do not go through Fourier.
        self.fourier_embedder = FourierEmbedder(
            num_freqs=pe_freqs,
            input_dim=3,
            include_input=include_input,
            include_pi=include_pi,
        )

    # ---- dims (to match the model's pe_dim; configs' encoder/decoder_pe_dim must agree with these) ----
    @property
    def decoder_pe_dim(self) -> int:        # SDF query points: only xyz goes through Fourier
        return self.fourier_embedder.out_dim

    @property
    def encoder_pe_dim(self) -> int:        # surface points: Fourier(xyz) + normal
        return self.fourier_embedder.out_dim + (self.in_channels - 3)

    @property
    def cache_signature(self) -> dict:
        # Only the heavy params that affect the big pool / SDF bank contents; changing light params
        # (downsample_ratio/random/important/pe/sdf_subset) does not require rebuilding the cache.
        return {
            "num_surface_samples": self.num_surface_samples,
            "n_query_surface": self.n_query_surface,
            "n_query_uniform": self.n_query_uniform,
            "sdf_sample_point_count": self.sdf_sample_point_count,
        }

    # ---- heavy: offline once ----
    def build_cache(self, mesh_path: str) -> dict:
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        if mesh_path.endswith(".gz"):
            inner = os.path.splitext(mesh_path[:-3])[1].lstrip(".")  # e.g. "off"
            with gzip.open(mesh_path, "rb") as f:
                mesh = trimesh.load(f, file_type=inner, force="mesh")
        else:
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
        mesh = self.mesh2query.normalize_mesh(mesh, 0.98)

        query_points, gt_sdf = self.mesh2query.sample_query_points(
            mesh, self.n_query_surface, self.n_query_uniform,
            sample_point_count=self.sdf_sample_point_count,
        )
        random_pc, random_normal = self.mesh2query.random_sample(mesh, self.num_surface_samples)
        important_pc, important_normal = self.mesh2query.important_sample(mesh, self.num_surface_samples)

        return {
            "surface_pool": torch.cat([random_pc, random_normal, torch.zeros(random_pc.shape[0], 1)], dim=1),     # (N, 7)
            "sharp_pool": torch.cat([important_pc, important_normal, torch.ones(important_pc.shape[0], 1)], dim=1),  # (N, 7)
            "sdf_query_points": query_points,   # (Q, 3)
            "gt_sdf": gt_sdf,                   # (Q, 1)
            "signature": self.cache_signature,
        } 

    # ---- light: every epoch ----
    def sample(self, cache: dict):
        # KV (data) = downsample_ratio * query, per branch — mirrors Hunyuan's derive so the ratio stays
        # fixed instead of drifting with num_latents. ponytail: downsample() caps at pool size, so keep
        # downsample_ratio * max(random, important) <= num_surface_samples (else the ratio silently drops).
        surf = self.mesh2query.downsample(cache["surface_pool"], self.downsample_ratio * self.random_sample_count)
        sharp = self.mesh2query.downsample(cache["sharp_pool"], self.downsample_ratio * self.important_sample_count)
        surf_fps = self.mesh2query.fps(surf, self.random_sample_count)
        sharp_fps = self.mesh2query.fps(sharp, self.important_sample_count)

        query = torch.cat([surf_fps, sharp_fps], dim=0)   # (random + important = num_latents, 7)
        data = torch.cat([surf, sharp], dim=0)            # (downsample_ratio * (random + important), 7)

        qp = cache["sdf_query_points"]
        gt = cache["gt_sdf"]
        if self.sdf_subset is not None and self.sdf_subset < qp.shape[0]:
            idx = torch.randperm(qp.shape[0])[: self.sdf_subset]
            qp, gt = qp[idx], gt[idx]

        _, q_emb, q_normal = self._embed_surface(query)
        q = torch.cat([q_emb, q_normal], dim=-1)
        _, d_emb, d_normal = self._embed_surface(data)
        d = torch.cat([d_emb, d_normal], dim=-1)

        sdf_query_points = self.fourier_embedder(qp)   # xyz only
        query_xyz = query[:, :3]                       # raw FPS query coords (num_latents, 3): anchor GT
        return q, d, sdf_query_points, gt, query_xyz

    def sample_disjoint(self, cache: dict): # sharp -> uniform
        surf = self.mesh2query.downsample(cache["surface_pool"], self.downsample_ratio * self.random_sample_count)
        sharp = self.mesh2query.downsample(cache["sharp_pool"], self.downsample_ratio * self.important_sample_count)
        sharp_fps = self.mesh2query.fps(sharp, self.important_sample_count)
        surf_fps = self.mesh2query.fps(surf, self.random_sample_count, seeds=sharp_fps)

        query = torch.cat([surf_fps, sharp_fps], dim=0)   # (random + important = num_latents, 7)
        data = torch.cat([surf, sharp], dim=0)            # (downsample_ratio * (random + important), 7)

        qp = cache["sdf_query_points"]
        gt = cache["gt_sdf"]
        if self.sdf_subset is not None and self.sdf_subset < qp.shape[0]:
            idx = torch.randperm(qp.shape[0])[: self.sdf_subset]
            qp, gt = qp[idx], gt[idx]

        _, q_emb, q_normal = self._embed_surface(query)
        q = torch.cat([q_emb, q_normal], dim=-1)
        _, d_emb, d_normal = self._embed_surface(data)
        d = torch.cat([d_emb, d_normal], dim=-1)

        sdf_query_points = self.fourier_embedder(qp)
        query_xyz = query[:, :3]
        return q, d, sdf_query_points, gt, query_xyz

    def double_stream_sample(self, cache: dict):
        # KV (data) = downsample_ratio * query, per branch — mirrors Hunyuan's derive so the ratio stays
        # fixed instead of drifting with num_latents. ponytail: downsample() caps at pool size, so keep
        # downsample_ratio * max(random, important) <= num_surface_samples (else the ratio silently drops).
        surf = self.mesh2query.downsample(cache["surface_pool"], self.downsample_ratio * self.random_sample_count)
        sharp = self.mesh2query.downsample(cache["sharp_pool"], self.downsample_ratio * self.important_sample_count)
        surf_fps = self.mesh2query.fps(surf, self.random_sample_count)
        sharp_fps = self.mesh2query.fps(sharp, self.important_sample_count)

        query = torch.cat([surf_fps, sharp_fps], dim=0)   # (random + important = num_latents, 7)
        data = torch.cat([surf, sharp], dim=0)            # (downsample_ratio * (random + important), 7)

        qp = cache["sdf_query_points"]
        gt = cache["gt_sdf"]
        if self.sdf_subset is not None and self.sdf_subset < qp.shape[0]:
            idx = torch.randperm(qp.shape[0])[: self.sdf_subset]
            qp, gt = qp[idx], gt[idx]

        q_xyz, q_emb, q_rest = self._embed_surface(query)
        d_xyz, d_emb, d_rest = self._embed_surface(data)
        q_normal, q_sharp = q_rest.split([3, 1], dim=-1)
        d_normal, d_sharp = d_rest.split([3, 1], dim=-1)

        q_normal = torch.nn.functional.normalize(q_normal, dim=-1)
        d_normal = torch.nn.functional.normalize(d_normal, dim=-1)
        
        q_emb = torch.cat([q_emb, q_sharp], dim=-1)
        d_emb = torch.cat([d_emb, d_sharp], dim=-1)

        sdf_query_points = self.fourier_embedder(qp)   # xyz only

        return q_xyz, d_xyz, q_normal, d_normal, q_emb, d_emb, sdf_query_points, gt

    def frame_sample(self, cache: dict):
        # FrameEncoder's query is 100% importance points -- unlike double_stream_sample, the uniform
        # branch's query FPS is never computed here (FrameEncoder discards it, so computing it would
        # be a wasted FPS pass + embedding). The KV/data side is unchanged: both cross-attention
        # branches still read their own downsampled pool.
        surf = self.mesh2query.downsample(cache["surface_pool"], self.downsample_ratio * self.random_sample_count)
        sharp = self.mesh2query.downsample(cache["sharp_pool"], self.downsample_ratio * self.important_sample_count)
        sharp_fps = self.mesh2query.fps(sharp, self.important_sample_count)

        query = sharp_fps                                 # (important = num_latents, 7)
        data = torch.cat([surf, sharp], dim=0)             # (downsample_ratio * (random + important), 7)

        qp = cache["sdf_query_points"]
        gt = cache["gt_sdf"]
        if self.sdf_subset is not None and self.sdf_subset < qp.shape[0]:
            idx = torch.randperm(qp.shape[0])[: self.sdf_subset]
            qp, gt = qp[idx], gt[idx]

        q_xyz, q_emb, q_rest = self._embed_surface(query)
        d_xyz, d_emb, d_rest = self._embed_surface(data)
        q_normal, q_sharp = q_rest.split([3, 1], dim=-1)
        d_normal, d_sharp = d_rest.split([3, 1], dim=-1)

        q_normal = torch.nn.functional.normalize(q_normal, dim=-1)
        d_normal = torch.nn.functional.normalize(d_normal, dim=-1)

        q_emb = torch.cat([q_emb, q_sharp], dim=-1)
        d_emb = torch.cat([d_emb, d_sharp], dim=-1)

        sdf_query_points = self.fourier_embedder(qp)   # xyz only

        return q_xyz, d_xyz, q_normal, d_normal, q_emb, d_emb, sdf_query_points, gt

    def _embed_surface(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pts: (L, in_channels) = [xyz(3) | normal(in_channels-3)]; Fourier acts on xyz only.
        xyz, normal = pts.split([3, pts.shape[-1] - 3], dim=-1)
        return xyz, self.fourier_embedder(xyz), normal


    # ---- heavy mode (on the fly) ----
    def forward(self, mesh_path: str):
        return self.sample(self.build_cache(mesh_path))
