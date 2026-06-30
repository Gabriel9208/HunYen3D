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
    """Mesh 取樣的 primitives(heavy 端的零件)。負責把一個 mesh 變成原始幾何取樣:
    表面點雲 + 法向量、近表面/均勻 SDF 監督點。不做 Fourier、不負責 IO。
    `Preprocessor` 會組合這些 primitive 成 build_cache(heavy)/sample(light)。
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
        fps_count: int
    ):
        n = surface.shape[0]
        fps_count = min(fps_count, n)
        surface_pc = surface[:, :3]

        min_dist = torch.full((n,), torch.inf, device=surface.device)
        selected = torch.empty(fps_count, dtype=torch.long, device=surface.device)

        current = 0
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
        ratio: float = 0.5
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
            surface_point_method='sample',  # 純 trimesh 取樣,不需 OpenGL(headless 可用)
            sign_method='normal',
        ).astype(np.float32)

        return torch.from_numpy(query_points), torch.from_numpy(gt_sdf).unsqueeze(-1)


class Preprocessor(nn.Module):
    """mesh -> 模型輸入 tensor 的「單一真相」,支援 heavy / light 兩條路:

    - build_cache(mesh_path): heavy。產出「大池」(surface/sharp 點雲含法向量)與 SDF bank
      (query_points + gt_sdf),都是 pre-fourier 的原始幾何。離線跑一次、存檔。
    - sample(cache): light。從大池子採樣 + downsample/FPS、從 SDF bank 抽子集,最後做 Fourier。
      每個 epoch 呼叫,便宜(~1s),且保留子採樣隨機性。
    - forward(mesh_path) = sample(build_cache(...)): heavy 模式(即時、慢;debug 或無快取時的後備)。

    Fourier 只作用在 xyz(3 維),法向量原樣 concat:
      encoder pe_dim = self.encoder_pe_dim = fourier(3) + (in_channels - 3)
      decoder pe_dim = self.decoder_pe_dim = fourier(3)
    """

    def __init__(
        self,
        pe_freqs: int = 6,
        in_channels: int = 6,          # 每個表面點通道數:xyz(3) + normal(3)
        include_input: bool = True,
        include_pi: bool = True,       # Fourier 頻率是否乘上 π(官方 vae-v2-1 為 false)
        random_sample_count: int = 4096,
        important_sample_count: int = 2048,
        down_sample_count: int = 4096,
        num_surface_samples: int = 249856,
        n_query_surface: int = 200_000,
        n_query_uniform: int = 50_000,
        sdf_subset: int | None = None,  # 每次 sample 監督的 SDF 點數(None = 用整個 bank)
    ):
        super().__init__()

        self.pe_freqs = pe_freqs
        self.in_channels = in_channels
        self.include_input = include_input
        self.include_pi = include_pi
        self.random_sample_count = random_sample_count
        self.important_sample_count = important_sample_count
        self.down_sample_count = down_sample_count
        self.num_surface_samples = num_surface_samples
        self.n_query_surface = n_query_surface
        self.n_query_uniform = n_query_uniform
        self.sdf_subset = sdf_subset

        self.mesh2query = Mesh2Query()
        # 只對 xyz(3 維)做 Fourier;法向量不過 Fourier。
        self.fourier_embedder = FourierEmbedder(
            num_freqs=pe_freqs,
            input_dim=3,
            include_input=include_input,
            include_pi=include_pi,
        )

    # ---- 維度(供對照 model 的 pe_dim;configs 的 encoder/decoder_pe_dim 需與此一致)----
    @property
    def decoder_pe_dim(self) -> int:        # SDF 查詢點:只有 xyz 過 Fourier
        return self.fourier_embedder.out_dim

    @property
    def encoder_pe_dim(self) -> int:        # 表面點:Fourier(xyz) + 法向量
        return self.fourier_embedder.out_dim + (self.in_channels - 3)

    @property
    def cache_signature(self) -> dict:
        # 只含影響「大池 / SDF bank」內容的 heavy 參數;light 參數(down/random/important/pe/sdf_subset)
        # 改變不需要重建快取。
        return {
            "num_surface_samples": self.num_surface_samples,
            "n_query_surface": self.n_query_surface,
            "n_query_uniform": self.n_query_uniform,
        }

    # ---- heavy:離線一次 ----
    def build_cache(self, mesh_path: str) -> dict:
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        mesh = trimesh.load_mesh(mesh_path, force="mesh")
        mesh = self.mesh2query.normalize_mesh(mesh, 0.98)

        query_points, gt_sdf = self.mesh2query.sample_query_points(
            mesh, self.n_query_surface, self.n_query_uniform
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

    # ---- light:每個 epoch ----
    def sample(self, cache: dict):
        surf = self.mesh2query.downsample(cache["surface_pool"], self.down_sample_count)
        sharp = self.mesh2query.downsample(cache["sharp_pool"], self.down_sample_count)
        surf_fps = self.mesh2query.fps(surf, self.random_sample_count)
        sharp_fps = self.mesh2query.fps(sharp, self.important_sample_count)

        query = torch.cat([surf_fps, sharp_fps], dim=0)   # (random + important, 7)
        data = torch.cat([surf, sharp], dim=0)            # (2 * down_sample_count, 7)

        qp = cache["sdf_query_points"]
        gt = cache["gt_sdf"]
        if self.sdf_subset is not None and self.sdf_subset < qp.shape[0]:
            idx = torch.randperm(qp.shape[0])[: self.sdf_subset]
            qp, gt = qp[idx], gt[idx]

        q = self._embed_surface(query)
        d = self._embed_surface(data)
        sdf_query_points = self.fourier_embedder(qp)   # 只有 xyz
        return q, d, sdf_query_points, gt

    def _embed_surface(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (L, in_channels) = [xyz(3) | normal(in_channels-3)];Fourier 只作用 xyz。
        xyz, normal = pts.split([3, pts.shape[-1] - 3], dim=-1)
        return torch.cat([self.fourier_embedder(xyz), normal], dim=-1)

    # ---- heavy 模式(即時)----
    def forward(self, mesh_path: str):
        return self.sample(self.build_cache(mesh_path))
