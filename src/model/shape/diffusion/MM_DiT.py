import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from src.model.general.attention import SingleStreamMultiHeadSelfAttention, DoubleStreamMultiHeadSelfAttention, AdaLN
from src.model.general.rope import _rope

# From Hunyuan's implementation, which is based on Open AI guided-diffusion implementation
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    freqs = freqs.to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MM_DiT(nn.Module):
    def __init__(
            self,
            double_stream_layers: int,
            single_stream_layers: int,    
            width: int,
            num_head: int,
            mlp_expansion: int,
            drop_prob: float,
            in_channel: int,
            out_channel: int,
            time_in_dim: int,
            cond_in_dim: int,
            latent_in_dim: int,
            time_factor: float,
            use_anchor: bool,
            rope_theta: int,
            rope_axes_dim: tuple,
            rope_coord_scale: float,
            use_checkpoint: bool,
        ):
        super().__init__()

        self.time_factor = time_factor
        self.head_dim = width // num_head

        self.use_anchor = use_anchor
        self.rope_theta = rope_theta
        self.rope_axes_dim = rope_axes_dim
        self.rope_coord_scale = rope_coord_scale
        self.use_checkpoint = use_checkpoint

        self.double_stream_layers = double_stream_layers

        self.time_proj = nn.Sequential(
            nn.Linear(time_in_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.anchor_gate = nn.Sequential(
            nn.Linear(time_in_dim, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.image_proj = nn.Linear(cond_in_dim, width)
        self.latent_proj = nn.Linear(latent_in_dim, width)

        self.double_stream = nn.ModuleList([
            *[DoubleStreamMultiHeadSelfAttention(
                width,
                num_head,
                mlp_expansion,
                drop_prob,
            ) for _ in range(double_stream_layers)]
        ])

        self.single_stream = nn.ModuleList([
            *[SingleStreamMultiHeadSelfAttention(
                width,
                num_head,
                mlp_expansion,
                drop_prob,
            ) for _ in range(single_stream_layers)]
        ])

        self.ln = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.proj = nn.Linear(width, out_channel)
        self.adaln = AdaLN(dim = width, num_mod = 1, gate = False)
    
    def forward(self, t, cond, x, anchor=None, probe_layer=None):
        probe_feature = None
        original_time_emb= timestep_embedding(t, 256, max_period=self.time_factor).to(dtype=x.dtype)
        t_emb = self.time_proj(original_time_emb)

        cond_proj = self.image_proj(cond)
        x_proj = self.latent_proj(x)

        pe_lat = None
        anchor_gate = None
        if self.use_anchor and anchor is not None:
            anchor_gate = torch.sigmoid(self.anchor_gate(original_time_emb)).unsqueeze(-1)
            pe_lat = _rope(anchor, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, anchor_gate)  # (B, L, D/2)

        for i, block in enumerate(self.double_stream):
            if self.use_checkpoint and self.training:
                x_proj, cond_proj = checkpoint(block, x_proj, cond_proj, y=t_emb, pe=pe_lat, use_reentrant=False)
            else:
                x_proj, cond_proj = block(x_proj, cond_proj, y=t_emb, pe=pe_lat) # (B, L, W)
            
            if i == probe_layer:
                probe_feature = x_proj # (B, L, W)

        x = torch.cat([x_proj, cond_proj], dim=-2) # (B, 2L, W)
        
        pe_full = None
        if pe_lat is not None:
            B, Lc, _ = cond_proj.shape
            pe_cond = _rope(x.new_zeros(B, Lc, 3), self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, anchor_gate)
            pe_full = torch.cat([pe_lat, pe_cond], dim=1) 

        for i, block in enumerate(self.single_stream):
            if self.use_checkpoint and self.training:
                x = checkpoint(block, t_emb, x, pe_full, use_reentrant=False)
            else:
                x = block(t_emb, x, pe=pe_full) # (B, 2L, W)
            
            if i + self.double_stream_layers == probe_layer:
                probe_feature = x[:, :x_proj.shape[-2], :] # (B, L, W)

        scale, shift = self.adaln(t_emb)
        x = x[:, :x_proj.shape[-2], :] # (B, L, W)
        x = (1 + scale) * self.ln(x) + shift # (B, L, W)
        x = self.proj(x)

        return x, probe_feature

    