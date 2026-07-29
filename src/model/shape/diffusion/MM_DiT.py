import torch
import torch.nn as nn
import math
from src.model.general.attention import SingleStreamMultiHeadSelfAttention, DoubleStreamMultiHeadSelfAttention, AdaLN

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
        ):
        super().__init__()
        
        self.time_factor = time_factor

        self.time_proj = nn.Sequential(
            nn.Linear(time_in_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
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
        
    
    def forward(self, t, cond, x):
        t_emb = self.time_proj(timestep_embedding(t, 256, max_period=self.time_factor).to(dtype=x.dtype))
        cond_proj = self.image_proj(cond)
        x_proj = self.latent_proj(x)

        for block in self.double_stream:
            x_proj, cond_proj = block(t_emb, x_proj, cond_proj) # (B, L, W)

        x = torch.cat([x_proj, cond_proj], dim=-2) # (B, 2L, W)

        for block in self.single_stream:
            x = block(t_emb, x) # (B, 2L, W)

        scale, shift = self.adaln(t_emb)
        x = x[:, :x_proj.shape[-2], :] # (B, L, W)
        x = (1 + scale) * self.ln(x) + shift # (B, L, W)
        x = self.proj(x)

        return x 