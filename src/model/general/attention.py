import torch
from torch import nn
import torch.nn.functional as F

"""
Drop path (Stochastic Depth) code from Hunyuan3D 2.1 (And it is also in timm)
"""
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'

class AdaLN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_mod: int = 2,
        gate: bool = True,
    ):  
        super().__init__()
        self.activate = nn.SiLU()
        self.proj = nn.Linear(dim, dim * 3 * num_mod) if gate else nn.Linear(dim, dim * 2 * num_mod)
        self.num_mod = num_mod
        self.gate = gate

        #init zero for adaLN 
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.activate(x)
        out = self.proj(x)[:, None, :]

        # gate_msa, scale_mse, shift_mse, gate_mlp, scale_mlp, shift_mlp 
        if self.gate:
            return out.chunk(3 * self.num_mod, dim=-1)
        
        # scale_mse, shift_mse
        return out.chunk(2 * self.num_mod, dim=-1)

class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        drop_prob: float = 0.
    ):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * expansion, dim),
            DropPath(drop_prob)
        )

    def forward(self, x):
        return self.ffn(x)
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_head: int,
        num_latents: int,
    ):
        super().__init__()
        assert width % num_head == 0

        self.width = width
        self.num_head = num_head
        self.num_latents = num_latents

        self.qkv_proj = nn.Linear(width, width * 3)
        self.q_norm = nn.LayerNorm(width // num_head, eps=1e-6)
        self.k_norm = nn.LayerNorm(width // num_head, eps=1e-6)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        assert  (length * width) % self.num_latents == 0

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, self.num_latents, self.num_head, -1)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, self.num_latents, width)
        attn = self.proj(attn)

        return attn
        
class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        data_dim: int,
        width: int,
        num_head: int,
        num_latents: int,
    ):
        super().__init__()
        assert width % num_head == 0

        self.num_head = num_head
        self.head_dim = width // num_head
        self.num_latents = num_latents
        
        self.q_proj = nn.Linear(width, width)
        self.kv_proj = nn.Linear(data_dim, width * 2)
        self.q_norm = nn.LayerNorm(width // num_head, eps=1e-6)
        self.k_norm = nn.LayerNorm(width // num_head, eps=1e-6)
        self.proj = nn.Linear(width, width)

    def forward(self, query, data):
        batch, q_len, width = query.shape
        _, data_len, data_dim = data.shape
        
        q = self.q_proj(query)
        kv = self.kv_proj(data)

        q = q.view(batch, q_len, self.num_head, -1)
        kv = kv.view(batch, data_len, self.num_head, -1)

        k, v = kv.chunk(2, dim=-1)
       

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(batch, q_len, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, data_len, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, data_len, self.num_head, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, q_len, width)
        attn = self.proj(attn)

        return attn

class ResidualMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_head: int,
        num_latents: int,
        mlp_expansion: int = 4,
        drop_prob: float = 0.
    ):
        super().__init__()

        self.ln_qkv = nn.LayerNorm(width, eps=1e-6)
        self.self_attention = MultiHeadSelfAttention(width, num_head, num_latents)

        self.ln_mlp = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLP(width, expansion=mlp_expansion, drop_prob=drop_prob)

    def forward(self, x):
        x = x + self.self_attention(self.ln_qkv(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

class ResidualMultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        data_dim: int,
        width: int,
        num_head: int,
        num_latents: int,
        mlp_expansion: int = 4,
        drop_prob: float = 0.
    ):
        super().__init__()

        self.ln_q = nn.LayerNorm(width, eps=1e-6)
        self.ln_data = nn.LayerNorm(data_dim, eps=1e-6)
        self.cross_attention = MultiHeadCrossAttention(data_dim, width, num_head, num_latents)

        self.ln_mlp = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLP(width, expansion=mlp_expansion, drop_prob=drop_prob)

    def forward(self, query, data):
        x = query + self.cross_attention(self.ln_q(query), self.ln_data(data))
        x = x + self.mlp(self.ln_mlp(x))
        return x
        
class SingleStreamMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_head: int,
        mlp_expansion: int,
        drop_prob: float = 0.
    ):
        super().__init__()
        assert width % num_head == 0

        self.width = width
        self.mlp_expansion = mlp_expansion
        self.num_head = num_head

        self.ln = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.adaln = AdaLN(width, num_mod=1)
        self.linear = nn.Linear(width, 3*width + mlp_expansion*width, bias=False)
        self.q_norm = nn.RMSNorm(width // num_head, eps=1e-6)
        self.k_norm = nn.RMSNorm(width // num_head, eps=1e-6)

        self.activate = nn.GELU(approximate="tanh")
        self.proj = nn.Linear(width + mlp_expansion*width, width)

    def forward(self, y, x):
        batch, length, width = x.shape

        gate, scale, shift = self.adaln(y)
        
        norm_x = self.ln(x)
        mod_x = (1 + scale) * norm_x + shift # (2, B, L, W)

        linear_out = self.linear(mod_x) # (2, B, L, 3W + mlp_expansion*W)
        qkv, mlp = torch.split(linear_out, [3*width, self.mlp_expansion*width], dim=-1)

        qkv = qkv.view(batch, length, self.num_head, -1).transpose(1, 2) # (B, H, L, 3W)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, H, L, W)

        q = self.q_norm(q) # (B, H, L, D)
        k = self.k_norm(k) # (B, H, L, D)

        attn = F.scaled_dot_product_attention(q, k, v) # (B, H, L, D)
        attn = attn.transpose(1, 2).contiguous().view(batch, length, -1) # (B, L, W)

        mlp = self.activate(mlp) # (B, L, mlp_expansion*W)

        combine = torch.cat((attn, mlp), dim=-1)
        proj_out = self.proj(combine)
        out = gate * proj_out + x

        return out
        

class DoubleStreamMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_head: int,
        mlp_expansion: int,
        drop_prob: float = 0.
    ):
        super().__init__()
        assert width % num_head == 0

        self.width = width
        self.num_head = num_head

        self.latent_norm1 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.latent_adaln = AdaLN(width, num_mod=2)
        self.latent_qkv_proj = nn.Linear(width, width * 3)
        self.latent_q_norm = nn.RMSNorm(width // num_head, eps=1e-6)
        self.latent_k_norm = nn.RMSNorm(width // num_head, eps=1e-6)

        self.cond_norm1 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.cond_adaln = AdaLN(width, num_mod=2)
        self.cond_qkv_proj = nn.Linear(width, width * 3)
        self.cond_q_norm = nn.RMSNorm(width // num_head, eps=1e-6)
        self.cond_k_norm = nn.RMSNorm(width // num_head, eps=1e-6)

        self.latent_proj = nn.Linear(width, width)
        self.cond_proj = nn.Linear(width, width)

        self.latent_norm2 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.cond_norm2 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)

        self.latent_mlp = MLP(width, expansion=mlp_expansion, drop_prob=drop_prob)
        self.cond_mlp = MLP(width, expansion=mlp_expansion, drop_prob=drop_prob)

    def forward(self, y, x, c):
        batch, length_x, width = x.shape
        batch, length_c, width = c.shape

        latent_gate_mse, latent_scale_mse, latent_shift_mse, latent_gate_mlp, latent_scale_mlp, latent_shift_mlp = self.latent_adaln(y)
        cond_gate_mse, cond_scale_mse, cond_shift_mse, cond_gate_mlp, cond_scale_mlp, cond_shift_mlp = self.cond_adaln(y)
        
        norm_x = self.latent_norm1(x)
        norm_c = self.cond_norm1(c)
        
        mod_x = (1 + latent_scale_mse) * norm_x + latent_shift_mse # (B, L, W)
        mod_c = (1 + cond_scale_mse) * norm_c + cond_shift_mse # (B, L, W)

        latent_qkv = self.latent_qkv_proj(mod_x) # (B, L, 3W)
        cond_qkv = self.cond_qkv_proj(mod_c) # (B, L, 3W)

        latent_qkv = latent_qkv.view(batch, length_x, self.num_head, -1).transpose(1, 2)# (B, H, L, D)
        cond_qkv = cond_qkv.view(batch, length_c, self.num_head, -1).transpose(1, 2)# (B, H, L, D)

        l_q, l_k, l_v = latent_qkv.chunk(3, dim=-1)
        c_q, c_k, c_v = cond_qkv.chunk(3, dim=-1)

        l_q, l_k = self.latent_q_norm(l_q), self.latent_k_norm(l_k)
        c_q, c_k = self.cond_q_norm(c_q), self.cond_k_norm(c_k)# (B, H, L, D)
                
        q = torch.cat([l_q, c_q], dim=-2).contiguous() # (B, H, L_x + L_c, D)
        k = torch.cat([l_k, c_k], dim=-2).contiguous() # (B, H, L_x + L_c, D)
        v = torch.cat([l_v, c_v], dim=-2).contiguous() # (B, H, L_x + L_c, D)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, length_x + length_c, -1)

        latent_atten, cond_atten = attn[:, :length_x, :], attn[:, length_x:, :]

        latent_proj_attn = self.latent_proj(latent_atten) # (B, L, W)
        cond_proj_attn = self.cond_proj(cond_atten)
        
        latent_residual = latent_gate_mse * latent_proj_attn + x
        cond_residual = cond_gate_mse * cond_proj_attn + c

        ln_latent = self.latent_norm2(latent_residual)
        ln_cond = self.cond_norm2(cond_residual)
        
        mod_latent = (1 + latent_scale_mlp) * ln_latent + latent_shift_mlp
        mod_cond = (1 + cond_scale_mlp) * ln_cond + cond_shift_mlp
        
        latent_mlp_out = self.latent_mlp(mod_latent)
        cond_mlp_out = self.cond_mlp(mod_cond)
        
        latent_out = latent_gate_mlp * latent_mlp_out + latent_residual
        cond_out = cond_gate_mlp * cond_mlp_out + cond_residual

        return latent_out, cond_out
