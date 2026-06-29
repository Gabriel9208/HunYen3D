from torch import nn
import torch.nn.functional as F

class ScaledDotProductAttention():
    def __call__(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value)

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
            nn.GELU(),
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
        self.head_dim = width // num_head
        self.num_latents = num_latents

        self.qkv_proj = nn.Linear(width, width * 3)
        self.q_norm = nn.LayerNorm(width // num_head, eps=1e-6)
        self.k_norm = nn.LayerNorm(width // num_head, eps=1e-6)
        self.attention = ScaledDotProductAttention()
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

        attn = self.attention(q, k, v)
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
        self.attention = ScaledDotProductAttention()
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

        attn = self.attention(q, k, v)
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
        