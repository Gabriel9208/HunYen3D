import torch
from torch import nn

from src.model.general.attention import ResidualMultiHeadSelfAttention, ResidualMultiHeadCrossAttention, DoubleStreamMultiHeadSelfAttention, MaskedResidualMultiHeadSelfAttention, MaskedResidualMultiHeadCrossAttention
from src.model.general.rope import _rope, apply_rope

class CrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        pe_dim: int,
        latent_dim: int,
        width: int,
        num_head: int,
        num_layers: int,
        rope_theta: int = 10000,
        rope_axes_dim: tuple = (),
        rope_coord_scale: float = 1.0,
        enable_rope: bool = False,
        enable_self_atten_rope: bool = False,
    ):
        super().__init__()

        self.enable_rope = enable_rope
        self.enable_self_atten_rope = enable_self_atten_rope
        if enable_rope:
            self.rope_theta = rope_theta
            self.rope_axes_dim = rope_axes_dim
            self.rope_coord_scale = rope_coord_scale

        self.proj = nn.Linear(pe_dim, width) # position emb to transformer width
    
        self.cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )
         
        self.self_attention = nn.ModuleList(
            [ResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents) for _ in range(num_layers)]
        )

        self.ln = nn.LayerNorm(width, eps=1e-6)
        self.fc = nn.Linear(width, latent_dim * 2) # encode to mu and logvar

    def forward(self, query, data):
        # hold the raw coordinates: self.proj below rebinds `query`, after which query[..., :3] is
        # three arbitrary learned features, not xyz (include_input=True puts xyz in channels 0:3).
        q_xyz = query[..., :3]

        q_pe_latent=None
        d_pe_latent=None
        if self.enable_rope:
            q_pe_latent = _rope(q_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)
            d_pe_latent = _rope(data[..., :3], self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)

        query = self.proj(query)
        data = self.proj(data)
            
        x = self.cross_attention(query, data, q_pe_latent, d_pe_latent)

        q_pe_latent=None
        if self.enable_self_atten_rope:
            q_pe_latent = _rope(q_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)

        for layer in self.self_attention:
            x = layer(x, q_pe_latent)

        x = self.ln(x)
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=-1)

        return mu, logvar


class DoubleStreamEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        pe_dim: int,
        latent_dim: int,
        width: int,
        num_head: int,
        num_layers: int,
        mlp_expansion:int, 
        drop_prob:float,
        rope_theta: int,
        rope_axes_dim: tuple,
        rope_coord_scale: float,
    ):
        super().__init__()

        self.rope_theta = rope_theta
        self.rope_axes_dim = rope_axes_dim
        self.rope_coord_scale = rope_coord_scale

        self.uni_proj = nn.Linear(pe_dim, width) 
        self.important_proj = nn.Linear(pe_dim, width)
        
        self.uni_cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )

        self.important_cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )
         
        self.double_strean_attention = nn.ModuleList(
            [DoubleStreamMultiHeadSelfAttention(
                width=width,
                num_head=num_head,
                mlp_expansion=mlp_expansion,
                drop_prob=drop_prob,
                enable_mod=False,
            ) for _ in range(num_layers)]
        )

        self.ln = nn.LayerNorm(width, eps=1e-6)
        self.fc = nn.Linear(width, latent_dim * 2) # encode to mu and logvar

    def forward(self, q_xyz, d_xyz, q_emb, d_emb, return_hidden: bool = False):
        q_uni_xyz, q_important_xyz = q_xyz.chunk(2, dim=1)
        q_uni_emb, q_important_emb = q_emb.chunk(2, dim=1)

        d_uni_xyz, d_important_xyz = d_xyz.chunk(2, dim=1)
        d_uni_emb, d_important_emb = d_emb.chunk(2, dim=1)

        q_uni_lat = _rope(q_uni_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)
        q_important_lat = _rope(q_important_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)
        d_uni_lat = _rope(d_uni_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)
        d_important_lat = _rope(d_important_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)

        q_uni = self.uni_proj(q_uni_emb)
        q_important = self.important_proj(q_important_emb) 
        d_uni = self.uni_proj(d_uni_emb) 
        d_important = self.important_proj(d_important_emb) 

        uni_x = self.uni_cross_attention(q_uni, d_uni, q_pe_latent=q_uni_lat, d_pe_latent=d_uni_lat) # x
        important_x = self.important_cross_attention(q_important, d_important, q_pe_latent=q_important_lat, d_pe_latent=d_important_lat) # x
 
        for layer in self.double_strean_attention:
            uni_x, important_x = layer(uni_x, important_x)

        pre_ln_uni, pre_ln_imp = uni_x, important_x
        x = self.ln(torch.cat([uni_x, important_x], dim=-2))   # LN is per-token -> splitting after
        post_ln_uni, post_ln_imp = x[:, :uni_x.shape[1], :], x[:, uni_x.shape[1]:, :]  # equals LN-ing each half separately
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=-1)

        if return_hidden:
            return mu, logvar, {"pre_ln_uni": pre_ln_uni, "pre_ln_imp": pre_ln_imp,
                                 "post_ln_uni": post_ln_uni, "post_ln_imp": post_ln_imp}
        return mu, logvar

class FrameEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        pe_dim: int,
        latent_dim: int,
        width: int,
        num_head: int,
        num_layers: int,
        mlp_expansion:int, 
        drop_prob:float,
        rope_theta: int,
        rope_axes_dim: tuple,
        rope_coord_scale: float,
    ):
        super().__init__()

        self.rope_theta = rope_theta
        self.rope_axes_dim = rope_axes_dim
        self.rope_coord_scale = rope_coord_scale

        self.uni_proj = nn.Linear(pe_dim, width) 
        self.important_proj = nn.Linear(pe_dim, width)
        
        self.uni_cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )

        self.important_cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )
         
        self.double_strean_attention = nn.ModuleList(
            [ResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents) for _ in range(num_layers)]
        )

        self.ln = nn.LayerNorm(width, eps=1e-6)
        self.fc = nn.Linear(width, latent_dim * 2) # encode to mu and logvar

    def forward(self, q_xyz, d_xyz, q_emb, d_emb, return_hidden: bool = False):
        # q_xyz/q_emb are already 100% importance-branch query points (preprocess.py's
        # frame_sample() never computes a uniform-branch query) -- no chunk needed here anymore.
        d_uni_xyz, d_important_xyz = d_xyz.chunk(2, dim=1)
        d_uni_emb, d_important_emb = d_emb.chunk(2, dim=1)

        q_lat = _rope(q_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)
        d_uni_lat = _rope(d_uni_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)
        d_important_lat = _rope(d_important_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)

        q_uni = self.uni_proj(q_emb)
        q_important = self.important_proj(q_emb)
        d_uni = self.uni_proj(d_uni_emb)
        d_important = self.important_proj(d_important_emb)

        uni_x = self.uni_cross_attention(q_uni, d_uni, q_pe_latent=q_lat, d_pe_latent=d_uni_lat) # x
        important_x = self.important_cross_attention(q_important, d_important, q_pe_latent=q_lat, d_pe_latent=d_important_lat) # x

        cross_attn_uni, cross_attn_important = uni_x, important_x  # raw per-branch cross-attention output, before the sum below discards which branch a value came from
        x = uni_x + important_x
        for layer in self.double_strean_attention:
            x = layer(x, None)

        x = self.ln(x)
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=-1)

        if return_hidden:
            return mu, logvar, {"cross_attn_uni": cross_attn_uni, "cross_attn_important": cross_attn_important}

        return mu, logvar

class MaskedCrossAttentionEncoder(CrossAttentionEncoder):
    def __init__(
        self,
        num_latents: int,
        pe_dim: int,
        latent_dim: int,
        width: int,
        num_head: int,
        num_layers: int,
        rope_theta: int = 10000,
        rope_axes_dim: tuple = (),
        rope_coord_scale: float = 1.0,
        enable_rope: bool = False,
        enable_self_atten_rope: bool = False,
    ):
        super().__init__(
            num_latents=num_latents,
            pe_dim=pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_layers,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            rope_coord_scale=rope_coord_scale,
            enable_rope=enable_rope,
            enable_self_atten_rope=enable_self_atten_rope
        )
        #assume num layers are even
        assert num_layers % 2 == 0, "num_layers must be even"
        
        del self.self_attention

        self.cross_attention = MaskedResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )
         
        self.self_attention_part1 = nn.ModuleList(
            [MaskedResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents) for _ in range((num_layers - 2) // 2)]
        )        

        self.mix1 = ResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents)
        
        self.self_attention_part2 = nn.ModuleList(
            [MaskedResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents) for _ in range((num_layers - 2) // 2)]
        )

        self.mix2 = ResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents)

    def forward(self, query, data):
        q_xyz = query[..., :3]

        q_pe_latent=None
        d_pe_latent=None
        if self.enable_rope:
            q_pe_latent = _rope(q_xyz, self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)
            d_pe_latent = _rope(data[..., :3], self.rope_axes_dim, self.rope_theta, self.rope_coord_scale, None)

        query = self.proj(query)
        data = self.proj(data)
            
        x = self.cross_attention(query, data, q_pe_latent, d_pe_latent)

        for layer in self.self_attention_part1:
            x = layer(x, None)

        x = self.mix1(x, None)
        for layer in self.self_attention_part2:
            x = layer(x, None)

        x = self.mix2(x, None)

        x = self.ln(x)
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=-1)

        return mu, logvar