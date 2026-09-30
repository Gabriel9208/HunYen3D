from torch import nn
import torch.nn.functional as F

from src.model.general.attention import ResidualMultiHeadSelfAttention, ResidualMultiHeadCrossAttention

class CrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        pe_dim: int,
        latent_dim: int,
        width: int,
        num_head: int,
        num_layers: int,
    ):
        super().__init__()

        self.proj_pe = nn.Linear(pe_dim, width)
        self.proj_latent = nn.Linear(latent_dim, width) 
         
        self.self_attention = nn.ModuleList(
            [
                ResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents) 
                for _ in range(num_layers)
            ]
        )

        self.cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )

        self.proj_sdf = nn.Linear(width, 1)

    def encode_latent(self, latent):
        """The half of the decoder that depends ONLY on the latent: projection + the self-attention
        stack. Split out so a caller decoding many query chunks against one latent (marching cubes
        does 128 chunks for a 128^3 grid) can run it once instead of per chunk -- measured at 57%
        of the total decode time at chunk=16384. forward() below keeps the original behaviour."""
        latent = self.proj_latent(latent)
        for layer in self.self_attention:
            latent = layer(latent)
        return latent

    def query_sdf(self, query, latent_processed):
        """The half that depends on the query points, given an already-processed latent."""
        x = self.cross_attention(self.proj_pe(query), latent_processed)  # (B, q_len, width)
        return self.proj_sdf(x)                                          # (B, q_len, 1)

    def forward(self, query, latent):
        return self.query_sdf(query, self.encode_latent(latent))

class AnchorDecoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        pe_dim: int,
        latent_dim: int,
        width: int,
        num_head: int,
        num_layers: int,
    ):
        super().__init__()

        self.proj_pe = nn.Linear(pe_dim, width)
        self.proj_anchor = nn.Linear(pe_dim, width)
        self.proj_latent = nn.Linear(latent_dim, width) 
         
        self.self_attention = nn.ModuleList(
            [
                ResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents) 
                for _ in range(num_layers)
            ]
        )

        self.anchor_cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )

        self.query_points_cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )

        self.proj_sdf = nn.Linear(width, 1)

    def forward(self, query, latent, anchors):
        assert latent.shape[1] == anchors.shape[1], "Latent and anchors must have the same number of tokens"
        
        latent = self.proj_latent(latent)
        query = self.proj_pe(query)
        anchor = self.proj_anchor(anchors)

        for layer in self.self_attention:
            latent = layer(latent)
 
        latent = self.anchor_cross_attention(anchor, latent) # (B, a_len, width)
        x = self.query_points_cross_attention(query, latent) # (B, q_len, width)

        sdf = self.proj_sdf(x) # (B, q_len, 1)
        return sdf

 
class MRLDecoder(CrossAttentionDecoder):
    def forward(self, query, latent, m):
        latent = latent[..., :m] # select only m channels
        latent = F.linear(latent, self.proj_latent.weight[:, :m], self.proj_latent.bias)
        query = self.proj_pe(query)

        for layer in self.self_attention:
            latent = layer(latent)

        x = self.cross_attention(query, latent) # (B, q_len, width)

        sdf = self.proj_sdf(x) # (B, q_len, 1)
        return sdf