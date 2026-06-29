from torch import nn

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

        self.num_latents = num_latents
        self.pe_dim = pe_dim
        self.latent_dim = latent_dim
        self.width = width
        self.num_head = num_head
        self.num_layers = num_layers

        self.proj_pe = nn.Linear(pe_dim, width) 
        self.proj_latent = nn.Linear(latent_dim, width) 
         
        self.self_attention = nn.ModuleList(
            [ResidualMultiHeadSelfAttention(width, num_head, num_latents=num_latents) for _ in range(num_layers)]
        )

        self.cross_attention = ResidualMultiHeadCrossAttention(
            data_dim=width,
            width=width,
            num_head=num_head,
            num_latents=num_latents
        )

        self.proj_sdf = nn.Linear(width, 1)

    def forward(self, query, latent):
        latent = self.proj_latent(latent)
        query = self.proj_pe(query)

        for layer in self.self_attention:
            latent = layer(latent)

        x = self.cross_attention(query, latent) # (B, q_len, width)

        sdf = self.proj_sdf(x) # (B, q_len, 1)
        return sdf

        