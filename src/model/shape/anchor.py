from torch import nn

from src.model.general.attention import ResidualMultiHeadSelfAttention

class Anchor(nn.Module):
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

        self.proj_latent = nn.Linear(latent_dim, width) 
         
        self.self_attention = nn.ModuleList(
            [
                ResidualMultiHeadSelfAttention(
                    width,
                    num_head,
                    num_latents=num_latents
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_anchor = nn.Linear(width, 3)
        
    def forward(self, latent):
        latent = self.proj_latent(latent)

        for layer in self.self_attention:
            latent = layer(latent)

        anchor = self.proj_anchor(latent) # (B, num_latents, 3)
        return anchor

        