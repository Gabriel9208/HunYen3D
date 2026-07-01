from torch import nn

from src.model.general.attention import ResidualMultiHeadSelfAttention, ResidualMultiHeadCrossAttention

class CrossAttentionEncoder(nn.Module):
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
        query = self.proj(query)
        data = self.proj(data)

        x = self.cross_attention(query, data)

        for layer in self.self_attention:
            x = layer(x)

        x = self.ln(x)
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=-1)

        return mu, logvar

        