import torch
from torch import nn

from src.model.shape.VAE.encoder import CrossAttentionEncoder
from src.model.shape.VAE.decoder import CrossAttentionDecoder, AnchorDecoder
from src.model.shape.anchor import Anchor

class Gaussian():

    def __init__(self, mu, logvar):
        logvar = torch.clamp(logvar, -30, 20)

        self.mu = mu
        self.logvar = logvar

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        eps = torch.randn_like(self.std)
        return self.mu + eps * self.std

    def mode(self):
        return self.mu
    
    def kl_divergence(self):
        return 0.5 * torch.mean(torch.pow(self.mu, 2) + self.var - 1.0 - self.logvar, dim=(-1, -2))

class VAE(nn.Module):
    def __init__(
        self,
        num_latents: int = 4096,
        encoder_pe_dim: int = 42,
        decoder_pe_dim: int = 39,
        latent_dim: int = 64,
        width: int = 1024,
        num_head: int = 16,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 16,
    ):
        super().__init__()

        self.encoder = CrossAttentionEncoder(
            num_latents=num_latents,
            pe_dim=encoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_encoder_layers,
        )

        self.decoder = CrossAttentionDecoder(
            num_latents=num_latents,
            pe_dim=decoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_decoder_layers,
        )

    def encode(self, query, data, sample_posterior: bool = True):
        # sample_posterior=False → z=μ (deterministic, distribution.mode()); used by the pure-capacity
        # overfit so σ is never sampled → no σ-explosion even at kl=0. Mirrors Hunyuan's encode flag.
        mu, logvar = self.encoder(query, data)
        distribution = Gaussian(mu, logvar)
        z = distribution.sample() if sample_posterior else distribution.mode()
        kl = distribution.kl_divergence()

        return z, kl

    def decode(self, z, query_pe):
        reconstructed = self.decoder(query_pe, z) # (B, q_len, )
        return reconstructed, None

    def forward(self, z, query_pe):
        return self.decode(z, query_pe)


class AnchorVAE(VAE):
    def __init__(
        self,
        position_encoder: nn.Module,
        num_latents: int = 4096,
        encoder_pe_dim: int = 42,
        decoder_pe_dim: int = 39,
        latent_dim: int = 64,
        width: int = 1024,
        num_head: int = 16,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 16,
        num_anchor_layers: int = 6,
        detach_anchor: bool = False,
    ):
        super().__init__(
            num_latents=num_latents,
            encoder_pe_dim=encoder_pe_dim,
            decoder_pe_dim=decoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        self.anchor = Anchor(
            num_latents=num_latents,
            pe_dim=decoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_anchor_layers,
        )

        self.pe = position_encoder
        self.detach_anchor = detach_anchor  # True → anchor branch gets no gradient into z/encoder

        self.decoder = AnchorDecoder(
            num_latents=num_latents,
            pe_dim=decoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_decoder_layers,
        )
        
    def decode(self, z, query_pe):
        z_anchor = z.detach() if self.detach_anchor else z  # cut anchor-branch gradient to z/encoder when set
        anchors = self.anchor(z_anchor)
        pe_anchors = self.pe(anchors)
        reconstructed = self.decoder(query_pe, z, pe_anchors)  # decoder still gets non-detached z (recon path intact)
        return reconstructed, anchors

        