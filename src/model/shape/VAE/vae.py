import math
import torch
from torch import nn

from src.model.shape.VAE.encoder import CrossAttentionEncoder, DoubleStreamEncoder, FrameEncoder, MaskedCrossAttentionEncoder
from src.model.shape.VAE.decoder import CrossAttentionDecoder, AnchorDecoder, MRLDecoder
from src.model.shape.anchor import Anchor

class Gaussian():

    def __init__(self, mu, logvar):
        logvar = torch.clamp(logvar, -30, 20)

        self.mu = mu
        self.logvar = logvar

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, lam=None):
        # lam: lambda-VAE (arXiv:2607.05531) per-channel exponent -> z = mu + sigma^lam * eps (Eq.9).
        # kl_divergence() is deliberately left alone: charging the KL for the ORIGINAL sigma while
        # the decoder only sees the reduced noise is the entire mechanism (Eq.10).
        # The min is Eq.16's max(1, .) applied ELEMENTWISE and taken in LOG SPACE. Both details are
        # load-bearing -- each NaN'd a real run. See docs/RESEARCH_LOG.md -> "lambda-VAE
        # implementation: two NaN incidents".
        eps = torch.randn_like(self.std)
        if lam is None:
            return self.mu + eps * self.std
        log_std = torch.log(self.std.clamp_min(1e-12))
        return self.mu + eps * torch.exp(torch.minimum(log_std, lam * log_std))

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
        lam_delta: float = 0.0,
        lam_ramp_steps: int = 20000,
        lam_ema: float = 0.9,
        lam_max: float = 100.0,
        lam_update_every: int = 3000,
    ):
        super().__init__()

        self.lam_delta = lam_delta
        self.lam_ramp_steps = lam_ramp_steps
        self.lam_ema = lam_ema
        self.lam_max = lam_max
        self.lam_update_every = lam_update_every
        # persistent=False: these are derived schedule state, not weights. Keeping them out of
        # state_dict means every checkpoint trained before lambda-VAE existed still loads, and a
        # lambda run's own checkpoints stay loadable by the DiT task and the diagnostic scripts.
        # Cost: a resume reseeds the EMA from its first window and restarts the ramp.
        self.register_buffer("sigma_ema", torch.full((latent_dim,), float("nan")), persistent=False)
        self.register_buffer("sigma_acc", torch.zeros(latent_dim), persistent=False)
        self.register_buffer("acc_n", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("lam_star", torch.ones(latent_dim), persistent=False)
        self.register_buffer("lam_step", torch.zeros((), dtype=torch.long), persistent=False)

        self.encoder = CrossAttentionEncoder(
            num_latents=num_latents,
            pe_dim=encoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_encoder_layers,
            rope_theta=10000,
            rope_axes_dim=(),
            rope_coord_scale=1.0,
            enable_rope=False,
        )

        self.decoder = CrossAttentionDecoder(
            num_latents=num_latents,
            pe_dim=decoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_decoder_layers,
        )

    def _lam(self, std: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.training:
                self.sigma_acc += std.detach().mean(dim=tuple(range(std.dim() - 1)))
                self.acc_n += 1
                self.lam_step += 1
                if self.acc_n >= self.lam_update_every:
                    window = self.sigma_acc / self.acc_n
                    if torch.isnan(self.sigma_ema).any():
                        self.sigma_ema.copy_(window)
                    else:
                        self.sigma_ema.mul_(self.lam_ema).add_(window, alpha=1.0 - self.lam_ema)
                    logs = torch.log(self.sigma_ema.clamp_min(1e-12))
                    lam = torch.where(logs < -1e-3,
                                      math.log1p(-1.0 / self.lam_delta) / (2.0 * logs),
                                      torch.ones_like(logs))
                    self.lam_star.copy_(lam.clamp(min=1.0, max=self.lam_max))
                    self.sigma_acc.zero_(); self.acc_n.zero_()
            ramp = (self.lam_step.float() / max(self.lam_ramp_steps, 1)).clamp(max=1.0)
            return 1.0 + ramp * (self.lam_star - 1.0)   # paper's guard against premature compression

    def encode(self, query, data, sample_posterior: bool = True, return_mean=False):
        # sample_posterior=False → z=μ (deterministic, distribution.mode()); used by the pure-capacity
        # overfit so σ is never sampled → no σ-explosion even at kl=0. Mirrors Hunyuan's encode flag.
        mu, logvar = self.encoder(query, data)
        distribution = Gaussian(mu, logvar)
        if sample_posterior:
            z = distribution.sample(self._lam(distribution.std) if self.lam_delta > 1.0 else None)
        else:
            z = distribution.mode()
        kl = distribution.kl_divergence()

        if return_mean:
            return z, kl, distribution.mode()   # mu, for regularisers that must not see the noise

        return z, kl

    def decode(self, z, query_pe):
        reconstructed = self.decoder(query_pe, z) # (B, q_len, )
        return reconstructed, None

    def forward(self, z, query_pe):
        return self.decode(z, query_pe)

class MRLVAE(VAE):
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
        lam_delta: float = 0.0,
        lam_ramp_steps: int = 20000,
        lam_ema: float = 0.9,
        lam_max: float = 100.0,
        lam_update_every: int = 3000,
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
            lam_delta=lam_delta,
            lam_ramp_steps=lam_ramp_steps,
            lam_ema=lam_ema,
            lam_max=lam_max,
            lam_update_every=lam_update_every,
        )

        self.decoder = MRLDecoder(
            num_latents=num_latents,
            pe_dim=decoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_decoder_layers,
        )
    def encode(self, query, data, m, sample_posterior: bool = True):
        mu, logvar = self.encoder(query, data)
        lam = self._lam(Gaussian(mu, logvar).std) if self.lam_delta > 1.0 else None
        distribution = Gaussian(mu[..., :m], logvar[..., :m])
        if sample_posterior:
            z = distribution.sample(lam[:m] if lam is not None else None)
        else:
            z = distribution.mode()
        kl = distribution.kl_divergence()

        return z, kl
    
    def decode(self, z, query_pe, m):
        reconstructed = self.decoder(query_pe, z, m) # (B, q_len, )
        return reconstructed, None

    def forward(self, z, query_pe, m):
        return self.decode(z, query_pe, m)

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

class DoubleStreamVAE(VAE):
    def __init__(
        self,
        rope_theta: int,
        rope_axes_dim: tuple,
        rope_coord_scale: float,
        num_latents: int = 4096,
        encoder_pe_dim: int = 42,
        decoder_pe_dim: int = 39,
        latent_dim: int = 64,
        width: int = 1024,
        num_head: int = 16,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 16,
        mlp_expansion:int = 4, 
        drop_prob:float = 0.0,
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

        self.encoder = DoubleStreamEncoder(
            num_latents=num_latents,
            pe_dim=encoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_encoder_layers,
            mlp_expansion=mlp_expansion, 
            drop_prob=drop_prob,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            rope_coord_scale=rope_coord_scale,
        )

        self.decoder = CrossAttentionDecoder(
            num_latents=num_latents,
            pe_dim=decoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_decoder_layers,
        )

    def encode(self, q_xyz, d_xyz, q_emb, d_emb, sample_posterior: bool = True):
        mu, logvar = self.encoder(q_xyz, d_xyz, q_emb, d_emb)
        distribution = Gaussian(mu, logvar)
        z = distribution.sample() if sample_posterior else distribution.mode()
        kl = distribution.kl_divergence()

        return z, kl

class FrameVAE(VAE):
    def __init__(
        self,
        rope_theta: int,
        rope_axes_dim: tuple,
        rope_coord_scale: float,
        num_latents: int = 512,
        encoder_pe_dim: int = 42,
        decoder_pe_dim: int = 39,
        latent_dim: int = 64,
        width: int = 1024,
        num_head: int = 16,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 16,
        mlp_expansion: int = 4,
        drop_prob: float = 0.0,
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

        self.encoder = FrameEncoder(
            num_latents=num_latents,
            pe_dim=encoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_encoder_layers,
            mlp_expansion=mlp_expansion,
            drop_prob=drop_prob,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            rope_coord_scale=rope_coord_scale,
        )

        self.decoder = CrossAttentionDecoder(
            num_latents=num_latents,
            pe_dim=decoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_decoder_layers,
        )

    def encode(self, q_xyz, d_xyz, q_emb, d_emb, sample_posterior: bool = True):
        mu, logvar = self.encoder(q_xyz, d_xyz, q_emb, d_emb)
        distribution = Gaussian(mu, logvar)
        z = distribution.sample() if sample_posterior else distribution.mode()
        kl = distribution.kl_divergence()

        return z, kl

class RoPEVAE(VAE):
    def __init__(
        self,
        rope_theta: int,
        rope_axes_dim: tuple,
        rope_coord_scale: float,
        num_latents: int = 4096,
        encoder_pe_dim: int = 42,
        decoder_pe_dim: int = 39,
        latent_dim: int = 64,
        width: int = 1024,
        num_head: int = 16,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 16,
        enable_self_atten_rope: bool = False,
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

        self.encoder = CrossAttentionEncoder(
            num_latents=num_latents,
            pe_dim=encoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_encoder_layers,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            rope_coord_scale=rope_coord_scale,
            enable_rope=True,
            enable_self_atten_rope=enable_self_atten_rope,
        )
    def decode(self, z, query_pe, xyz_fourier_emb=None):
        reconstructed = self.decoder(query_pe, z, xyz_fourier_emb) # (B, q_len, )
        return reconstructed, None


class MaskedVAE(VAE):
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
        lam_delta: float = 0.0,
        lam_ramp_steps: int = 20000,
        lam_ema: float = 0.9,
        lam_max: float = 100.0,
        lam_update_every: int = 3000,
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
            lam_delta=lam_delta,
            lam_ramp_steps=lam_ramp_steps,
            lam_ema=lam_ema,
            lam_max=lam_max,
            lam_update_every=lam_update_every,
        )

        self.encoder = MaskedCrossAttentionEncoder(
            num_latents=num_latents,
            pe_dim=encoder_pe_dim,
            latent_dim=latent_dim,
            width=width,
            num_head=num_head,
            num_layers=num_encoder_layers,
            rope_theta=10000,
            rope_axes_dim=(),
            rope_coord_scale=1.0,
            enable_rope=False,
        )

