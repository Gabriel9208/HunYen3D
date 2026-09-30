import torch
from torch import nn
import torch.nn.functional as F
import math
import tqdm
import tqdm

class FlowMatching:
    def __init__(self, std=1, mean=0, shift_scale=3):
        # SD3 Paper suggested that std=1, mean=0, shift_scale=3 have good performance
        super().__init__()
        self.std = std
        self.mean = mean
        self.sqrt_shift_scale = math.sqrt(shift_scale) 
        
    def transport(self, model, x1, cond, scaling_factor=1.0, anchor_head=None, self_cond=False):
        t, x0, x1 = self.sample(x1)
        tt = t[:, None, None]
        x1 = x1 * scaling_factor
        xt = (1 - tt) * x0 + tt * x1

        anchor = None
        if self_cond and anchor_head is not None and torch.rand(()) < 0.5:
            with torch.no_grad():
                v0, _ = model(t, cond, xt)                 
                z_star = xt + (1 - tt) * v0             
                anchor = anchor_head(z_star).detach()   

        pred, _ = model(t, cond, xt, anchor=anchor) 
        noise_pred_loss = F.mse_loss(pred, x1 - x0)
        return noise_pred_loss

    def sample(self, x1):
        x0 = torch.randn(x1.shape, device=x1.device)

        # logit-normal 
        t = torch.randn((x1.shape[0],)) * self.std + self.mean
        t = t.to(x1.device)
        t = 1 / (1 + torch.exp(-t)) # sigmoid
        t = self.sqrt_shift_scale * t / (1 + (self.sqrt_shift_scale - 1) * t)

        return t, x0, x1 

class SRA2FM(FlowMatching):
    def __init__(self, std=1, mean=0, shift_scale=3):
        super().__init__(std, mean, shift_scale)

    def transport(self, model, sra, x1, cond, probe_layer=0, scaling_factor=1.0, lambda_sra2=1, self_cond=False):
        t, x0, x1 = self.sample(x1)
        tt = t[:, None, None]
        x1 = x1 * scaling_factor
        xt = (1 - tt) * x0 + tt * x1

        pred, probe_feature = model(t, cond, xt, probe_layer=probe_layer)        
        noise_pred_loss = F.mse_loss(pred, x1 - x0) 
        sra_loss = self.sra2_loss(sra, probe_feature, x1)
        return noise_pred_loss + lambda_sra2 * sra_loss, sra_loss, noise_pred_loss

    def sra2_loss(self, sra, feature, x1):
        # feature: (B, L, W), x1: (B, L, D)
        f = sra(feature)
        loss = F.smooth_l1_loss(f, x1, beta=0.05)

        return loss

        
class Sampler:
    @staticmethod
    @torch.no_grad()
    def sample_ode(model, shape, cond, steps=50, guidance=1.0, scaling_factor=1.0, anchor_head=None):
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        dt = 1 / steps

        anchor = None  
        for i in tqdm.tqdm(range(steps)):
            t = torch.full((shape[0],), i * dt, device=device)
            v, _ = model(t, cond, x, anchor=anchor)
            if guidance != 1.0:
                v_uncond, _ = model(t, torch.zeros_like(cond), x, anchor=anchor)
                v = v_uncond + guidance * (v - v_uncond)
            if anchor_head is not None:
                z_star = x + (1 - t[:, None, None]) * v     
                anchor = anchor_head(z_star)
            x = x + v * dt
        
        return x / scaling_factor
            

        