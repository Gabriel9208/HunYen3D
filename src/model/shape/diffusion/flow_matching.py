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
        
    def transport(self, model, x1, cond):
        t, x0, x1 = self.sample(x1)
        tt = t[:, None, None]
        xt = (1 - tt) * x0 + tt * x1
        pred = model(t, cond, xt)
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
        
class Sampler:
    @staticmethod
    @torch.no_grad()
    def sample_ode(model, shape, cond, steps=50):
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device) 
        dt = 1 / steps

        for i in tqdm.tqdm(range(steps)):
            t = torch.full((shape[0],), i * dt, device=device)
            v = model(t, cond, x)
            x = x + v * dt

        return x
            

        