import torch

def _rope(coords, axes_dim, theta_base, coord_scale, anchor_gate):
    if anchor_gate is None:
        anchor_gate = torch.ones_like(coords[..., 0:1])

    angles = []
    for a, dim_a in enumerate(axes_dim):
        omega_a = theta_base ** (-torch.arange(0, dim_a, 2, device=coords.device).float() / dim_a)
        angle_a = coords[..., a, None] * omega_a * coord_scale * anchor_gate
        angles.append(angle_a)

    angles = torch.cat(angles, dim=-1)
    return torch.polar(torch.ones_like(angles), angles) # (B, L, D/2)

def apply_rope(x, freqs_cis):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
    return x_out.type_as(x)