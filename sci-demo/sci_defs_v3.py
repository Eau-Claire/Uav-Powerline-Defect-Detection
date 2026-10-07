"""SCI v3 modules used by the audited YOLO11s checkpoint."""
from __future__ import annotations

import torch
from torch import nn

try:
    from ultralytics.nn.modules import Detect
except ImportError:  # Allows importing this file for lightweight inspection.
    Detect = nn.Module


class StructuralConsistencyV3(nn.Module):
    def __init__(self, channels: list[int] | tuple[int, ...], latent: int = 128):
        super().__init__()
        self.channels = list(channels)
        self.latent = latent
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, latent, 1, bias=False), nn.BatchNorm2d(latent), nn.SiLU())
            for c in self.channels
        )
        self.context = nn.Sequential(
            nn.Conv2d(latent * len(self.channels), latent, 1, bias=False),
            nn.GroupNorm(8, latent), nn.SiLU()
        )
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Conv2d(latent * 2, latent, 1), nn.SiLU(), nn.Conv2d(latent, latent, 1), nn.Sigmoid())
            for _ in self.channels
        )
        self.restore = nn.ModuleList(
            nn.Sequential(nn.Conv2d(latent, c, 1, bias=False), nn.BatchNorm2d(c))
            for c in self.channels
        )
        for block in self.restore:
            nn.init.zeros_(block[1].weight)
            nn.init.zeros_(block[1].bias)
        self.residual_scale = nn.Parameter(torch.full((len(self.channels),), 0.10))

    def forward(self, sources):
        if not isinstance(sources, (list, tuple)):
            return sources
        projected = [layer(src) for layer, src in zip(self.project, sources)]
        pooled = [x.mean((2, 3), keepdim=True) for x in projected]
        context = self.context(torch.cat(pooled, dim=1))
        outputs = []
        for i, (src, feature) in enumerate(zip(sources, projected)):
            gate = self.gates[i](torch.cat([feature, context.expand_as(feature)], dim=1))
            outputs.append(src + self.residual_scale[i] * self.restore[i](feature * gate))
        return outputs


class SCIDetectV3(Detect):
    """Ultralytics Detect head with SCI applied to its input feature maps."""
    def __init__(self, nc=80, ch=(), *args, **kwargs):
        super().__init__(nc, ch, *args, **kwargs)
        channels = list(ch) if isinstance(ch, (list, tuple)) else []
        self.sci = StructuralConsistencyV3(channels) if channels else nn.Identity()

    def forward(self, x):
        return super().forward(self.sci(x))


def register_sci() -> None:
    import ultralytics.nn.tasks as tasks
    tasks.Detect = SCIDetectV3

