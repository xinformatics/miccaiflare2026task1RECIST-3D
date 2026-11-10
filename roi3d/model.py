from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


DOMAIN_GROUPS = ("liver", "pancreas", "kidney", "lung", "colon", "other")


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear",
                              align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class RoutedResidualAdapter(nn.Module):
    """Small zero-initialized residual experts with per-sample hard routing."""

    def __init__(self, channels: int, rank: int, experts: int):
        super().__init__()
        self.down_weight = nn.Parameter(torch.empty(experts, rank, channels))
        self.up_weight = nn.Parameter(torch.zeros(experts, channels, rank))
        nn.init.kaiming_uniform_(self.down_weight, a=0.01)

    def forward(self, x: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        # Select per-sample 1x1-convolution weights. This performs only B expert
        # passes rather than B*E and compiles without data-dependent Python.
        down = self.down_weight[routing]
        up = self.up_weight[routing]
        hidden = F.leaky_relu(
            torch.einsum("brc,bczxy->brzxy", down, x), 0.01, inplace=True
        )
        residual = torch.einsum("bcr,brzxy->bczxy", up, hidden)
        return x + residual


class SoftRoutedResidualAdapter(nn.Module):
    """Low-rank experts mixed per lesion by a differentiable soft router.

    The residual branch starts at zero, so adding this module to a pretrained
    generalist is function preserving.  Different random down projections
    break expert symmetry while the router learns.
    """

    def __init__(self, channels: int, rank: int, experts: int):
        super().__init__()
        self.down_weight = nn.Parameter(torch.empty(experts, rank, channels))
        self.up_weight = nn.Parameter(torch.zeros(experts, channels, rank))
        nn.init.kaiming_uniform_(self.down_weight, a=0.01)

    def forward(self, x: torch.Tensor, routing: torch.Tensor,
                focus: torch.Tensor | None = None) -> torch.Tensor:
        residual = torch.zeros_like(x)
        # Looping over a small, static number of experts avoids materializing
        # B x E x C x Z x Y x X activations at the full decoder resolution.
        for expert in range(self.down_weight.shape[0]):
            hidden = F.leaky_relu(
                torch.einsum("rc,bczxy->brzxy", self.down_weight[expert], x),
                0.01, inplace=False,
            )
            update = torch.einsum(
                "cr,brzxy->bczxy", self.up_weight[expert], hidden
            )
            residual.add_(
                update * routing[:, expert, None, None, None, None]
            )
        if focus is not None:
            residual = residual * focus
        return x + residual


class RECISTUNet3D(nn.Module):
    """The 16.5M-parameter, base-32 five-level U-Net from REPRODUCE.md."""

    def __init__(self, in_channels: int = 4, base: int = 32,
                 deep_supervision: bool = True, organ_experts: bool = False,
                 expert_rank: int = 8):
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8, base * 10)
        self.enc0 = ConvBlock(in_channels, channels[0])
        self.enc1 = ConvBlock(channels[0], channels[1])
        self.enc2 = ConvBlock(channels[1], channels[2])
        self.enc3 = ConvBlock(channels[2], channels[3])
        self.bottleneck = ConvBlock(channels[3], channels[4])
        self.pool = nn.MaxPool3d(2)
        self.up3 = UpBlock(channels[4], channels[3], channels[3])
        self.up2 = UpBlock(channels[3], channels[2], channels[2])
        self.up1 = UpBlock(channels[2], channels[1], channels[1])
        self.up0 = UpBlock(channels[1], channels[0], channels[0])
        self.head = nn.Conv3d(channels[0], 1, 1)
        self.deep_supervision = deep_supervision
        self.ds1 = nn.Conv3d(channels[1], 1, 1)
        self.ds2 = nn.Conv3d(channels[2], 1, 1)

        self.organ_experts = organ_experts
        if organ_experts:
            self.router = nn.Linear(channels[4], len(DOMAIN_GROUPS))
            self.expert_d1 = RoutedResidualAdapter(
                channels[1], expert_rank, len(DOMAIN_GROUPS)
            )
            self.expert_d0 = RoutedResidualAdapter(
                channels[0], expert_rank, len(DOMAIN_GROUPS)
            )

    def forward(self, x: torch.Tensor, domain: torch.Tensor | None = None):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        router_logits = None
        if self.organ_experts:
            router_logits = self.router(F.adaptive_avg_pool3d(b, 1).flatten(1))
            routing = domain if domain is not None else router_logits.argmax(1)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        if self.organ_experts:
            d1 = self.expert_d1(d1, routing)
        d0 = self.up0(d1, e0)
        if self.organ_experts:
            d0 = self.expert_d0(d0, routing)
        main = self.head(d0)
        if not self.training or not self.deep_supervision:
            return main
        outputs = (main, self.ds1(d1), self.ds2(d2))
        return (*outputs, router_logits) if self.organ_experts else outputs


class GlanceFocusMoE(RECISTUNet3D):
    """Coarse-to-fine prompt-conditioned mixture of residual experts.

    The bottleneck and a coarse auxiliary mask form the Glance pathway.  A
    phenotype router softly mixes Focus adapters in the two finest decoder
    stages.  The coarse mask's uncertainty emphasizes the adapter residuals at
    ambiguous boundaries while leaving confident generalist predictions
    largely unchanged.
    """

    def __init__(self, in_channels: int = 4, base: int = 32,
                 deep_supervision: bool = True, experts: int = 4,
                 expert_rank: int = 8, router_temperature: float = 1.0):
        super().__init__(in_channels=in_channels, base=base,
                         deep_supervision=deep_supervision)
        channels = (base, base * 2, base * 4, base * 8, base * 10)
        hidden = max(32, base * 2)
        # The final scalar is the RECIST-derived log-diameter metadata.  It is
        # available at inference and encourages reproducible specialization.
        self.router = nn.Sequential(
            nn.Linear(channels[4] + 1, hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, experts),
        )
        self.glance_head = nn.Conv3d(channels[2], 1, 1)
        self.focus_expert_d1 = SoftRoutedResidualAdapter(
            channels[1], expert_rank, experts
        )
        self.focus_expert_d0 = SoftRoutedResidualAdapter(
            channels[0], expert_rank, experts
        )
        self.experts = experts
        self.router_temperature = router_temperature

        # Start with uniform routing and an exact copy of the pretrained model.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        nn.init.zeros_(self.glance_head.weight)
        nn.init.zeros_(self.glance_head.bias)

    @staticmethod
    def _focus_map(glance_logits: torch.Tensor,
                   shape: tuple[int, int, int]) -> torch.Tensor:
        probability = torch.sigmoid(glance_logits.detach())
        uncertainty = 4.0 * probability * (1.0 - probability)
        uncertainty = F.interpolate(
            uncertainty, size=shape, mode="trilinear", align_corners=False
        )
        # Retain a gradient path everywhere; concentrate most expert capacity
        # where the coarse branch is unsure.
        return 0.25 + 1.5 * uncertainty

    def forward(self, x: torch.Tensor):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        pooled = F.adaptive_avg_pool3d(b, 1).flatten(1)
        # Channel 3 is constant log(D/30)/1.5 by construction.
        size_token = x[:, 3, 0, 0, 0, None]
        router_logits = self.router(torch.cat((pooled, size_token), dim=1))
        routing = F.softmax(router_logits / self.router_temperature, dim=1)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        glance_logits = self.glance_head(d2)
        d1 = self.up1(d2, e1)
        d1 = self.focus_expert_d1(
            d1, routing, self._focus_map(glance_logits, d1.shape[2:])
        )
        d0 = self.up0(d1, e0)
        d0 = self.focus_expert_d0(
            d0, routing, self._focus_map(glance_logits, d0.shape[2:])
        )
        main = self.head(d0)
        if not self.training or not self.deep_supervision:
            return main
        return main, self.ds1(d1), self.ds2(d2), router_logits, glance_logits


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
