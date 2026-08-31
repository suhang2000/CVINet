"""
CVINet: Cognitive Visual Imagination Network for Camouflaged Object Detection.

Core components:
- CSI_Block (Cognitive Semantic Injection): Injects template semantics into original features during encoding.
- CSR_Block (Cognitive Structure Refinement): Refines structural details using high-frequency template information.

Data flow:
- Dual-stream PVTv2 encoder: Encodes both original image and generated template.
- CSI fusion at each encoder scale.
- FPN decoder with CSR high-frequency refinement.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Building Blocks
# -----------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU block with preserved spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CSI_Block(nn.Module):
    """
    Cognitive Semantic Injection Block.

    Fuses template features with original features through:
    1. Spatial attention: Multi-branch convolutions (1x1, 3x3, 3x3_dilated)
    2. Channel attention: Adaptive pooling + FC layers
    3. Residual injection: f_orig + delta

    Args:
        in_channels: Number of input channels
        reduced_channels: Number of reduced channels for attention computation
    """

    def __init__(self, in_channels: int, reduced_channels: int | None = None) -> None:
        super().__init__()
        reduced = reduced_channels if reduced_channels is not None else max(32, in_channels // 4)
        self.reduce_orig = nn.Conv2d(in_channels, reduced, kernel_size=1)
        self.reduce_gen = nn.Conv2d(in_channels, reduced, kernel_size=1)

        # Multi-scale spatial attention
        self.spatial_b1 = nn.Sequential(
            nn.Conv2d(reduced * 2, reduced, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )
        self.spatial_b2 = nn.Sequential(
            nn.Conv2d(reduced * 2, reduced, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )
        self.spatial_b3 = nn.Sequential(
            nn.Conv2d(reduced * 2, reduced, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )
        self.spatial_out = nn.Sequential(
            nn.Conv2d(reduced * 3, reduced, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Channel attention + residual fusion
        self.fuse = ConvBNReLU(in_channels * 3, in_channels, kernel_size=3, padding=1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(16, in_channels // 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(16, in_channels // 4), in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_orig: torch.Tensor, f_gen: torch.Tensor) -> torch.Tensor:
        # Spatial attention: concatenate + three-branch convolutions
        r_o = self.reduce_orig(f_orig)
        r_g = self.reduce_gen(f_gen)
        r_cat = torch.cat([r_o, r_g], dim=1)
        s1 = self.spatial_b1(r_cat)
        s2 = self.spatial_b2(r_cat)
        s3 = self.spatial_b3(r_cat)
        attn_spatial = self.spatial_out(torch.cat([s1, s2, s3], dim=1))

        # Semantic difference + channel attention
        f_cat = torch.cat([f_orig, f_gen, f_gen - f_orig], dim=1)
        delta = self.fuse(f_cat)
        delta = delta * self.channel_attn(delta)
        delta = delta * attn_spatial
        return f_orig + delta


class CSR_Block(nn.Module):
    """
    Cognitive Structure Refinement Block.

    Refines decoder features using high-frequency information from templates:
    1. High-frequency extraction: Local difference (f_temp - local_avg(f_temp))
    2. Spatial attention: Selects "where" to refine
    3. Channel attention: Selects "what" features are important
    4. Adaptive injection with learnable scale

    Args:
        channels: Number of input/output channels
        scale_init: Initial value for the learnable scale parameter
    """

    def __init__(self, channels: int, scale_init: float = 1e-5) -> None:
        super().__init__()
        # Local average pool for low-frequency extraction
        self.local_avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)

        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Channel attention (SE-like)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Learnable scale, initialized small for stable training start
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(self, f_dec: torch.Tensor, f_temp: torch.Tensor) -> torch.Tensor:
        # Extract high-frequency (edges, textures)
        f_low = self.local_avg(f_temp)
        f_high = f_temp - f_low

        # Spatial selection
        spatial_map = self.spatial_attn(torch.cat([f_dec, f_high], dim=1))

        # Channel selection
        channel_map = self.channel_attn(f_high)

        # Adaptive injection
        residue = f_high * spatial_map * channel_map
        return f_dec + residue * self.scale


# -----------------------------------------------------------------------------
# Encoder: Dual-stream PVTv2
# -----------------------------------------------------------------------------
class DualPVTv2(nn.Module):
    """
    Siamese PVTv2 encoder with features_only output at 4 scales.

    Args:
        backbone_name: Name of PVTv2 variant (e.g., pvt_v2_b2, pvt_v2_b4)
        pretrained: Whether to load pretrained weights
    """

    def __init__(self, backbone_name: str = "pvt_v2_b4", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.out_channels: Sequence[int] = self.backbone.feature_info.channels()

    def forward(self, img_orig: torch.Tensor, img_gen: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        feats_orig = self.backbone(img_orig)
        feats_gen = self.backbone(img_gen)
        return feats_orig, feats_gen


# -----------------------------------------------------------------------------
# Main Network: CVINet
# -----------------------------------------------------------------------------
class CVI_Net(nn.Module):
    """
    CVINet: Dual-stream PVTv2 + CSI (encoder fusion) + FPN + CSR (decoder refinement).

    Args:
        fpn_channels: Number of channels in FPN
        pretrained_backbone: Whether to load pretrained backbone weights
        backbone_name: Name of PVTv2 variant
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        pretrained_backbone: bool = True,
        backbone_name: str = "pvt_v2_b4",
    ) -> None:
        super().__init__()
        self.fpn_channels = fpn_channels

        self.encoder = DualPVTv2(backbone_name=backbone_name, pretrained=pretrained_backbone)
        c1, c2, c3, c4 = self.encoder.out_channels

        # CSI blocks for each scale
        self.csi_blocks = nn.ModuleList([
            CSI_Block(in_channels=c1),
            CSI_Block(in_channels=c2),
            CSI_Block(in_channels=c3),
            CSI_Block(in_channels=c4),
        ])

        # FPN 1x1 reduction + smoothing
        self.reduce_convs = nn.ModuleList([
            ConvBNReLU(c1, fpn_channels, kernel_size=1, padding=0),
            ConvBNReLU(c2, fpn_channels, kernel_size=1, padding=0),
            ConvBNReLU(c3, fpn_channels, kernel_size=1, padding=0),
            ConvBNReLU(c4, fpn_channels, kernel_size=1, padding=0),
        ])
        self.smooth3 = ConvBNReLU(fpn_channels, fpn_channels)
        self.smooth2 = ConvBNReLU(fpn_channels, fpn_channels)
        self.smooth1 = ConvBNReLU(fpn_channels, fpn_channels)

        # Template projection to FPN channels for CSR
        self.template_projs = nn.ModuleList([
            ConvBNReLU(c1, fpn_channels, kernel_size=1, padding=0),
            ConvBNReLU(c2, fpn_channels, kernel_size=1, padding=0),
            ConvBNReLU(c3, fpn_channels, kernel_size=1, padding=0),
            ConvBNReLU(c4, fpn_channels, kernel_size=1, padding=0),
        ])
        self.csr_blocks = nn.ModuleList([CSR_Block(fpn_channels) for _ in range(4)])

        # Prediction head (on P1, outputs logits)
        self.decode_head = nn.Sequential(
            ConvBNReLU(fpn_channels, fpn_channels // 2),
            ConvBNReLU(fpn_channels // 2, fpn_channels // 4),
            nn.Conv2d(fpn_channels // 4, 1, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize custom layers, preserving pretrained backbone weights."""
        for name, module in self.named_modules():
            if name.startswith("encoder.backbone"):
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _decode_with_fpn(
        self,
        fused_feats: List[torch.Tensor],
        template_feats: List[torch.Tensor],
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """FPN decoding with CSR refinement at each level."""
        f1_f, f2_f, f3_f, f4_f = fused_feats
        t1, t2, t3, t4 = template_feats

        # P4 (1/32): Top-level feature
        p4 = self.reduce_convs[3](f4_f)
        p4 = self.csr_blocks[3](p4, t4)

        # P3 (1/16)
        p3 = self.reduce_convs[2](f3_f) + F.interpolate(
            p4, size=f3_f.shape[-2:], mode="bilinear", align_corners=False
        )
        p3 = self.smooth3(p3)
        p3 = self.csr_blocks[2](p3, t3)

        # P2 (1/8)
        p2 = self.reduce_convs[1](f2_f) + F.interpolate(
            p3, size=f2_f.shape[-2:], mode="bilinear", align_corners=False
        )
        p2 = self.smooth2(p2)
        p2 = self.csr_blocks[1](p2, t2)

        # P1 (1/4)
        p1 = self.reduce_convs[0](f1_f) + F.interpolate(
            p2, size=f1_f.shape[-2:], mode="bilinear", align_corners=False
        )
        p1 = self.smooth1(p1)
        p1 = self.csr_blocks[0](p1, t1)

        logits = self.decode_head(p1)
        return F.interpolate(logits, size=img_size, mode="bilinear", align_corners=False)

    def forward(self, img_orig: torch.Tensor, img_gen: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, returns segmentation logits at input resolution:
        1. Encode: Dual-stream PVTv2 -> feats_orig, feats_temp
        2. CSI: Inject template semantics at each scale
        3. Decode: FPN + CSR high-frequency refinement
        """
        # Dual-stream encoding
        feats_orig, feats_temp = self.encoder(img_orig, img_gen)
        f1_o, f2_o, f3_o, f4_o = feats_orig
        f1_t, f2_t, f3_t, f4_t = feats_temp

        # CSI fusion at each scale
        f1_f = self.csi_blocks[0](f1_o, f1_t)
        f2_f = self.csi_blocks[1](f2_o, f2_t)
        f3_f = self.csi_blocks[2](f3_o, f3_t)
        f4_f = self.csi_blocks[3](f4_o, f4_t)

        img_size = (img_orig.shape[-2], img_orig.shape[-1])

        # Project template features and decode with FPN + CSR
        template_proj = [proj(feat) for proj, feat in zip(self.template_projs, [f1_t, f2_t, f3_t, f4_t])]
        return self._decode_with_fpn([f1_f, f2_f, f3_f, f4_f], template_proj, img_size)


__all__ = [
    "CVI_Net",
    "CSI_Block",
    "CSR_Block",
    "ConvBNReLU",
    "DualPVTv2",
]
