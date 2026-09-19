"""
Two 2D segmentation architectures chosen to differ in ONE thing that matters:
how much of the parameter budget is matrix-valued.

  UNet2D        -- conv-dominated. Muon touches flattened conv kernels only.
  TransUNetLite -- U-Net encoder/decoder + ViT bottleneck. The QKV/MLP
                   projections are genuine linear maps, so Muon's surface
                   area is much larger.

If Muon's advantage tracks frac_muon across these two, that supports the
spectral-geometry story. If it doesn't, the story is wrong -- which is the
more interesting outcome to report.

Both are sized to train comfortably on a 20 GB card at 224x224.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), gn(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), gn(cout), nn.GELU(),
        )

    def forward(self, x):
        return self.b(x)


class UNet2D(nn.Module):
    """Plain conv U-Net. The conv-dominated arm of the comparison."""

    def __init__(self, in_ch=1, n_classes=4, width=32, depth=4):
        super().__init__()
        chs = [width * 2 ** i for i in range(depth + 1)]
        self.in_conv = ConvBlock(in_ch, chs[0])          # 'in_conv' -> AdamW
        self.downs = nn.ModuleList(
            [ConvBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList(
            [ConvBlock(chs[i] * 2, chs[i]) for i in range(depth)])
        self.out_conv = nn.Conv2d(chs[0], n_classes, 1)  # 'out_conv' -> AdamW
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.in_conv(x)
        skips = [x]
        for d in self.downs:
            x = d(self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(reversed(self.ups), reversed(self.dec)):
            x = up(x)
            s = skips.pop()
            x = dec(torch.cat([x, s], dim=1))
        return self.out_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads=8, mlp=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(),
                                 nn.Linear(dim * mlp, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TransUNetLite(nn.Module):
    """
    U-Net encoder -> ViT bottleneck -> U-Net decoder.

    Deliberately NOT a faithful TransUNet reimplementation -- it is the
    matrix-heavy arm of a controlled comparison, and it must share the
    conv stem/decoder with UNet2D so the only variable is bottleneck type.
    """

    def __init__(self, in_ch=1, n_classes=4, width=32, depth=4,
                 dim=384, n_blocks=6, heads=6):
        super().__init__()
        chs = [width * 2 ** i for i in range(depth + 1)]
        self.in_conv = ConvBlock(in_ch, chs[0])
        self.downs = nn.ModuleList(
            [ConvBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)

        self.to_tok = nn.Conv2d(chs[-1], dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, 14 * 14, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.from_tok = nn.Conv2d(dim, chs[-1], 1)

        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList(
            [ConvBlock(chs[i] * 2, chs[i]) for i in range(depth)])
        self.out_conv = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        x = self.in_conv(x)
        skips = [x]
        for d in self.downs:
            x = d(self.pool(x))
            skips.append(x)
        x = skips.pop()

        B, C, H, W = x.shape
        t = self.to_tok(x).flatten(2).transpose(1, 2)
        pos = self.pos
        if pos.size(1) != t.size(1):                      # tolerate other sizes
            g = int(pos.size(1) ** 0.5)
            pos = F.interpolate(
                pos.reshape(1, g, g, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bicubic", align_corners=False
            ).flatten(2).transpose(1, 2)
        t = t + pos
        for b in self.blocks:
            t = b(t)
        x = self.from_tok(self.norm(t).transpose(1, 2).reshape(B, -1, H, W))

        for up, dec in zip(reversed(self.ups), reversed(self.dec)):
            x = up(x)
            s = skips.pop()
            x = dec(torch.cat([x, s], dim=1))
        return self.out_conv(x)


def build_model(name: str, in_ch=1, n_classes=4) -> nn.Module:
    if name == "unet":
        return UNet2D(in_ch, n_classes)
    if name == "transunet":
        return TransUNetLite(in_ch, n_classes)
    if name == "vmunet":
        from models_mamba import build_vmunet   # lazy: only imports mamba on demand
        return build_vmunet(in_ch, n_classes)
    raise ValueError(name)
