"""
VM-UNet-style Mamba segmentation model.

Faithful to the VM-UNet design (Ruan & Xiang, 2024, github.com/JCruan519/VM-UNet):
a U-shaped encoder/decoder built from VSS (Visual State Space) blocks, whose
core is SS2D -- a 2D selective scan that runs the Mamba selective-scan along
FOUR directions (H->, <-H, W->, <-W), merges them, and gates the result.

REAL RUN (GPU): uses mamba_ssm.selective_scan_fn (custom CUDA kernel). Install:
    pip install causal-conv1d>=1.1.0 mamba-ssm>=1.2.0
These require a CUDA GPU; they will not build/run on CPU.

FALLBACK (no kernel / CPU): if mamba_ssm import fails, SS2D falls back to a
pure-PyTorch sequential-scan implementation of the SAME recurrence. It is
numerically equivalent but slow (a Python loop over sequence length) -- only
for import checks, unit tests, and CPU smoke tests, NEVER for real timing.
A one-time warning is printed so you never mistake a fallback run for a real one.

PARAMETER GEOMETRY (why this model matters for the Muon study):
  * in_proj / out_proj / the SSM x_proj and dt_proj are genuine 2D linear maps
    -> Muon (native-2D, exact spectral-norm territory).
  * A_log (state matrix, log-parameterized), D (skip), dt bias, and the depthwise
    conv1d are 1D or degenerate -> routed to AdamW by split_params (ndim<2 or
    depthwise). This gives VM-UNet a DIFFERENT native-2D-of-Muon fraction than
    the conv U-Net (~0%) or TransUNet (57%), which is the contrast the study
    needs. split_params in muon.py already handles these by the existing rules;
    verify with `run_study.py --stage 0` once integrated.
"""
from __future__ import annotations

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- optional fast path (GPU CUDA kernel) --------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _HAS_MAMBA = True
except Exception:
    selective_scan_fn = None
    _HAS_MAMBA = False
    warnings.warn(
        "mamba_ssm not available -- SS2D using SLOW pure-PyTorch fallback scan. "
        "Fine for import/unit tests; install causal-conv1d + mamba-ssm on the GPU "
        "for real runs.", RuntimeWarning)


def gn(c):
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


# ---------------------------------------------------------------- SS2D core
def _selective_scan_ref(u, delta, A, B, C, D, delta_bias=None):
    """Pure-PyTorch selective scan (fallback). Matches mamba_ssm semantics.

    u:      (b, d, L)      input
    delta:  (b, d, L)      timestep
    A:      (d, n)         state matrix (negative, real)
    B, C:   (b, n, L)      input/output projections
    D:      (d,)           skip
    returns (b, d, L)
    """
    b, d, L = u.shape
    n = A.shape[1]
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    delta = F.softplus(delta)                          # (b,d,L)
    dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))     # (b,d,L,n)
    dB = torch.einsum("bdl,bnl->bdln", delta, B)                # (b,d,L,n)
    dBu = dA * 0 + dB * u.unsqueeze(-1)                          # (b,d,L,n)
    x = u.new_zeros(b, d, n)
    ys = []
    for t in range(L):                                          # sequential scan
        x = dA[:, :, t] * x + dBu[:, :, t]
        y = torch.einsum("bdn,bnl->bd", x, C[:, :, t:t+1])
        ys.append(y)
    y = torch.stack(ys, dim=-1)                                  # (b,d,L)
    return y + u * D[None, :, None]


class SS2D(nn.Module):
    """2D selective scan: four-directional Mamba scan + gating (VM-UNet core)."""

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv // 2, groups=self.d_inner, bias=True)
        # per-direction x_proj (dt, B, C) and dt_proj; 4 scan directions
        self.K = 4
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, (self.d_state * 2 + self.d_inner // self.d_inner)
                      + self.d_state * 0 + self._dt_rank() + self.d_state * 2, bias=False)
            for _ in range(self.K)])
        # simpler explicit projections (dt_rank + 2*d_state) per direction:
        self.dt_rank = self._dt_rank()
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(self.K)])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(self.K)])

        # A (state matrix) log-parameterized, and D skip -- per direction
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A[None, :].repeat(self.d_inner, 1)                  # (d_inner, d_state)
        self.A_logs = nn.ParameterList([nn.Parameter(torch.log(A.clone()))
                                        for _ in range(self.K)])
        self.Ds = nn.ParameterList([nn.Parameter(torch.ones(self.d_inner))
                                    for _ in range(self.K)])

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def _dt_rank(self):
        return max(1, self.d_model // 16)

    def _scan_dirs(self, x):
        """x: (b, d, H, W) -> 4 sequences (b, d, L) for the four directions."""
        b, d, H, W = x.shape
        hor = x.reshape(b, d, H * W)                            # H->
        ver = x.transpose(2, 3).reshape(b, d, H * W)            # W->
        return [hor, hor.flip(-1), ver, ver.flip(-1)]

    def _merge_dirs(self, ys, H, W):
        b, d, L = ys[0].shape
        hor = ys[0] + ys[1].flip(-1)
        ver = (ys[2] + ys[3].flip(-1)).reshape(b, d, W, H).transpose(2, 3).reshape(b, d, H * W)
        return (hor + ver).reshape(b, d, H, W)

    def _one_scan(self, seq, k):
        """Run selective scan for direction k on seq (b, d_inner, L)."""
        b, d, L = seq.shape
        xdbl = self.x_proj[k](seq.transpose(1, 2))              # (b,L,dt_rank+2*state)
        dt, B, C = torch.split(xdbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj[k](dt).transpose(1, 2)                # (b,d_inner,L)
        B = B.transpose(1, 2).contiguous()                      # (b,state,L)
        C = C.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_logs[k].float())                  # (d_inner,state)
        D = self.Ds[k].float()
        if _HAS_MAMBA and seq.is_cuda:
            y = selective_scan_fn(seq, dt, A, B.unsqueeze(1), C.unsqueeze(1),
                                  D, delta_bias=self.dt_proj[k].bias.float(),
                                  delta_softplus=True)
        else:
            y = _selective_scan_ref(seq, dt, A, B, C, D,
                                    delta_bias=self.dt_proj[k].bias.float())
        return y

    def forward(self, x):                                       # x: (b, H, W, d_model)
        b, H, W, _ = x.shape
        xz = self.in_proj(x)                                    # (b,H,W,2*d_inner)
        xi, z = xz.chunk(2, dim=-1)
        xi = xi.permute(0, 3, 1, 2).contiguous()                # (b,d_inner,H,W)
        xi = self.act(self.conv2d(xi))
        seqs = self._scan_dirs(xi)
        ys = [self._one_scan(seqs[k], k) for k in range(self.K)]
        y = self._merge_dirs(ys, H, W)                          # (b,d_inner,H,W)
        y = y.permute(0, 2, 3, 1).contiguous()                  # (b,H,W,d_inner)
        y = self.out_norm(y)
        y = y * self.act(z)
        return self.out_proj(y)                                 # (b,H,W,d_model)


class VSSBlock(nn.Module):
    """LayerNorm -> SS2D -> residual (VM-UNet basic block)."""
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state)
        self.scale = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x):                                       # (b,H,W,dim)
        return x + self.scale * self.ss2d(self.norm(x))


class _Down(nn.Module):
    def __init__(self, cin, cout, d_state):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.norm = gn(cout)
        self.block = VSSBlock(cout, d_state)

    def forward(self, x):
        x = F.silu(self.norm(self.reduce(x)))
        x = x.permute(0, 2, 3, 1)
        x = self.block(x)
        return x.permute(0, 3, 1, 2).contiguous()


class _Up(nn.Module):
    def __init__(self, cin, cout, d_state):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.reduce = nn.Conv2d(cout * 2, cout, 3, padding=1, bias=False)
        self.norm = gn(cout)
        self.block = VSSBlock(cout, d_state)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.silu(self.norm(self.reduce(torch.cat([x, skip], 1))))
        x = x.permute(0, 2, 3, 1)
        x = self.block(x)
        return x.permute(0, 3, 1, 2).contiguous()


class VMUNet(nn.Module):
    """VM-UNet-style Mamba U-Net. Matches the study's model interface:
    build_model(name, in_ch, n_classes) -> forward(x)->logits (b,n_classes,H,W)."""

    def __init__(self, in_ch=1, n_classes=4, width=32, depth=4, d_state=16):
        super().__init__()
        chs = [width * 2 ** i for i in range(depth + 1)]
        self.stem = nn.Sequential(nn.Conv2d(in_ch, chs[0], 3, padding=1, bias=False),
                                  gn(chs[0]), nn.SiLU())          # 'stem' -> AdamW
        self.downs = nn.ModuleList(
            [_Down(chs[i], chs[i + 1], d_state) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            [_Up(chs[i + 1], chs[i], d_state) for i in range(depth)])
        self.head = nn.Conv2d(chs[0], n_classes, 1)              # 'head' -> AdamW

    def forward(self, x):
        x = self.stem(x)
        skips = [x]
        for d in self.downs:
            x = d(self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up in reversed(self.ups):
            x = up(x, skips.pop())
        return self.head(x)


def build_vmunet(in_ch=1, n_classes=4):
    return VMUNet(in_ch, n_classes)
