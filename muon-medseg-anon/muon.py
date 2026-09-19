"""
Muon optimizer + hybrid Muon/AdamW parameter grouping.

Muon = momentum on the gradient matrix, then replace it with its orthogonal
polar factor (all singular values -> 1) via Newton-Schulz, then step.
This is steepest descent under the spectral norm.

Muon NEVER runs alone. 1D params (norm gains, biases), the stem/patch-embed,
the output head, and degenerate matrices (e.g. depthwise convs) go to AdamW.
Which params land where is a design decision -- see split_params() and the
Stage 0 report, which is a result in its own right.
"""
from __future__ import annotations

import math
import torch


# --------------------------------------------------------------------------
# Newton-Schulz: matmul-only approximation of the orthogonal polar factor UV^T
# --------------------------------------------------------------------------
# Quintic iteration with the coefficients from Keller Jordan's reference impl.
# These are deliberately NOT the coefficients that converge to the exact polar
# factor -- they are tuned to push all singular values into roughly [0.7, 1.3]
# in ~5 steps, which is all the optimizer needs and is far cheaper. Do not
# "fix" them to textbook Newton-Schulz values; that converges more slowly.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    assert G.ndim == 2, f"expected 2D, got {tuple(G.shape)}"
    a, b, c = NS_COEFFS
    # bf16 on GPU is the point: cheap, and NS is numerically forgiving.
    # On CPU, bf16 matmul is orders of magnitude slower than fp32, which makes
    # CPU smoke tests look like a hang. Fall back so the harness stays testable.
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + eps)

    transposed = G.size(0) > G.size(1)
    if transposed:                       # iterate on the smaller Gram matrix
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon for >=2D parameters only.

    Args
    ----
    lr            : Muon LRs live 1-2 orders of magnitude above AdamW's.
                    Sweep separately. Never share a grid with AdamW.
    momentum      : heavy-ball coefficient on the gradient matrix.
    nesterov      : use Nesterov-style lookahead on the momentum buffer.
    weight_decay  : decoupled. The original Muon omitted this; adding it is
                    one of the two changes that made Muon scale (Moonlight).
    rms_match     : the other one. Rescales the update so its RMS matches what
                    AdamW would produce for a matrix of the same shape, so a
                    single LR transfers across differently-shaped layers.
    ns_steps      : Newton-Schulz iterations. 5 is standard.
    """

    def __init__(self, params, lr=2e-2, momentum=0.95, nesterov=True,
                 weight_decay=0.0, rms_match=True, ns_steps=5,
                 orthogonalize=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, rms_match=rms_match,
                        ns_steps=ns_steps, orthogonalize=orthogonalize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:                       # conv: (Cout, Cin, *k)
                    g = g.view(g.size(0), -1)        # -> (Cout, Cin*prod(k))

                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(g)
                buf = st["mom"]
                buf.mul_(mu).add_(g)
                upd = g.add(buf, alpha=mu) if group["nesterov"] else buf

                if group["orthogonalize"]:
                    upd = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                    if group["rms_match"]:
                        upd = upd * (0.2 * math.sqrt(max(upd.size(0), upd.size(1))))
                else:
                    # orthogonalization-free control: same momentum + magnitude,
                    # only the singular-value structure differs.
                    if group["rms_match"]:
                        m_, n_ = upd.size(0), upd.size(1)
                        target = 0.2 * math.sqrt(max(m_, n_)) * \
                                 math.sqrt(min(m_, n_) / (m_ * n_))
                        rms = upd.square().mean().sqrt().clamp_min(1e-12)
                        upd = upd * (target / rms)

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(upd.view_as(p), alpha=-lr)

        return loss


# --------------------------------------------------------------------------
# Parameter grouping  (Stage 0)
# --------------------------------------------------------------------------
def _is_depthwise(module) -> bool:
    return getattr(module, "groups", 1) > 1 and \
        getattr(module, "groups", 1) == getattr(module, "in_channels", -1)


def split_params(model, head_names=("head", "out_conv", "classifier", "seg_head"),
                 stem_names=("stem", "patch_embed", "in_conv"),
                 embed_names=("pos", "embed", "cls_token"),
                 muon_depthwise=False):
    """
    Returns (muon_params, adamw_params, report).

    Rules, all of them contestable -- state whichever you use in the paper:
      * ndim < 2                      -> AdamW  (norm gains, biases)
      * ndim == 3 / positional embeds -> AdamW  (an embedding table is not a
                                        linear map; flattening (1,N,D) to
                                        (1, N*D) makes orthogonalisation a
                                        no-op that just renormalises)
      * stem / patch-embed            -> AdamW  (input layer)
      * segmentation head             -> AdamW  (output layer)
      * depthwise conv                -> AdamW  (flattens to (C, k*k); the
                                        orthogonality constraint on a C x 9
                                        matrix is close to vacuous)
      * everything else with ndim>=2  -> Muon

    The report distinguishes NATIVE-2D Muon params (nn.Linear -- genuine
    linear maps, where the spectral-norm argument actually holds) from
    FLATTENED-CONV Muon params (4D kernels reshaped to 2D, where it is an
    analogy). That split, not the raw Muon fraction, is the contrast that
    tests the mechanism.
    """
    dw_params = set()
    for m in model.modules():
        if _is_depthwise(m) and hasattr(m, "weight"):
            dw_params.add(id(m.weight))

    muon, adamw, rows = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if p.ndim < 2:
            why = "1D (norm/bias)"
        elif p.ndim == 3 or any(e in low.split(".") for e in embed_names):
            why = "embedding"
        elif any(h in low for h in head_names):
            why = "output head"
        elif any(s in low for s in stem_names):
            why = "input stem"
        elif id(p) in dw_params and not muon_depthwise:
            why = "depthwise"
        else:
            why = "MUON"

        (muon if why == "MUON" else adamw).append(p)
        flat = (p.size(0), p.numel() // p.size(0)) if p.ndim >= 2 else None
        rows.append((name, tuple(p.shape), flat, p.numel(), why, p.ndim))

    n_m = sum(p.numel() for p in muon)
    n_a = sum(p.numel() for p in adamw)
    n_lin = sum(r[3] for r in rows if r[4] == "MUON" and r[5] == 2)
    n_cnv = sum(r[3] for r in rows if r[4] == "MUON" and r[5] == 4)
    report = dict(rows=rows, n_muon=n_m, n_adamw=n_a,
                  frac_muon=n_m / max(1, n_m + n_a),
                  n_muon_linear=n_lin, n_muon_conv=n_cnv,
                  frac_linear_of_muon=n_lin / max(1, n_m))
    return muon, adamw, report


def print_split_report(report, title="", show_layers=False):
    """Stage 0 output. Run this BEFORE training anything.

    frac_muon is the ceiling on how much of the network Muon can possibly
    affect. If it is 0.35 for your U-Net and 0.80 for your transformer model,
    that asymmetry IS the hypothesis -- report it up front.
    """
    tot = report["n_muon"] + report["n_adamw"]
    print(f"\n=== parameter split: {title} ===")
    print(f"  total          {tot:>12,}")
    print(f"  Muon           {report['n_muon']:>12,}  ({report['frac_muon']:6.1%})")
    print(f"    native 2D    {report['n_muon_linear']:>12,}  "
          f"({report['frac_linear_of_muon']:6.1%} of Muon)  <- genuine linear maps")
    print(f"    flat. conv   {report['n_muon_conv']:>12,}  "
          f"({1-report['frac_linear_of_muon']:6.1%} of Muon)  <- 4D reshaped to 2D")
    print(f"  AdamW          {report['n_adamw']:>12,}  ({1-report['frac_muon']:6.1%})")

    import numpy as np
    for lbl, nd in (("2D  ", 2), ("conv", 4)):
        ar = [r for r in report["rows"] if r[4] == "MUON" and r[5] == nd and r[2]]
        if not ar:
            continue
        asp = np.array([max(f) / max(1, min(f)) for _, _, f, _, _, _ in ar])
        print(f"  {lbl} matrices: {len(ar):>3}  aspect ratio "
              f"median {np.median(asp):6.1f}  max {asp.max():8.1f}")
    print("  (aspect >> 1 => orthogonality is a weak constraint: few rows "
          "in a very high-dim space)")
    if show_layers:
        for name, shape, flat, n, why, nd in report["rows"]:
            print(f"    {why:>14}  {name:<44} {str(shape):<22} {n:>10,}")


def build_optimizers(model, optimizer: str, lr: float, weight_decay: float = 1e-2,
                     adamw_lr_for_hybrid: float | None = None, verbose=True):
    """
    optimizer='adamw' -> everything on AdamW (the baseline).
    optimizer='muon'  -> hybrid: matrices on Muon, the rest on AdamW.

    Returns a list of optimizers; step() them all.

    NOTE on the hybrid: the AdamW side needs its own LR. Holding it fixed at
    the tuned AdamW-baseline value keeps the comparison interpretable -- the
    only thing that changed is which rule the matrices get.
    """
    muon_p, adamw_p, report = split_params(model)
    if verbose:
        print_split_report(report, title=f"{model.__class__.__name__} / {optimizer}")

    if optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=weight_decay)], report

    if optimizer in ("muon", "muon_noortho"):
        aux_lr = adamw_lr_for_hybrid if adamw_lr_for_hybrid is not None else 3e-4
        ortho = (optimizer == "muon")
        opts = [Muon(muon_p, lr=lr, momentum=0.95, nesterov=True,
                     weight_decay=weight_decay, rms_match=True,
                     orthogonalize=ortho)]
        if adamw_p:
            opts.append(torch.optim.AdamW(adamw_p, lr=aux_lr, betas=(0.9, 0.999),
                                          weight_decay=weight_decay))
        return opts, report

    raise ValueError(optimizer)


# --------------------------------------------------------------------------
# Diagnostic: is Newton-Schulz actually orthogonalising YOUR momentum?
# --------------------------------------------------------------------------
@torch.no_grad()
def ns_diagnostic(optimizer: "Muon", steps_grid=(5, 8, 12)) -> list[dict]:
    """
    Muon's premise is that momentum matrices are near-low-rank. But NS with
    5 steps only flattens a WELL-conditioned spectrum; on a genuinely
    near-low-rank matrix it reduces conditioning by orders of magnitude and
    still leaves it far from flat -- i.e. it may under-orthogonalise exactly
    where it is supposed to help most.

    Whether that bites depends on your actual momentum spectra, so measure it
    rather than assuming. Call this a few times mid-training:

        if step % 500 == 0:
            for d in ns_diagnostic(muon_opt):
                print(d)

    If cond_out at 5 steps is large (say >10) for the layers that carry most
    of the parameters, raise ns_steps and re-run Stage 1 -- otherwise you are
    benchmarking a truncated approximation of Muon, not Muon.
    """
    out = []
    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            st = optimizer.state.get(p, {})
            if "mom" not in st:
                continue
            M = st["mom"].float()
            s0 = torch.linalg.svdvals(M)
            rec = dict(shape=tuple(p.shape),
                       flat=tuple(M.shape),
                       cond_in=float(s0.max() / s0.clamp_min(1e-12).min()),
                       # effective rank: exp(entropy of normalised spectrum)
                       eff_rank=float(torch.exp(-(lambda q: (q * q.log()).sum())(
                           s0 / s0.sum() + 1e-12))))
            for k in steps_grid:
                s1 = torch.linalg.svdvals(
                    zeropower_via_newtonschulz5(M, steps=k).float())
                rec[f"cond_out@{k}"] = float(s1.max() / s1.clamp_min(1e-12).min())
            out.append(rec)
    return out
