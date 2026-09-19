"""
Dice and HD95.

The empty-mask cases are where segmentation metrics quietly lie:

  * GT empty AND pred empty   -> Dice is undefined (0/0). Scoring it 1.0
    inflates every mean on sparse classes. Returned as NaN and excluded.
  * exactly one of them empty -> Dice 0, HD95 undefined (no surface to
    measure from). Returned as NaN for HD95 and reported separately as a
    failure count. Silently dropping these flatters whichever method fails
    more often, so the count has to appear in the paper.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.sum(), gt.sum()
    if p == 0 and g == 0:
        return np.nan
    return 2.0 * np.logical_and(pred, gt).sum() / (p + g)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask ^ binary_erosion(mask, border_value=0)


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0)) -> float:
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    sp, sg = _surface(pred), _surface(gt)
    if sp.sum() == 0 or sg.sum() == 0:
        return np.nan
    dt_g = distance_transform_edt(~sg, sampling=spacing)
    dt_p = distance_transform_edt(~sp, sampling=spacing)
    a, b = dt_g[sp], dt_p[sg]
    return float(max(np.percentile(a, 95), np.percentile(b, 95)))


def evaluate_slice(pred: np.ndarray, gt: np.ndarray, n_classes: int,
                   spacing=(1.0, 1.0)):
    """Returns per-foreground-class dice / hd95 arrays (NaN where undefined)
    plus a count of one-sided-empty failures."""
    d = np.full(n_classes - 1, np.nan)
    h = np.full(n_classes - 1, np.nan)
    fails = 0
    for c in range(1, n_classes):
        p, g = pred == c, gt == c
        d[c - 1] = dice_binary(p, g)
        h[c - 1] = hd95_binary(p, g, spacing)
        if (p.sum() == 0) != (g.sum() == 0):
            fails += 1
    return d, h, fails


def evaluate_volume(pred, gt, n_classes, spacing=(1.0, 1.0, 1.0)):
    """Score a whole (N,H,W) volume per foreground class.

    This is NOT the same as averaging per-slice Dice: Dice is computed on the
    3D intersection/union over the whole stack, and HD95 over the full 3D
    surface. That is the ACDC-correct definition and it will differ from a
    slice-averaged number, usually by being a little lower on Dice and much
    more stable on HD95 (no per-slice empty-mask blowups).

    spacing defaults to isotropic; if your test .npz stores real voxel spacing
    pass it here so HD95 comes out in mm.
    """
    d = np.full(n_classes - 1, np.nan)
    h = np.full(n_classes - 1, np.nan)
    fails = 0
    for c in range(1, n_classes):
        p, g = pred == c, gt == c
        d[c - 1] = dice_binary(p, g)
        h[c - 1] = _hd95_3d(p, g, spacing)
        if (p.sum() == 0) != (g.sum() == 0):
            fails += 1
    return d, h, fails


def _hd95_3d(pred, gt, spacing=(1.0, 1.0, 1.0)):
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    sp = pred ^ binary_erosion(pred, border_value=0)
    sg = gt ^ binary_erosion(gt, border_value=0)
    if sp.sum() == 0 or sg.sum() == 0:
        return np.nan
    dt_g = distance_transform_edt(~sg, sampling=spacing)
    dt_p = distance_transform_edt(~sp, sampling=spacing)
    return float(max(np.percentile(dt_g[sp], 95), np.percentile(dt_p[sg], 95)))


def aggregate_by_case(rows, n_classes):
    """rows: list of (case_id, dice_vec, hd95_vec).

    Averages within patient FIRST, then across patients. Slices from one
    patient are correlated; pooling them as if independent inflates n by
    ~an order of magnitude and will hand you significance that isn't there.
    """
    from collections import defaultdict
    per = defaultdict(lambda: ([], []))
    for cid, d, h in rows:
        per[cid][0].append(d)
        per[cid][1].append(h)

    cases, dice, hd = [], [], []
    # All-NaN slices are expected (a class absent from every slice of a case);
    # nanmean is the correct behaviour there -- only the warning is noise.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", "Mean of empty slice")
        for cid, (ds, hs) in sorted(per.items()):
            cases.append(cid)
            dice.append(np.nanmean(np.stack(ds), axis=0))
            hd.append(np.nanmean(np.stack(hs), axis=0))
    return cases, np.stack(dice), np.stack(hd)
