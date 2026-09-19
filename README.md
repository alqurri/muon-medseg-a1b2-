# Muon vs. AdamW for Medical Image Segmentation

Code to reproduce *"A Controlled Evaluation of Muon for Medical Image Segmentation"* (anonymous submission, under
review at TMLR).

We evaluate the Muon optimizer against AdamW across seven medical segmentation
datasets, three architectures (U-Net, TransUNet, a Mamba/state-space network),
and two augmentation regimes, using seed-level, patient-clustered statistics.

## Installation

```bash
pip install -r requirements.txt
```
Tested with Python 3.11, PyTorch 2.x, single GPU.

## Data

We do not redistribute the datasets. Obtain each from its source and place it
under `./data/` (or pass `--data-root`). Paths in the loaders default to
`./data/<DATASET>`.

| Dataset  | Source                          | Notes |
|----------|---------------------------------|-------|
| BraTS    | BraTS 2020 training set         | single FLAIR; labels {0,1,2,4}->{0,1,2,3} |
| AMOS22   | AMOS22 challenge                | MRI subset only (see `verify_amos_mri.py`) |
| FLARE22  | FLARE22 challenge (labeled)     | CT, HU window [-125, 275] |
| Synapse  | Multi-atlas abdominal CT (BTCV) | 8 organs + background |
| ACDC     | ACDC challenge                  | cardiac MRI, ED/ES |
| CAMUS    | CAMUS echocardiography          | cardiac ultrasound, ED/ES |
| ISIC     | ISIC 2018 (task 1)              | dermoscopy, binary |

Volumetric NIfTI datasets are preprocessed to slice `.npz`:
```bash
python preprocess_flare.py --root ./data/FLARE22Train --out ./data/FLARE22_npz
python preprocess_brats.py --root ./data/BraTS2020/... --out ./data/BraTS2020_npz
python preprocess_amos.py  --root ./data/AMOS22 --out ./data/AMOS22_npz --mri-min-id 500
```
Splits are patient-level with a fixed seed; validation is carved from the
training patients and the test set is held out. **Learning-rate selection uses
the validation split only.**

## Reproducing the study

Staged pipeline (`run_study.py`):
```bash
# Stage 1: LR sweep (heavy aug), 1 seed -> best_lrs.json
python run_study.py --stage 1 --dataset synapse --n-classes 9 --data-root ./data/Synapse
# Stage 2: final runs, 5 seeds, at selected LRs
python run_study.py --stage 2 --dataset synapse --n-classes 9 --data-root ./data/Synapse
# Stage 3: augmentation-interaction arm (heavy + light)
python run_study.py --stage 3 --stage3-model unet --dataset synapse --n-classes 9 --data-root ./data/Synapse
```

Controls used in the paper:
```bash
# Full-grid light-augmentation tuning (stages 1L / 3L)
python run_study.py --stage 1L --dataset synapse --n-classes 9 --models unet --data-root ./data/Synapse
python run_study.py --stage 3L --stage3-model unet --dataset synapse --n-classes 9 --data-root ./data/Synapse

# Orthogonalization-free control (Muon with Newton-Schulz removed)
python run_study.py --stage 1 --dataset brats_npz --n-classes 4 --models unet \
    --optimizers adamw muon_noortho --outdir results_brats_noortho --data-root ./data/BraTS2020_npz

# Difficulty-by-training-size probe (Synapse, fixed classes)
python run_study.py --stage 1 --dataset synapse --n-classes 9 --train-frac 0.5 \
    --models unet --outdir results_syn_frac0.5 --data-root ./data/Synapse
```
Convenience recipes: `run_frac.sh`, `run_noortho.sh`, `run_light_fulltune.sh`,
`run_acdc_noortho.sh` (edit the data paths at the top of each).

## Analysis: single source of truth

**`regenerate_all.py` computes every number and LaTeX table row in the paper**
from the results directories (deduplicated newest-per-seed, LR-filtered via each
dir's `best_lrs.json`/`best_lrs_light.json`, patient-clustered with ED/ES
averaging).

```bash
python regenerate_all.py            # human-readable summary
python regenerate_all.py --latex    # ready-to-paste LaTeX table rows
```

Supporting scripts:
- `analyze_v2.py`  — seed-level, patient-clustered deltas + hierarchical bootstrap CIs
- `interaction.py` — augmentation x optimizer interaction + per-seed consistency
- `check_ablation.py` — LR-filtered label-merge ablation deltas
- `paired_noortho.py`, `noortho_consistent.py` — Muon vs. Muon-NS paired comparison
- `verify_interaction.py` — dedup + LR-filter interaction check
- `verify_amos_mri.py` — confirms the AMOS22 MRI subset by intensity fingerprinting
- `make_figure.py` — the benefit-vs-baseline figure

## Statistical protocol

Patient-level clustering (ED/ES frames averaged), five training seeds treated as
a variance component, seed-level paired analysis, and a hierarchical bootstrap
resampling seeds and patients jointly. See the paper's Methods and Appendix.

## License

MIT (see LICENSE).
