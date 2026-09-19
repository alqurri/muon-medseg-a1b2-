#!/bin/bash
# Full-grid light-aug LR tuning, U-Net. AMOS omitted (16-class volume eval OOMs).
BRATS=./data/BraTS2020_npz
SYNAPSE=./data/Synapse
ACDC=./data/ACDC
ISIC=./data/ISIC2018
CAMUS=./data
FLARE=./data/FLARE22_npz

run_lt () {   # $1=dataset $2=n_classes $3=outdir $4=data_root
  echo "===== $1 light full-tune ====="
  python run_study.py --stage 1L --dataset "$1" --n-classes "$2" --models unet \
      --outdir "$3" --data-root "$4" && \
  python run_study.py --stage 3L --stage3-model unet --dataset "$1" --n-classes "$2" \
      --outdir "$3" --data-root "$4"
  echo "===== DONE $1 ====="
}

run_lt brats_npz 4  results_brats   "$BRATS"
run_lt synapse   9  results_synapse "$SYNAPSE"
run_lt acdc      4  results          "$ACDC"
run_lt isic      2  results_isic    "$ISIC"
run_lt camus     4  results_camus   "$CAMUS"
run_lt flare_npz 14 results_flare   "$FLARE"

echo "===== analysis ====="
for D in "results_brats brats" "results_synapse synapse" "results acdc" \
         "results_isic isic" "results_camus camus" "results_flare flare"; do
  set -- $D
  echo "===== $2 ====="
  python analyze_v2.py --outdir "$1" --dataset "$2"
done
echo "===== LIGHT FULL-TUNE COMPLETE ====="
