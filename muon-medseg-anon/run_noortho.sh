#!/bin/bash
# Experiment #5: orthogonalization-free control (muon_noortho vs adamw)
# AMOS omitted (16-class volume eval OOMs); 3 datasets suffice for the control.

BRATS=./data/BraTS2020_npz
SYNAPSE=./data/Synapse
ACDC=./data/ACDC

run_ds () {   # $1=dataset $2=n_classes $3=outdir $4=data_root
  echo "===== $1 (noortho) ====="
  python run_study.py --stage 1 --dataset "$1" --n-classes "$2" --models unet \
      --optimizers adamw muon_noortho --outdir "$3" --data-root "$4" && \
  python run_study.py --stage 2 --dataset "$1" --n-classes "$2" --models unet \
      --optimizers adamw muon_noortho --outdir "$3" --data-root "$4"
  echo "===== DONE $1 ====="
}

run_ds brats_npz 4  results_brats_noortho "$BRATS"
run_ds synapse   9  results_syn_noortho   "$SYNAPSE"
run_ds acdc      4  results_acdc_noortho  "$ACDC"

echo "===== ALL TRAINING DONE, running analysis ====="
for D in "results_brats_noortho brats" "results_syn_noortho synapse" \
         "results_acdc_noortho acdc"; do
  set -- $D
  echo "===== analyze $2 ====="
  python analyze_v2.py --outdir "$1" --dataset "$2"
done
echo "===== EXPERIMENT 5 COMPLETE ====="
