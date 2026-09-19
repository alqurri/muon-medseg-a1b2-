#!/bin/bash
DATA=./data/Synapse   # <-- adjust to your actual Synapse path

for FR in 0.25 0.5 0.75 1.0; do
  echo "===== train_frac $FR ====="
  python run_study.py --stage 1 --dataset synapse --n-classes 9 \
      --train-frac $FR --models unet --outdir results_syn_frac${FR} --data-root $DATA && \
  python run_study.py --stage 2 --dataset synapse --n-classes 9 \
      --train-frac $FR --models unet --outdir results_syn_frac${FR} --data-root $DATA
  echo "===== DONE frac $FR ====="
done

echo "===== ALL TRAINING DONE, running analysis ====="
for FR in 0.25 0.5 0.75 1.0; do
  echo "===== analyze frac $FR ====="
  python analyze_v2.py --outdir results_syn_frac${FR} --dataset synapse
done
echo "===== EXPERIMENT 3 COMPLETE ====="
