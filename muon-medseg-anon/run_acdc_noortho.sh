#!/bin/bash
# ACDC orthogonalization-free control (val->valid symlink now fixed)
ACDC=./data/ACDC

# clear failed markers + empty best_lrs from the crashed attempts
rm -f results_acdc_noortho/*.failed results_acdc_noortho/best_lrs.json

python run_study.py --stage 1 --dataset acdc --n-classes 4 --models unet \
    --optimizers adamw muon_noortho --outdir results_acdc_noortho --data-root "$ACDC" && \
python run_study.py --stage 2 --dataset acdc --n-classes 4 --models unet \
    --optimizers adamw muon_noortho --outdir results_acdc_noortho --data-root "$ACDC" && \
python analyze_v2.py --outdir results_acdc_noortho --dataset acdc

echo "===== VERIFY: completed vs failed runs ====="
echo "completed: $(ls results_acdc_noortho/*.json 2>/dev/null | grep -v failed | wc -l)"
echo "failed:    $(ls results_acdc_noortho/*.failed 2>/dev/null | wc -l)"
echo "===== ACDC NO-ORTHO COMPLETE ====="
