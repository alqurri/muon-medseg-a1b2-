# Code corrections (in response to code review, round 2)

Round-1 fixes confirmed correct by the reviewer: momentum cycling
(`cycle_momentum=False`), `--optimizers` for heavy tuning/final, Mamba A_logs
routing, Mamba dt bias applied once, aux AdamW rate fixed at 3e-4, RMS matching,
cache-key fields (n_classes/train_frac/batch/aux LR), BraTS stride 2 / cap 60.

## Round-2 fixes applied

- run_study.py: stages 3 and 3L now honor `--optimizers` (previously they
  silently dropped muon_noortho and ran AdamW only).
- run_study.py: cache key now also includes size, clip, weight_decay,
  eval_every, and a PROTOCOL_VERSION constant; `do()` validates the stored
  config (dataset, n_classes, train_frac, model, optimizer, lr, seed, aug)
  against the requested config on a cache hit and refuses a mismatch.
- regenerate_all.py `load()`: now enforces dataset identity, class count, and
  training fraction, so a BraTS-configured result can no longer satisfy a
  Synapse request.
- Inconsistent analysis scripts (analyze.py, analyze_v2.py, interaction.py,
  make_figure.py, paired_noortho.py, check_ablation.py) carry a HISTORICAL/
  DEMONSTRATION header. regenerate_all.py is the single pipeline for reported
  numbers.
- requirements.txt: added pandas, matplotlib, Pillow.

## Corrections to the round-1 FIXES doc (reviewer was right)

- ISIC version: the **manuscript names ISIC 2016**, and the loader is written
  for ISIC 2016; the **README incorrectly said 2018**. The README has been
  corrected to 2016. (Earlier note had this backwards.)
- Newton--Schulz RMS: the round-1 note called the NS-output RMS
  "~0.177*sqrt(max)". That is wrong -- 0.177676 was the already-scaled update
  RMS for one specific 32x32 matrix, not a universal formula. The corrected
  control computes the reference orthogonalized update on the SAME input
  momentum and matches its RMS; describe it as "removes the orthogonalized
  direction while retaining that update's reference RMS", not as removing the
  Newton--Schulz computation.

## STILL REQUIRES YOUR ACTION (cannot be fixed from code alone)

1. regenerate_all.py does not yet emit HD95, the label-merge ablation, the
   training-fraction probe, or the orthogonalization-ablation tables. Either
   extend it to generate every reported number, or state in the paper which
   script produces each table (and validate that script the same way).

2. Data/protocol vs manuscript mismatches to reconcile in the WRITEUP:
   - Synapse loader ignores --data-root (relies on externally prepared,
     CT-windowed files) -> document this, or wire --data-root through.
   - ACDC uses center crop/pad, not resize -> fix the manuscript wording.
   - BraTS/AMOS normalize per slice, not per volume -> fix wording.
   - Validation runs every `eval_every` (default 5) epochs -> "best epoch"
     means best among evaluated checkpoints; say so.
   - Every sweep run also writes test metrics -> soften "test untouched until
     final evaluation" to "test is never used for model or hyperparameter
     selection" (LR selection uses best_val_dice only).

3. Regenerating preprocessed data: changing BraTS stride/cap in code does NOT
   update existing .npz slice files. Re-run preprocess_brats.py to adopt
   stride 2 / cap 60 before any BraTS rerun.

## The bottom line is unchanged

Corrected code does not retroactively validate the paper's tables. The path
remains: recover the exact code+results behind the paper, reconcile the writeup
with what actually ran, rerun the conditions affected by the confirmed bugs
(momentum, Mamba, and any cache-contaminated ablation/fraction cells), and
recompute EVERY table through regenerate_all.py.

## Round-3 fixes applied

1. Protocol version is now saved in every result's config (run_study.do stamps
   `_protocol_version`), and regenerate_all.load() REQUIRES it to match, REQUIRES
   an exact 200-epoch budget (not >=150), and REQUIRES dataset/n_classes/seed/
   epochs metadata to be present -- missing or mismatched metadata now raises
   instead of silently accepting the run. This prevents old (uncorrected) and
   corrected runs from mixing in one analysis.

2. The shell recipes (run_noortho.sh, run_acdc_noortho.sh, run_light_fulltune.sh,
   run_frac.sh) now carry a header pointing to regenerate_all.py /
   noortho_consistent.py for reported numbers and marking the analyze_v2.py call
   as a historical sanity check only.

3. regenerate_all.compute() now selects the light-LR file PER MODEL: it uses
   best_lrs_light.json for a given model only if that file contains that model's
   keys, else falls back to best_lrs.json -- so TransUNet light cells no longer
   disappear when only U-Net was light-tuned. The README no longer claims
   regenerate_all covers every table (HD95/ablation/fraction/noortho still need
   dedicated, provenance-guarded scripts).

## STILL OUTSTANDING (unchanged, require your action)

- Implement the missing table analyses (HD95, label-merge, training-fraction,
  orthogonalization) inside the provenance-guarded pipeline, or the paper cannot
  claim all tables come from one validated source.
- Manuscript reconciliation (point 4, all rounds): Synapse --data-root, ACDC
  crop-vs-resize, per-slice vs per-volume normalization, eval-every-5, test
  evaluated during sweeps. These must be DESCRIBED accurately in the paper.
- Rerun every condition affected by the confirmed training/model bugs, bump the
  protocol version so old runs are rejected, and recompute through one pipeline.

## Round-4 fixes applied

1. [analysis no longer crashes on sweep files] Runs now carry a `run_role`
   field ("tuning" for sweep budgets, "final" for the full budget), stamped by
   run_study.do(). regenerate_all.load() and noortho_consistent.load() SKIP
   tuning runs BEFORE applying the exact-200-epoch requirement, so an 80-epoch
   sweep at the selected LR no longer raises. (Verified: an 80-epoch tuning file
   and a 200-epoch final file in one directory -> only the final run is used.)

2. [cache-hit protocol validation] do() now validates the stored
   `_protocol_version` on a cache hit and REFUSES to reuse a result whose
   protocol differs (including unstamped results from a previous release), rather
   than returning it. (Verified: a protocol-1 run is rejected, not silently
   reused.) Note: PROTOCOL_VERSION is NOT stamped onto historical results; their
   provenance must be established by rerunning, not relabeling.

3. [auxiliary-LR + config validation] load() now also validates the auxiliary
   AdamW LR (`adamw_lr_for_hybrid`) against an expected value when provided, in
   addition to protocol, epochs, dataset, class count, and training fraction.

4. [recipes route through validated analyses] run_noortho.sh,
   run_light_fulltune.sh, and run_acdc_noortho.sh no longer EXECUTE the
   historical analyze_v2.py; they invoke regenerate_all.py / noortho_consistent.py.
   noortho_consistent.py now enforces the protocol/run_role/epoch guards too.

## STILL OUTSTANDING (unchanged)

- Missing table analyses (HD95, label-merge, training-fraction, orthogonalization)
  must be implemented inside the provenance-guarded pipeline.
- Manuscript/data-processing mismatches must be described accurately in the paper.
- No training-correctness is established by static checks: the affected
  conditions must be RERUN at protocol 2 and every table recomputed through one
  validated pipeline. Old result files (no protocol stamp) are now correctly
  rejected by that pipeline.

## Round-5 fixes applied (connecting the checks, not just providing them)

1+2. Auxiliary-LR validation is now ACTIVE in the main pipeline: cell() passes
   expect_aux_lr=3e-4 and require_aux=True for Muon runs, and load() REQUIRES the
   adamw_lr_for_hybrid metadata to be present and equal to 3e-4 (missing/None now
   raises). (Verified: a Muon run with aux LR 0.001 is rejected by cell().)

3. noortho_consistent.load() now applies the same strict identity checks as
   regenerate_all: dataset root match, required dataset/n_classes/seed metadata,
   in addition to protocol and epochs. (A BraTS result no longer satisfies a
   Synapse request.)

4. run_noortho.sh and run_light_fulltune.sh now EXECUTE
   `python regenerate_all.py --latex` (previously they only echoed the command).

5. run_acdc_noortho.sh now READS the muon/adamw/noortho learning rates from the
   relevant best_lrs.json files instead of hardcoding 0.08/0.0003/0.001.

6. run_role is set EXPLICITLY in every stage (tuning in stage1/stage1_light,
   final in stage2/stage3/stage3_light) rather than inferred from epoch count,
   so a non-default --sweep-epochs (e.g. 180) no longer misclassifies tuning as
   final. do() still falls back to an epoch-based guess only for configs that
   never set the field (older/manual runs).

## STILL OUTSTANDING (unchanged)

- Missing table analyses (HD95, label-merge, training-fraction, orthogonalization)
  are not yet implemented inside the guarded pipeline.
- Manuscript/data-processing mismatches must be described accurately in the paper.
- No experiment outputs are included; the reported numbers cannot be verified.
  Affected conditions must be rerun at protocol 2 and all tables recomputed
  through one validated pipeline.

## Round-6 fixes applied (all functionally verified)

1. run_role is now in the cache key AND validated on cache hits, so a
   200-epoch tuning run and a final run no longer share a filename/cache entry.
   stage3_light() now sets run_role="final" explicitly (no longer relies on the
   epoch-count fallback). Verified: tuning vs final produce different cache hashes.

2. noortho_consistent.load() now validates the n_classes VALUE (not just its
   presence), the train_frac, and the auxiliary AdamW LR (require_aux), matching
   regenerate_all's guards; new --n-classes/--train-frac/--aux-lr args feed these.
   Verified: a 2-class result is rejected for a 9-class request; a 0.001 aux LR
   is rejected when 3e-4 is expected.

3. Recipes now invoke the correct analysis:
   - run_noortho.sh calls noortho_consistent.py per dataset (BraTS/Synapse/ACDC),
     reading each dataset's selected muon/adamw/noortho LRs from best_lrs.json.
   - run_light_fulltune.sh runs `regenerate_all.py --latex` ONCE after the loop
     (previously it repeated the full output for every dataset).

## STILL OUTSTANDING (unchanged)

- Missing table analyses (HD95, label-merge, training-fraction) inside the
  guarded pipeline.
- Manuscript/data-processing mismatches to describe accurately in the paper.
- No experiment outputs; numbers unverifiable. Affected conditions must be rerun
  at protocol 2 and all tables recomputed through the validated pipeline.

## Round-7 fix

- run_acdc_noortho.sh now passes --n-classes 4. More robustly,
  noortho_consistent.py now DERIVES the expected class count from the dataset
  (synapse=9, flare=14, amos=16, brats/acdc/camus=4, isic=2) when --n-classes is
  not given, and errors on an unknown dataset -- so the class-count value check
  can no longer be silently skipped by any recipe. Verified: an ACDC run with
  n_classes=9 is rejected against the derived expected value of 4.

## Round-8 fix (found by an actual rerun)

- data_synapse.py: build_synapse_datasets() now honors --data-root. It was
  ignoring the passed root and using hardcoded ./data/Synapse/{train_npz,
  test_vol_h5,lists_Synapse}, so every Synapse run failed with
  FileNotFoundError on ./data/Synapse/lists_Synapse/train.txt. When root is
  given (train.py passes cfg["data_root"]), the three paths are now derived
  from it; explicit train_npz/test_h5/lists args still override. This is the
  reviewer's "Synapse ignores --data-root" item, confirmed live by a rerun and
  now fixed.

## Round-9 fix (found by the full rerun)

- train.py _forward_pred(): the evaluation forward pass now chunks the input
  over its first dim (default 16 slices per forward) instead of pushing an
  entire test volume through the model at once. AMOS22's 16-class volume
  evaluation OOMed (3 GiB group_norm allocation) regardless of --batch, because
  --batch controls training memory, not the eval forward. Chunking makes the
  eval memory bounded; predictions are identical. AMOS22 can now be evaluated.
