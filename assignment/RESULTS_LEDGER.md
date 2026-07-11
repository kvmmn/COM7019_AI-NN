# Results Ledger: COM7019 (FROZEN)

Source of truth for every number in the report. All values come from one clean top-to-bottom Colab run of `_code/COM7019_25199053.ipynb` (the final, simplified student notebook; executed with outputs, 0 stored errors, 24/24 code cells executed sequentially, execution_count 1-24 with no re-runs or skips).

- Run completed: 2026-07-11 (exact time not printed; notebook removes version/timestamp printing by design, see D-011). Best available timestamp is the output table's file time, 17:32 UTC.
- Environment: Google Colab, GPU T4 (from notebook metadata). Exact Python/TensorFlow/NumPy/pandas versions were not captured in-notebook this run, because the "print the versions" cell was deliberately removed for readability (D-011). If exact versions are needed for the appendix, they must be read from the Colab runtime info panel and added manually (open item, see below).
- Seed: 25199053 (student ID), set once via `keras.utils.set_random_seed`.
- Do not edit values here. If the notebook is rerun, replace this ledger wholesale and re-verify every quoted number in drafts.

**This ledger supersedes the previous version**, which was frozen against the earlier, more advanced-looking notebook (`COM7019_25199053_experiments.ipynb`, now in `_code/archive/`, superseded by D-009 to D-012). The final notebook for the code appendix is `_code/COM7019_25199053.ipynb`; the professional-style original and the not-yet-rerun seed-robustness notebook are kept in `_code/archive/` for reference only.

## Dataset facts

| Fact | Value |
|---|---|
| Rows | 9,909 trading days |
| Date range | 1980-12-12 to 2020-04-01 |
| Missing values / duplicate dates | 0 / 0 |
| Target variable | Adj Close (USD) |
| Split (chronological 70/15/15) | train rows 0-6935 (ends 2008-06-10); val rows 6936-8421 (ends 2014-05-06); test rows 8422-9908 (ends 2020-04-01) |
| Price range | min 0.1556, max 327.20 across the full series |
| Windowing | 60-day look-back (30-day tested as an ablation), per-window normalisation against the window's last price |
| Samples after windowing | 9,849 total → train 6,876 / val 1,486 / test 1,487 |

## Test-set results (all runs, sorted by test RMSE; values in USD unless noted)

| Model | Params | Train time (s) | Val RMSE | Test RMSE | Test MAE | Test MAPE % |
|---|---:|---:|---:|---:|---:|---:|
| CNN32k3_LSTM64_finetuned | 25,025 | 22.0 | - | 3.2213 | 1.8253 | 1.1606 |
| CNN32k3_LSTM64 | 25,025 | 24.3 | - | 3.2278 | 1.8237 | 1.1586 |
| Baseline (persistence) | 0 | - | - | 3.2380 | 1.8269 | 1.1589 |
| GRU_64_d0_lr1e-4 | 12,929 | 5.7 | - | 3.2385 | 1.8249 | 1.1577 |
| LSTM_64x32_d0 | 29,345 | 32.4 | - | 3.2403 | 1.8247 | 1.1582 |
| GRU_64x32_d02 | 22,305 | 8.9 | 0.8785 | 3.2422 | 1.8491 | 1.1734 |
| LSTM_64x32_d02 | 29,345 | 20.0 | 0.8738 | 3.2427 | 1.8252 | 1.1585 |
| LSTM_64_d02 | 16,961 | 18.8 | 0.8743 | 3.2486 | 1.8281 | 1.1607 |
| GRU_64_d02 | 12,929 | 17.5 | 0.8734 | 3.2519 | 1.8279 | 1.1597 |
| GRU_64_d0 | 12,929 | 18.8 | - | 3.2627 | 1.8304 | 1.1610 |
| GRU_64_d0_w30 | 12,929 | 30.0 (40 epochs, hit the cap) | - | 3.2682 | 1.8318 | 1.1626 |

Naming: `d02` = dropout 0.2, `d0` = no dropout, `w30` = 30-day window, `lr1e-4` = fine-tuned learning rate 0.0001. Val RMSE is only printed in the notebook for the four Stage-A models (plain LSTM/GRU, one vs two recurrent layers) used for the LSTM-vs-GRU pick; model selection between stages used validation RMSE only, never the test set.

## Headline findings (for drafting; wording must not exceed these)

1. **The fine-tuned CNN-LSTM hybrid has the lowest test RMSE overall**: 3.2213 vs baseline 3.2380, a 0.52% improvement. The un-fine-tuned hybrid alone (3.2278) already beats the baseline by 0.31%. Both edges are real in this single run but small; wording must stay modest ("a small edge", not "clearly outperforms").
2. **Neither LSTM nor GRU beats the baseline on test RMSE** in this run (best recurrent-only result: GRU_64_d0_lr1e-4 at 3.2385, essentially level with persistence at 3.2380).
3. **LSTM vs GRU (the brief's required comparison), by validation RMSE:** best GRU (GRU_64_d02, 0.8734) is marginally ahead of best LSTM (LSTM_64x32_d02, 0.8738), a 0.04% difference, i.e. practically tied. GRU_64_d02 also has about 24% fewer parameters than LSTM_64x32_d02-class models at equal width (12,929 vs 16,961+), so GRU is the more efficient choice at equivalent accuracy.
4. **Dropout ablation (GRU, 64 units):** dropout 0.2 gives a lower test RMSE than no dropout (3.2519 vs 3.2627), about 0.33% better. This run does not print a validation-side comparison for the no-dropout variant, so no claim should be made about validation-side ranking.
5. **Window length:** the 30-day window (GRU_64_d0_w30, test RMSE 3.2682) is worse than the 60-day window (GRU_64_d0, test RMSE 3.2627), and the 30-day run used its full 40-epoch allowance without early stopping firing.
6. **Fine-tuning** the hybrid at a 10x lower learning rate improved test RMSE from 3.2278 to 3.2213 (about 0.20% better), while test MAE moved from 1.8237 to 1.8253 (very slightly worse): the improvement is real but small and metric-dependent, matching the pattern from the earlier run.
7. **Honest headline for Task 2:** every model, including the naive baseline, has a test MAPE of about 1.16%. One-day-ahead price prediction barely improves on "tomorrow equals today"; this is the anchor for the reliability and ethics discussion in Task 2.
8. **Reproducibility / rebuild check:** the final student notebook does not include a separate "rebuild and retrain" reproducibility cell (that check existed only in the earlier, now-archived professional notebook). If a reproducibility claim is needed in the report, it must either cite the multi-seed evidence below or be re-obtained by rerunning this notebook a second time.

## Seed robustness (supplementary, not yet reconfirmed against the final notebook)

The multi-seed check below comes from `_code/archive/COM7019_25199053_seed_robustness.ipynb`, a companion notebook that has **not** been through the readability/diagram simplification pass (D-009 to D-012) and has **not** been rerun since the file reorganisation. It is kept as supplementary evidence of run-to-run spread, not as a like-for-like match to the main table above (its RMSE values, e.g. LSTM_64x32_d02 at 3.2427 for seed 25199053, are close to but not identical to this ledger's numbers, consistent with ordinary GPU/library-version noise).

Run finished: 2026-07-09 16:45:08 UTC | Environment: Python 3.12.13, TF 2.21.0, CPU | Seeds: 25199053, 7, 2026

| Model | Mean test RMSE | RMSE spread | vs baseline % | Beats baseline (N/3) | Mean direction acc % |
|---|---:|---|---:|---:|---:|
| CNN32k3_LSTM64 | 3.2347 | 3.2235 - 3.2526 | -0.10 | 2 | 51.4 |
| LSTM_64x32_d02 | 3.2449 | 3.2327 - 3.2593 | +0.21 | 1 | 51.0 |
| GRU_64_d02 | 3.2520 | 3.2398 - 3.2645 | +0.43 | 0 | 51.8 |

Persistence baseline (that run): RMSE 3.2380 | Up-day base rate (always guess up): 52.9%

**Decision (11 Jul 2026):** cite these numbers as-is in the report, with an explicit footnote/caveat that they came from the pre-simplification code version (same logic and results-affecting behaviour, only comments and structure differ per D-009 to D-012; no numeric changes were made in that pass).

## Figures (frozen artifacts, final notebook)

| Figure | File | Content |
|---|---|---|
| 1 | `_code/outputs/figures/fig1_price_history.png` | Full price history with train/val/test colour-coded |
| 2 | `_code/outputs/figures/fig2_loss_curves.png` | Training vs validation loss for the four Stage-A models + hybrid |
| 3 | `_code/outputs/figures/fig3_predictions_vs_actual.png` | Next-day predictions vs actual, final test period |
| Architecture x5 | `_code/outputs/figures/arch_*.png` | Hand-drawn layer diagrams for LSTM_64, LSTM_64x32, GRU_64, GRU_64x32, and the CNN-LSTM hybrid (D-012) |
| 4 (supplementary) | `_code/archive/outputs_seeds/figures/fig4_seed_robustness.png` | Test RMSE per seed for the three headline models vs baseline (pre-simplification run) |

## Supporting artifacts

- `_code/COM7019_25199053.ipynb` (final notebook, executed with outputs; source of every number above)
- `_code/outputs/tables/results_all_runs.csv` (machine-readable results)
- `_code/outputs/models/model_summaries.txt` (layer tables for the appendix)
- `_code/archive/COM7019_25199053_experiments.ipynb` (earlier professional-style version, superseded, kept for reference)
- `_code/archive/COM7019_25199053_seed_robustness.ipynb` (seed-robustness companion, not yet rerun against the final code)
- `_code/archive/outputs/`, `_code/archive/outputs_seeds/` (artifacts from the superseded runs)

## Open items raised by this refresh

1. Exact Python/TensorFlow/NumPy/pandas versions for this run are not captured anywhere (the version-print cell was intentionally removed). Not required unless a marker asks; leave as a known gap.
2. Seed-robustness notebook: resolved 11 Jul 2026, cite as-is with a code-version caveat (see above), no rerun needed.
3. Screenshot evidence (brief requirement): resolved 11 Jul 2026. Student captured 21 screenshots from the Colab session; all were verified one by one against the executed notebook's stored outputs (every visible number matches this ledger). Files live in `_code/screenshots/`, renamed to match notebook steps:

| File | Notebook step | Content |
|---|---|---|
| `01_step02_load_data.png` | STEP 2 | Row/column counts + df.head() |
| `02_step03_data_checks.png` | STEP 3 | Missing values, duplicates, sorted-dates check, price range |
| `03_step04_split_fig1.png` | STEP 4 | Split boundaries + Figure 1 (price history) |
| `04_step05_windowing.png` | STEP 5 | Sample count, X shape, window sanity check |
| `05_step07_baseline.png` | STEP 7 | Persistence baseline RMSE/MAE/MAPE |
| `06_step09_lstm64_summary_arch_training.png` | STEP 9 | LSTM_64_d02: summary + architecture diagram + training |
| `07_step10_lstm64x32_arch_training.png` | STEP 10 | LSTM_64x32_d02: architecture + training |
| `08_step11_gru64_arch_training.png` | STEP 11 | GRU_64_d02: architecture + training |
| `09_step12_gru64x32_arch_training.png` | STEP 12 | GRU_64x32_d02: architecture + training |
| `10_step13_best_model_pick.png` | STEP 13 | Validation RMSE comparison, best LSTM/GRU pick |
| `11_step14_lstm_no_dropout.png` | STEP 14 | LSTM dropout ablation |
| `12_step15_gru_no_dropout.png` | STEP 15 | GRU dropout ablation |
| `13_step16_window30.png` | STEP 16 | 30-day window experiment |
| `14_step17_low_learning_rate.png` | STEP 17 | Learning rate 1e-4 experiment |
| `15_step18_cnn_lstm_summary_arch_training.png` | STEP 18 | CNN-LSTM hybrid: summary + architecture + training |
| `16_step19_finetune.png` | STEP 19 | Fine-tuning result |
| `17_step20_final_results_table.png` | STEP 20 | Full FINAL RESULTS table (key evidence) |
| `18_step21_fig2_loss_curves.png` | STEP 21 | Figure 2, loss curves |
| `19_step22_fig3_predictions.png` | STEP 22 | Figure 3, predictions vs actual |
| `20_step23_summaries_saved.png` | STEP 23 | Model summaries saved + best-model recap |
| `21_step24_conclusion.png` | STEP 24 | Computed conclusion cell |

Known minor gap: STEP 6's small output (per-split sample counts after windowing: train 6,876 / val 1,486 / test 1,487) has no dedicated screenshot; the same numbers are stored in the executed notebook and quoted in this ledger, and the split row boundaries are visible in `03_step04_split_fig1.png`. Capture one more shot only if the marker's checklist demands literally every cell.
