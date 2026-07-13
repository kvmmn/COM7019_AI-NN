# Tutor feedback implementation log (Jul 2026)

| # | Tutor request | Implementation | Files touched |
|---|---------------|--------------|---------------|
| 1 | Professional GitHub repo (Docker, UV, requirements, reusable) | `pyproject.toml`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `src/com7019/`, `README.md` | Project root |
| 2 | Explain unpredictable nature of financial data | Expanded STEP 3 prose + report Section 2 | Notebook, `DRAFT_REPORT.md` |
| 3 | Richer EDA with candlestick charts | STEP 3b–3d: mplfinance candlesticks, returns histogram, correlation heatmap | Notebook |
| 4 | Standard windows 1W(5), 1M(21), 1Y(252) trading days | Primary = 1M; ablation compares all three | Notebook, `config.py` |
| 5 | Readable model names | `build_model_name()` helper; all runs renamed | Notebook, `config.py` |
| 6 | Graphical model comparison | Figure 4 bar charts + window comparison plot | Notebook, `visualization.py` |
| 7 | Business / advisory reflection in Task 2 | New subsection in `DRAFT_REPORT.md` Section 5 | `_brain/DRAFT_REPORT.md` |
| 8 | Regularisation matrix (optional) | Heatmap of family × regularisation vs test RMSE | Notebook STEP 20b |

## Re-run required

Changing the primary window from 60 → 21 trading days **invalidates** the frozen `RESULTS_LEDGER.md`. After Colab re-run:

1. Replace `RESULTS_LEDGER.md` wholesale from new `outputs/tables/results_all_runs.csv`
2. Re-capture screenshots
3. Update every number in `DRAFT_REPORT.md`
