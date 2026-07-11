# COM7019: Artificial Intelligence and Neural Networks

Arden University MSc Data Science, student ID **25199053**.

## Contents

| Folder | What it is |
|---|---|
| `assignment/` | Stock price prediction portfolio (Task 1 + Task 2) |
| `assignment/notebook/` | Executed Colab notebook (`COM7019_25199053.ipynb`) |
| `assignment/outputs/` | Frozen figures, tables, and model summaries from the final run |
| `assignment/screenshots/` | Colab screenshot evidence (21 images, mapped to notebook steps) |
| `assignment/DRAFT_REPORT.md` | Plain-text report draft (500-word Task 1 + 900-word Task 2) |
| `assignment/RESULTS_LEDGER.md` | Frozen numbers ledger (source of truth for all quoted metrics) |
| `workshops/` | Module workshop notebooks (backpropagation, gradient descent, optimisation) |

## Assignment at a glance

- **Task 1:** LSTM vs GRU comparison on `Stock_Price_Data_[3921].csv`, plus a CNN-LSTM hybrid extension.
- **Task 2:** Critical evaluation of results, reliability, ethics, and reflection.
- **Best model:** CNN32k3_LSTM64_finetuned (test RMSE 3.2213 vs persistence baseline 3.2380).
- **Verdict:** GRU is the more efficient of the two required architectures; only the hybrid beats the baseline.

Dataset is supplied by the university via iLearn and is not stored in this repo.
