# DL-I Final Project — Weather Forecasting + CNN Filter Visualization

## What's here
| File | What it is |
|---|---|
| `task1_timeseries.ipynb` | **Task 1 deliverable** — Jena weather, 12 h-ahead forecast (baseline / RNN / LSTM) + lookback ablation. Clean, run on Colab. |
| `task2_cnn_viz.ipynb` | **Task 2 deliverable** — CIFAR-10 CNN, confusion matrix, filter & feature-map visualization. Clean, run on Colab. |
| `task1_full.ipynb`, `task2_full.ipynb` | Same notebooks **already executed with outputs/figures** — open these to read results without re-running. |
| `report/REPORT.md` | 5–6 page report (export to PDF). Figures in `report/figures/`. |
| `slides/SLIDES.md` | ≤15-slide deck outline with the real numbers. |
| `evaluation_guide.pdf` | Course evaluation guide. |

## Headline results
- **Task 1:** LSTM RMSE **2.81 °C** (best) > RNN 2.86 > moving-avg 3.73 > persistence 5.41 (LSTM −48% vs baseline).
- **Task 2:** **82.2%** test accuracy on CIFAR-10.

## How to run on Google Colab
1. Go to <https://colab.research.google.com> → **File → Upload notebook** → pick the `.ipynb`.
2. **Runtime → Change runtime type → T4 GPU** → Save.
3. **Runtime → Run all.** Datasets download automatically. ~2–4 min each.

## TODO before submission
- Run both clean notebooks on Colab GPU (saved outputs required by the guide).
- Add member names / roll numbers + contributions table in the report.
