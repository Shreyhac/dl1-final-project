# Full Slide Deck Content (copy-paste into Google Slides)

> 14 slides (≤15 limit). Each slide: **Title**, bullet text, the figure to insert (from
> `report/figures/`), and the assigned presenter. Keep bullets short on the slide — say the rest.

---

### Slide 1 — Title
**Weather Forecasting (RNN/LSTM) & CNN Filter Visualization**
- Deep Learning I: Neural Networks · Semester 4 · Final Project
- Members: Shreyansh Arora (24BCS10252), Bhavya Patel (24BCS10420), Rudra Longaonkar (24BCS10312), Mohammed Rehan (24BCS10620), Avishkar Chavan (24BCS10065), Milap Kothari (24BCS10004), Soham Ambore (24BCS10258), Shubh Shukla (24BCS10093)
- *Presenter: Shreyansh*

### Slide 2 — Agenda
**Two tasks, one project**
- **Task 1 (Applied):** Forecast temperature 12 h ahead — baseline vs RNN vs LSTM
- **Task 2 (Experimental):** Visualize what a CNN learns on CIFAR-10
- Both: proper evaluation (curves, error metrics, confusion matrix) + analysis
- *Presenter: Shreyansh*

---

## TASK 1 — Time-Series Forecasting

### Slide 3 — Problem & Data
**Forecasting Jena weather, 12 hours ahead**
- Jena Climate dataset (2009–2016), 10-min readings → resampled hourly (~70k points)
- Target: air temperature `T (degC)`; strong **daily (day/night) seasonality**
- Insert: `t1_series.png`
- *Presenter: Bhavya*

### Slide 4 — Setup (no data leakage)
**How we prepared the data**
- **Chronological** 70/15/15 split (never shuffle time → no peeking at the future)
- Standardised using **train-only** mean/std; metrics reported back in °C
- Sliding windows: last `LOOKBACK` hours → value 12 h later
- *Presenter: Bhavya*

### Slide 5 — Models
**Three approaches, compared fairly**
- **Baseline:** persistence (next = last) & moving average — no training
- **Vanilla RNN** (`nn.RNN`, hidden 64) and **LSTM** (`nn.LSTM`, hidden 64)
- RNN vs LSTM identical except the cell → clean comparison; Adam + MSE, 12 epochs
- *Presenter: Mohammed Rehan*

### Slide 6 — Results
**LSTM wins — beats persistence by ~48%**
- RMSE (°C): **LSTM 2.81 < RNN 2.86 < moving-avg 3.73 < persistence 5.41**
- Persistence is a full half-day **out of phase** at a 12 h horizon
- Insert: `t1_forecast.png` (and optionally `t1_training_curves.png`)
- *Presenter: Mohammed Rehan*

### Slide 7 — Ablation: Lookback Window
**How much past context helps?**
- Swept lookback = 6/12/24/48/96 h
- 6 h too short (misses daily cycle) → error drops sharply by ~24 h → then flattens
- **LSTM keeps improving to 96 h** (gates retain long context); RNN plateaus
- Insert: `t1_ablation_lookback.png`
- *Presenter: Rudra*

---

## TASK 2 — CNN Filter Visualization

### Slide 8 — Problem & Model
**A small VGG-style CNN on CIFAR-10**
- 10 classes, 32×32 RGB images; 4 conv layers (32→64→128→128) + FC head; ~2.3M params
- Light augmentation (random crop + flip) + channel normalisation
- Insert: `t2_samples.png`
- *Presenter: Avishkar*

### Slide 9 — Training & Accuracy
**~82% test accuracy in 12 epochs**
- Healthy curves, small train/test gap (not overfitting)
- Insert: `t2_curves.png`
- *Presenter: Avishkar*

### Slide 10 — Confusion Matrix
**Where it gets confused**
- Errors cluster between similar classes: **cat↔dog**, **car↔truck**
- Hardest: cat 61%, bird 70% · Easiest: ship 92%, plane 90%, truck 89%
- Insert: `t2_confusion.png`
- *Presenter: Milap*

### Slide 11 — First-Layer Filters
**What conv1 learns**
- 32 filters (3×3 RGB) → **oriented edges, colour-opponent blobs** — low-level detectors
- They operate directly on raw pixels
- Insert: `t2_filters.png`
- *Presenter: Soham*

### Slide 12 — Feature Maps: Early vs Late
**Representations get abstract with depth**
- **Early (conv1):** sharp, high-res, retains the object outline → answers *where*
- **Late (conv4):** coarse, **sparse**, abstract → answers *what*
- Insert: `t2_feat_early.png` and `t2_feat_late.png` side by side
- *Presenter: Soham*

### Slide 13 — Why Depth = Abstraction
**The hierarchy**
- pixels → edges/colours → textures/parts → class-discriminative concepts
- Growing receptive field + pooling trade spatial detail for semantic meaning
- This is *why* deep conv stacks classify well (and why cat↔dog still confuse)
- *Presenter: Milap*

---

### Slide 14 — Conclusions & Q&A
**Takeaways**
- TS: LSTM > RNN ≫ baselines at 12 h; lookback sweet-spot ≈ daily cycle, LSTM gains from longer context
- CNN: 82% accuracy; visualizations show clear depth→abstraction
- Limitations / future: multi-step + exogenous features (TS); deeper net / Grad-CAM (CNN)
- **Thank you — Questions?**
- *Presenter: Shubh*

---

**Presenter summary (every member presents ≥1):** Shreyansh (1,2) · Bhavya (3,4) ·
Mohammed Rehan (5,6) · Rudra (7) · Avishkar (8,9) · Milap (10,13) · Soham (11,12) · Shubh (14).
