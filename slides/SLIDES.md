# Slide Deck Outline (≤15 slides)

> Max 15 slides. **Every member must present at least one slide** — note the presenter on each.
> Build in Google Slides/PowerPoint; paste figures exported from the notebooks.

1. **Title** — "Weather Forecasting (RNN/LSTM) + CNN Filter Visualization", DL-I Sem 4. Members: Shreyansh Arora (24BCS10252), Bhavya Patel (24BCS10420), Rudra (24BCS10312), Mohammed Rehan (24BCS10620), Avishkar Chavan (24BCS10065), Milap Kothari (24BCS10004), Soham Ambore (24BCS10258), Shubh Shukla (24BCS10093). _(Presenter: Shreyansh)_
2. **Agenda & two tasks** — applied (forecasting) + experimental (CNN viz). _(__)_

### Task 1 — Time Series Forecasting
3. **Problem & dataset** — Jena weather, forecast temp **12 h ahead**; `t1_series.png` (daily cycle). _(__)_
4. **Setup** — chronological split, train-only normalisation, windowing diagram. _(__)_
5. **Models** — baseline vs vanilla RNN vs LSTM (one-line each + tiny diagram). _(__)_
6. **Results** — **LSTM best: RMSE 2.81 °C vs persistence 5.41 (−48%)**; `t1_forecast.png` (persistence is 12 h out of phase). _(__)_
7. **Ablation: lookback window** — `t1_ablation_lookback.png`; sweet spot ≈ 24 h, LSTM keeps gaining to 96 h. _(__)_

### Task 2 — CNN Filter Visualization
8. **Problem & model** — CIFAR-10, small VGG-style CNN (2.3M params). _(__)_
9. **Training & accuracy** — `t2_curves.png`; **82.2% test accuracy**. _(__)_
10. **Confusion matrix** — `t2_confusion.png`; errors cluster at cat↔dog, car↔truck (cat worst, 61%). _(__)_
11. **First-layer filters** — `t2_filters.png`, edge/colour detectors. _(__)_
12. **Feature maps: early vs late** — `t2_feat_early.png` vs `t2_feat_late.png` side by side. _(__)_
13. **Abstraction with depth** — the pixels→edges→parts→semantics story. _(__)_

### Wrap-up
14. **Conclusions & limitations** — what we learned, what's next. _(__)_
15. **Q&A / Thank you** — be ready for viva (anyone can be called). _(__)_

---
**Presentation tips (from the guide):** 12 min goes fast — rehearse twice; lead with insight,
not code; every member must be able to explain *any* slide for the individual viva.
