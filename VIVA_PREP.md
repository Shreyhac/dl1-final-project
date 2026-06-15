# Viva Cheat-Sheet — DL-I Final Project

> **Why this matters:** the viva is ~11/35 marks (30%). One **random** member is called; their
> score becomes the **whole group's** viva score. Everyone must be able to explain **any** part.
> Read this, then explain each answer out loud in your own words.

---

## Part A — Project: Task 1 (Time-Series Forecasting)

**Q: What's the problem and dataset?**
Forecast air temperature **12 hours ahead** from the **Jena Climate** dataset (Max Planck
Institute; 2009–2016, recorded every 10 min, we resampled to **hourly**). It has strong
**daily seasonality** (day/night cycle).

**Q: What models did you compare, and the result?**
A **persistence/moving-average baseline**, a **vanilla RNN**, and an **LSTM**. Test RMSE:
**LSTM 2.81 °C (best) < RNN 2.86 < moving-avg 3.73 < persistence 5.41**. The LSTM cuts error
~48% vs persistence.

**Q: Why did you split the data chronologically instead of shuffling?**
Time series have temporal order. Random shuffling would let the model train on *future* points
and predict *past* ones → **data leakage** and over-optimistic scores. We split 70/15/15 in time.

**Q: Why normalize using only the training statistics?**
Using val/test mean & std would leak information about unseen data into preprocessing. We fit the
scaler on **train only**, then apply it to val/test. Metrics are reported in °C after
**de-normalising**.

**Q: What is "persistence" and why is it a strong baseline at 1 hour but weak at 12 hours?**
Persistence predicts "next value = last observed value." At a **1-hour** horizon temperature
barely changes, so it's very hard to beat. At **12 hours** the last value is from roughly the
opposite point of the day/night cycle, so persistence is badly out of phase → the learned models
win clearly. (This is *why* we chose the 12 h horizon.)

**Q: RNN vs LSTM — what's the difference?**
A vanilla RNN updates a single hidden state each step and suffers **vanishing/exploding
gradients** over long sequences. An **LSTM** adds a **cell state** and three **gates**
(forget / input / output) that control what to keep, add, and expose — so it retains long-range
information far better. That's why our LSTM kept improving at longer lookbacks while the RNN
plateaued.

**Q: Explain the lookback-window ablation and what it showed.**
We retrained both models for lookbacks **[6, 12, 24, 48, 96] hours**. RMSE is **worst at 6 h**
(can't see the daily cycle), drops sharply by ~24 h (one full day), then **flattens**. Beyond
that the **LSTM keeps improving to 96 h** (2.74) while the **RNN plateaus / slightly worsens** —
consistent with the LSTM's better long-range memory. Takeaway: ~24–48 h is the practical sweet
spot; more context helps only the LSTM.

**Q: What loss and optimizer did you use?**
**MSE** loss, **Adam** optimizer (lr 1e-3). MSE penalises large errors quadratically — good for
regression. We report **MAE** and **RMSE** (RMSE is more sensitive to big misses).

---

## Part B — Project: Task 2 (CNN Filter Visualization)

**Q: What's the task and final accuracy?**
Train a small **VGG-style CNN** on **CIFAR-10** (10 classes, 32×32 RGB), then visualise what it
learns. Final **test accuracy ≈ 82%** (12 epochs, ~2.3M params).

**Q: Describe the architecture.**
Four conv layers (3→32→64→128→128, all 3×3, padding 1), two 2×2 max-pools, then FC 256 → 10.
ReLU activations. Light augmentation (random crop + horizontal flip) + channel normalisation.

**Q: How do you compute a conv layer's output size?**
`out = (W − K + 2P)/S + 1`. E.g. 32×32 input, 3×3 kernel, padding 1, stride 1 → stays 32×32.
A 2×2 max-pool with stride 2 halves it to 16×16. Padding 1 with a 3×3 kernel keeps spatial size.

**Q: What does the confusion matrix tell you?**
Strong diagonal (mostly correct). The biggest confusions are **cat↔dog** and **car↔truck** —
visually/semantically similar classes. **cat (61%)** and **bird (70%)** are hardest;
**ship (92%), plane (90%), truck (89%)** easiest. This is the expected CIFAR-10 behaviour.

**Q: What do the first-layer (conv1) filters look like, and why?**
Small **oriented edges, colour-opponent blobs, simple gradients** — low-level detectors,
because conv1 operates directly on raw RGB pixels. (Same as the classic Gabor-like filters in
AlexNet's first layer.)

**Q: How did you extract feature maps, and what's the early-vs-late difference?**
We registered **forward hooks** on conv1 (early) and conv4 (late) to capture activations for one
image. **Early maps** are sharp, high-resolution, and clearly retain the object's **outline**
(they answer *where* edges/colours are). **Late maps** are low-resolution (16×16), **sparse**,
and **abstract** — most channels are silent; a few fire for high-level parts (they answer
*what*). This demonstrates **increasing abstraction with depth**.

**Q: Why does depth produce abstraction?**
Each layer composes features from the previous one and (via pooling) grows the **receptive
field**: pixels → edges/colours → textures/parts → class-discriminative concepts. The network
trades **spatial detail** for **semantic meaning** — which is exactly what the classifier needs.

---

## Part C — Theory (Weeks 1–7, likely cross-questions)

- **Why activations / non-linearity?** Without them, stacked linear layers collapse to one linear
  map (can't solve XOR). ReLU adds non-linearity and avoids saturation for positive inputs.
- **Vanishing/exploding gradients?** In deep/recurrent nets, repeated multiplication of small
  (<1) or large (>1) Jacobians shrinks/explodes gradients. Fixes: ReLU, good init (Xavier/He),
  batch norm, residual connections, and **LSTM/GRU gates** for sequences.
- **Backprop in one line?** Chain rule applied backward through the computational graph to get
  ∂loss/∂weights; the optimizer (SGD/Adam) then steps against the gradient.
- **Overfitting & fixes?** Train ≫ test performance. Fixes: more data / augmentation,
  **dropout**, **L2 (weight decay)**, **early stopping**, **batch norm**. (We used augmentation
  + a small model + a modest epoch count.)
- **Why CNNs over MLPs for images?** Parameter sharing + local receptive fields → translation
  equivariance and far fewer parameters than a fully-connected net on pixels.
- **What is BPTT?** Backprop Through Time — unroll the RNN over the sequence and backprop through
  every time step; long sequences cause the vanishing-gradient problem LSTMs mitigate.
- **Adam vs SGD?** Adam adapts per-parameter learning rates using running estimates of the
  gradient mean and variance (momentum + RMSProp) → faster, more robust convergence.
- **Max-pooling — why?** Downsamples, adds small translation invariance, grows the receptive
  field, and cuts computation.

---

## Part D — Quick numbers to memorise
| | |
|---|---|
| TS horizon / lookback default | 12 h ahead / 24 h window |
| Task 1 best (LSTM) RMSE / MAE | **2.81 / 2.19 °C** |
| Task 1 persistence RMSE | 5.41 °C (beaten by ~48%) |
| Ablation sweet spot | ~24 h; LSTM best at 96 h (2.74) |
| CNN test accuracy / params | **~82% / 2.3M** |
| Hardest / easiest class | cat 61% / ship 92% |
| Loss / optimizer (both) | MSE & CrossEntropy / Adam 1e-3 |

**Golden rule:** if unsure, explain the *intuition* and tie it to a figure in your notebook.
Don't bluff exact numbers — point to the cell that prints them.
