# COM7019 Portfolio Report
**Student ID:** 25199053

---

## Task 1: Stock Market Prediction using LSTM and GRU

I was asked to act as an AI Analyst and build a deep learning solution for short-term stock trend prediction on the supplied CSV. Task 1 designs, trains, and compares LSTM and GRU as required, adds a CNN-LSTM hybrid as an extension, and judges every result against a naive persistence baseline (tomorrow's price equals today's).

**Data and preparation.** The file holds 9,909 trading days from December 1980 to April 2020. Loading confirmed 9,909 rows and six columns (Open, High, Low, Close, Adj Close, Volume).

> **[Screenshot 1]** STEP 2: row/column counts and `df.head()`.

Checks found no missing values, no duplicate dates, and prices already sorted oldest to newest (range $0.16 to $327.20).

> **[Screenshot 2]** STEP 3: data-quality checks.

I predict Adj Close because it reflects investor return after dividends. The split is chronological 70/15/15 (train to June 2008, validation to May 2014, test to April 2020) so no future information leaks into training. Figure 1 shows how far apart the price levels are across eras.

> **[Screenshot 3]** STEP 4: split boundaries and Figure 1 (price history).

Each model sees 60-day sliding windows. Because training prices sit under $25 while test prices exceed $300, a train-fitted MinMax scaler would push test inputs out of range. I normalise each window against its own last price instead, so a 1985 window and a 2019 window look alike to the network. After 9,849 windows were built, the persistence baseline scored test RMSE 3.238, MAE 1.827, MAPE 1.159%.

> **[Screenshot 4]** STEP 5: window shape and sanity check.
> **[Screenshot 5]** STEP 7: persistence baseline metrics.

**Models and training.** Four base models were trained under one protocol: LSTM 64, LSTM 64+32, GRU 64, GRU 64+32, all with dropout 0.2, MSE loss, Adam (lr 0.001), batch 64, early stopping (patience 5), seed 25199053. Recurrent gates keep sigmoid/tanh; Conv1D uses ReLU; the output Dense layer is linear. Architecture diagrams show layer shapes and the recurrent loop on LSTM/GRU boxes.

> **[Screenshot 6]** STEP 9: LSTM_64 summary, architecture, training (RMSE 3.2486).
> **[Screenshot 7]** STEP 10: LSTM_64x32 architecture and training (RMSE 3.2427).
> **[Screenshot 8]** STEP 11: GRU_64 architecture and training (RMSE 3.2519).
> **[Screenshot 9]** STEP 12: GRU_64x32 architecture and training (RMSE 3.2422).

Validation RMSE picked the best LSTM (LSTM_64x32_d02, 0.8738) and best GRU (GRU_64_d02, 0.8734), a 0.04% gap.

> **[Screenshot 10]** STEP 13: validation comparison and best-model selection.

**Experiments.** Dropout ablation was mixed: no dropout helped the LSTM slightly (3.2403 vs 3.2427) but hurt the GRU (3.2627 vs 3.2519). A 30-day window was worse than 60 days (3.2682 vs 3.2627). Lowering the learning rate to 0.0001 gave the best recurrent-only test RMSE (3.2385). The CNN-LSTM hybrid (Conv1D 32, k=3, MaxPool, LSTM 64) reached 3.2278; fine-tuning at lr 0.0001 improved this to 3.2213.

> **[Screenshot 11]** STEP 14: LSTM dropout ablation.
> **[Screenshot 12]** STEP 15: GRU dropout ablation.
> **[Screenshot 13]** STEP 16: 30-day window experiment.
> **[Screenshot 14]** STEP 17: lower learning-rate experiment.
> **[Screenshot 15]** STEP 18: CNN-LSTM summary, architecture, training (RMSE 3.2278).
> **[Screenshot 16]** STEP 19: fine-tuning result (RMSE 3.2213).

**Results and verdict.** Table 1 ranks all runs. Only the fine-tuned hybrid beat persistence (0.52% better on RMSE). Plain LSTM and GRU did not. LSTM and GRU are tied on accuracy; GRU wins on efficiency (~24% fewer parameters, faster training). I therefore recommend GRU for this scenario. Loss curves flatten early with no train/validation divergence. Figure 3 shows predictions tracking the final test year closely, including March 2020, but the metrics confirm the models sit near the baseline.

> **[Screenshot 17]** STEP 20: full results table.
> **[Screenshot 18]** STEP 21: Figure 2 (loss curves).
> **[Screenshot 19]** STEP 22: Figure 3 (predictions vs actual).
> **[Screenshot 20]** STEP 23: saved model summaries and best-model recap.

---

## Task 2: Critical Evaluation Report

**What the numbers actually mean.** At first glance the hybrid looks successful: test RMSE 3.2213 beats persistence at 3.2380. Looked at honestly, that is a 0.52% margin, erased by a different seed in one of three supplementary runs. Every model, baseline included, sits at MAPE ~1.16%. Direction accuracy averages 51-52%, below the 52.9% you get by always guessing "up". The correct inference is not that deep learning predicts this stock; it is that one-day-ahead price prediction on a single series is almost entirely a persistence problem, and the networks mostly learn a slightly smoothed version of "tomorrow equals today".

**Why that happened, and what it taught me about the methods.** Before running anything I assumed the recurrent layers would find momentum or mean-reversion patterns in 60 days of history. They did not, or at least not enough to beat a one-line baseline. That was a useful shock: it forced me to stop reading the loss curve as proof of skill and start reading it as proof of convergence. The curves dropped quickly and then flatlined with validation loss below training loss, which told me the models were stable, not that they were clever. Per-window normalisation was the design choice that changed my thinking most. Fitting a global scaler on training data felt "correct" statistically, but it would have handed the model test-era inputs it had never seen in dollar terms. Normalising inside each window reframed the task as "what happened recently, relative to yesterday?" rather than "what is the absolute price level?" That matches how a trader might glance at a chart, and it is probably why the models could generalise across four decades at all. The dropout ablation taught a different lesson: a technique that helps one architecture can hurt another on the same data. I had treated dropout as a universal good; the GRU without it was clearly worse, the LSTM marginally better without it. That inconsistency is why I now treat each regulariser as something to test, not something to apply by default. The CNN-LSTM result gave me an architectural intuition I did not have at the start: convolution over three-day spans picks up very local moves (a small bounce, a three-day slide) and hands the LSTM a shorter, pre-filtered sequence. The gain is tiny (0.31% before fine-tuning) but repeatable on average across seeds, which suggests the front-end is doing real work, not just adding parameters.

**Reliability and limitations.** The process side is sound: chronological split, validation-only selection, fixed protocol, seeded notebook, metrics against persistence. The weaknesses are equally clear. One test period ending in a pandemic crash is a hard, unrepeatable stress test. Three seeds bound noise roughly, not tightly. Volume sits unused in the CSV. Walk-forward validation would have been stronger. These limits do not break the LSTM-vs-GRU comparison, which was always the brief's core question, but they cap how far any absolute RMSE should travel outside this report.

**Broader implications.** A model whose best case beats persistence by half a percent should not drive trades without human oversight. Figure 3 is the ethical trap: it looks accurate enough to trust, yet my own direction-accuracy check says it is not. Deploying this would risk overtrust, automation bias, and diffused accountability if losses were blamed on "the model" rather than the decision to use it [S-004, S-005, S-006]. Mitigations are straightforward and this report already applies the first one: always show the naive baseline, state the margin honestly, keep a human in the loop, and treat research output as evidence for further validation, not as advice.

**Reflection.** Going in, I thought the hard part would be building the networks. Keras and Colab made that almost easy; the hard part was staying honest when the numbers were disappointing. The single habit that saved this report was putting persistence in before the first model run. Once that baseline existed, every result had a reference point, and I could not pretend a 3.25 RMSE was impressive just because the plot looked good. The gap between Figure 3 and the direction-accuracy table was the biggest personal lesson: my eye is drawn to curves that track the price, but the business question is whether tomorrow moves up or down, and on that question the models failed. I also learned to respect seed sensitivity. Seeing the hybrid beat the baseline in two of three seeds but not three of three stopped me from writing a headline the data did not fully support. If I repeated the project I would predict returns or direction explicitly, add volume, and build walk-forward folds from day one, because those choices match the question users actually ask. What I take forward is not scepticism about neural networks in general, but a concrete workflow: define the naive answer first, measure against it, distrust beautiful plots, and treat a 0.5% edge as real but fragile until repeated evidence says otherwise. That shift, from "can I build a predictor?" to "is there signal here at all?", is the main thing this assignment changed in how I would approach any applied ML problem.

> **[Screenshot 21]** STEP 24: computed conclusion from the notebook.

---

## References

To be completed from `SOURCE_LEDGER.md` at final assembly (Harvard format).

## Appendix: Code

`COM7019_25199053.ipynb` (executed notebook; 1,600 word-equivalence allowance per brief).
