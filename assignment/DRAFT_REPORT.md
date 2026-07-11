# COM7019 Report Draft (plain text, working copy)

Status: in progress. This file is the plain-text draft body. No formatting, figures, or references are applied here; that happens only at final assembly per the agreed workflow. Word counts below are rough (prose only, excludes headings) and checked against the budget in `ASSIGNMENT_CONTROL.md`.

Every numeric claim in this draft must trace to `RESULTS_LEDGER.md`. If you spot a number here that is not in the ledger, flag it, it should not be there.

Screenshot files live in `_code/screenshots/`. Markers below show where each image goes in the final document.

---

## Section 1: Introduction and problem framing (target ~200 words)

I am asked to act as an AI Analyst at a financial technology company. My line manager wants a deep learning investigation into whether historical stock price data can support short-term trend prediction, so the business can judge whether a neural network approach is worth pursuing further.

Task 1 of this report designs, builds, and compares the two recurrent neural network architectures required by the brief, LSTM and GRU, on the single dataset supplied: daily Open, High, Low, Close, Adjusted Close, and Volume prices for one stock, covering 1980 to 2020. As a self-directed extension, I add a third model, a CNN-LSTM hybrid, which pairs a convolutional layer with an LSTM to test whether picking up short local patterns first helps the recurrent layer that follows. All three models are trained under an identical protocol (same data split, same input windows, same loss function, same random seed) so that the comparison between them is fair. Every result is also checked against a naive persistence baseline, which simply predicts that tomorrow's price equals today's price. On daily financial data this baseline is a genuinely difficult benchmark to beat, and it anchors every claim made later in this report.

Task 2 then steps back from the numbers to critically evaluate what the results actually mean: how reliable the models are, what their limitations are, and what the ethical implications are of using a model like this to inform real trading or investment decisions.

*(word count: ~205)*

---

## Section 2: Data and methodology (target ~400 words)

The dataset contains 9,909 trading days for a single stock, from 12 December 1980 to 1 April 2020, with Open, High, Low, Close, Adjusted Close, and Volume columns. Before modelling I checked it: there are no missing values, no duplicate dates, and the rows are already in chronological order. I predict Adjusted Close, because it corrects for dividends and so reflects what an investor actually earned; plain Close would carry artificial jumps that have nothing to do with market movement.

> **[Screenshot 1]** `_code/screenshots/01_step02_load_data.png`  
> STEP 2: data load: 9,909 rows, six columns, `df.head()`.

> **[Screenshot 2]** `_code/screenshots/02_step03_data_checks.png`  
> STEP 3: data quality: missing values, duplicate dates, sorted order, price range.

Because this is a forecasting task, the data must be split by time, not at random. Shuffling would let the model train on days that come after its test days, which is a form of data leakage and would make the results look better than they could ever be in real use. I therefore use a chronological 70/15/15 split: training up to June 2008, validation up to May 2014, and testing up to April 2020. All model selection decisions use the validation set only; the test set is touched once per model, at final evaluation.

> **[Screenshot 3]** `_code/screenshots/03_step04_split_fig1.png`  
> STEP 4: split boundaries and Figure 1 (price history with train/val/test bands).

The models receive sliding windows of 60 past days and predict the next day. Scaling needed care: the price grows from under $0.16 to over $327 across the series, so a MinMax scaler fitted on training data (the only leakage-free option) would map every test price outside its fitted range. Instead I normalise each window against its own last price, so every input becomes a relative change. A window from 1985 and one from 2019 then look alike to the model. Predictions are converted back to dollars before scoring, so all metrics are in USD.

> **[Screenshot 4]** `_code/screenshots/04_step05_windowing.png`  
> STEP 5: windowing: 9,849 samples, shape (9849, 60, 1), sanity check.

I compare three architectures under an identical protocol: LSTM and GRU as required by the brief, and a CNN-LSTM hybrid as my extension, where a Conv1D layer picks up short local patterns before the LSTM reads the longer trend. The recurrent layers keep their built-in gate activations (sigmoid gates, tanh cell state), the convolution uses ReLU, and every model ends in a single linear Dense unit, because a regression target that can be positive or negative must not be squashed by an output activation. All models train with MSE loss, the Adam optimiser at learning rate 0.001, batch size 64, up to 40 epochs with early stopping (patience 5, best weights restored), and a fixed seed of 25199053, my student ID, so the run is reproducible. Performance is measured by RMSE, MAE, and MAPE against a naive persistence baseline that predicts tomorrow's price equals today's.

> **[Screenshot 5]** `_code/screenshots/05_step07_baseline.png`  
> STEP 7: persistence baseline on the test set: RMSE 3.238, MAE 1.827, MAPE 1.159%.

*(word count: ~395)*

## Section 3: Experiments and optimisation (target ~500 words)

I ran the experiments in stages, using validation RMSE for every selection decision so the test set stays untouched until final scoring.

Stage one compared architecture width and depth. I trained four base models: an LSTM with 64 units, a stacked LSTM with 64 then 32 units, and the same two shapes as GRUs, all with dropout 0.2 after each recurrent layer. On validation RMSE the deeper LSTM won among the LSTMs (0.8738 vs 0.8743) while the single-layer GRU won among the GRUs (0.8734 vs 0.8785). The differences are small, which is itself a finding: on this data, adding a second recurrent layer buys little.

> **[Screenshot 6]** `_code/screenshots/06_step09_lstm64_summary_arch_training.png`  
> STEP 9: LSTM_64_d02: `model.summary()`, architecture diagram, training (test RMSE 3.2486).

> **[Screenshot 7]** `_code/screenshots/07_step10_lstm64x32_arch_training.png`  
> STEP 10: LSTM_64x32_d02: architecture diagram, training (test RMSE 3.2427).

> **[Screenshot 8]** `_code/screenshots/08_step11_gru64_arch_training.png`  
> STEP 11: GRU_64_d02: architecture diagram, training (test RMSE 3.2519).

> **[Screenshot 9]** `_code/screenshots/09_step12_gru64x32_arch_training.png`  
> STEP 12: GRU_64x32_d02: architecture diagram, training (test RMSE 3.2422).

> **[Screenshot 10]** `_code/screenshots/10_step13_best_model_pick.png`  
> STEP 13: validation RMSE comparison; best LSTM and best GRU selected.

Stage two was a dropout ablation. I retrained each stage-one winner with dropout removed, changing nothing else, so any difference is attributable to dropout alone. The outcome was mixed: removing dropout slightly improved the LSTM on the test set (RMSE 3.2403 vs 3.2427) but made the GRU clearly worse (3.2627 vs 3.2519). So dropout helped one architecture and not the other, and the differences are small enough that I avoid strong claims; the honest conclusion is that dropout at 0.2 is a safe default here rather than a decisive win.

> **[Screenshot 11]** `_code/screenshots/11_step14_lstm_no_dropout.png`  
> STEP 14: LSTM dropout ablation (LSTM_64x32_d0, test RMSE 3.2403).

> **[Screenshot 12]** `_code/screenshots/12_step15_gru_no_dropout.png`  
> STEP 15: GRU dropout ablation (GRU_64_d0, test RMSE 3.2627).

Stage three tested the input window length. Rebuilding the data with 30-day windows instead of 60 made the GRU worse (test RMSE 3.2682 vs 3.2627) and the run used its full 40-epoch allowance without early stopping firing, suggesting the shorter window gives the model less signal to converge on. Sixty days, roughly a quarter of trading days, stays the better choice.

> **[Screenshot 13]** `_code/screenshots/13_step16_window30.png`  
> STEP 16: 30-day window experiment (GRU_64_d0_w30, test RMSE 3.2682).

Stage four lowered the learning rate from 0.001 to 0.0001. This produced the best recurrent-only result of the whole investigation (GRU test RMSE 3.2385) and converged in only 6 epochs. A smaller learning rate takes smaller, more careful optimisation steps, which appears to suit this noisy target.

> **[Screenshot 14]** `_code/screenshots/14_step17_low_learning_rate.png`  
> STEP 17: lower learning-rate experiment (GRU_64_d0_lr1e-4, test RMSE 3.2385).

Overfitting was controlled throughout by early stopping with patience 5 and best-weight restoration, and its effect is visible in how differently the runs behaved: stopping points ranged from 6 to 40 epochs depending on the model. The loss curves show training and validation loss flattening early with no divergence between them, which indicates the models were not memorising the training data.

> **[Screenshot 18]** `_code/screenshots/18_step21_fig2_loss_curves.png`  
> STEP 21: Figure 2: training vs validation loss for LSTM_64x32_d02, GRU_64_d02, and CNN32k3_LSTM64.

Finally, my extension. The CNN-LSTM hybrid places a Conv1D layer (32 filters, kernel size 3, ReLU) and a max-pooling layer in front of a 64-unit LSTM. The convolution scans the window for short, local patterns over a few neighbouring days, and pooling halves the sequence length, so the LSTM reads a shorter, pre-digested sequence. Under the identical protocol the hybrid reached a test RMSE of 3.2278, and fine-tuning it, continuing training at a ten times lower learning rate, improved this to 3.2213, although MAE moved marginally the other way (1.8237 to 1.8253). Fine-tuning therefore gave a real but small and metric-dependent gain, which is worth knowing before spending the extra training time.

> **[Screenshot 15]** `_code/screenshots/15_step18_cnn_lstm_summary_arch_training.png`  
> STEP 18: CNN32k3_LSTM64: `model.summary()`, architecture diagram, training (test RMSE 3.2278).

> **[Screenshot 16]** `_code/screenshots/16_step19_finetune.png`  
> STEP 19: fine-tuning result (CNN32k3_LSTM64_finetuned, test RMSE 3.2213).

*(word count: ~470)*

## Section 4: Results and comparison (target ~500 words)

Table 1 collects every run, sorted by test RMSE. The single most important row is the baseline: persistence scores a test RMSE of 3.2380 USD with a MAPE of 1.159%. Every neural model in this investigation lands within about 1% of that number, on either side. That tight clustering is the central result, and it reflects a property of the data rather than a failure of the models: day-to-day price changes are close to unpredictable, so "tomorrow equals today" is very hard to beat.

> **[Screenshot 17]** `_code/screenshots/17_step20_final_results_table.png`  
> STEP 20: full results table (all models, sorted by test RMSE).

Within that narrow band, the ranking is still informative. Only the CNN-LSTM hybrid beat the baseline: 3.2278 as trained (0.31% better) and 3.2213 after fine-tuning (0.52% better). No plain LSTM or GRU configuration managed it; the best recurrent-only run, the GRU at the lower learning rate, scored 3.2385, essentially level with persistence.

On the brief's required comparison, LSTM versus GRU, the two are practically tied on accuracy: the best validation RMSEs differ by 0.04% (LSTM 0.8738, GRU 0.8734). The tiebreaker is efficiency. At the same width the GRU uses about a quarter fewer parameters than the LSTM (12,929 vs 16,961), because a GRU cell has two gates where an LSTM cell has three plus a separate cell state. In several runs the GRU also trained faster (for example 8.9 vs 20.0 seconds for the stacked variants). My verdict is therefore that GRU is the more suitable of the two required architectures for this scenario: equal accuracy at lower cost. This matches the general pattern reported in the literature, where GRUs frequently match LSTMs on smaller or less complex sequence problems.

Two further checks qualify these results. First, a supplementary multi-seed run (three seeds; executed on an earlier version of the notebook whose logic is identical, only comments and structure were later simplified) shows the same picture on average: the hybrid has the lowest mean test RMSE (3.2347) and beat the baseline in two of three seeds, while the LSTM managed it in one and the GRU in none. The hybrid's edge is therefore real on average but seed-sensitive, and I state it accordingly. Second, the same run measured direction accuracy, whether the model at least calls tomorrow's up-or-down correctly: all three models average 51.0 to 51.8%, below the 52.9% you would get by always guessing "up". The models track price levels closely but do not reliably predict direction.

Figure 3 makes the level-versus-direction point visible. The predicted curves hug the actual price impressively through the final test year, including the March 2020 crash, but on close inspection they trail the actual price by roughly one day: the models have largely learned a slightly refined version of persistence. This is exactly what the metrics say, and it is why I treat the visual fit as weak evidence on its own.

> **[Screenshot 19]** `_code/screenshots/19_step22_fig3_predictions.png`  
> STEP 22: Figure 3: predicted vs actual prices (final test year).

In summary: the hybrid extension delivered the best accuracy and the only baseline-beating runs; GRU wins the required comparison on efficiency; and every improvement is small enough that Section 5 must ask what these models are actually worth in use.

> **[Screenshot 20]** `_code/screenshots/20_step23_summaries_saved.png`  
> STEP 23: saved model summaries and best-model recap.

*(word count: ~490)*

## Section 5: Critical evaluation, Task 2 (target ~900 words)

*(Citation markers like [S-004] refer to the source ledger and will become Harvard citations at assembly, after each source is verified.)*

**What the results are worth.** Taken at face value, my best model beats a naive baseline by half a percent of RMSE. Taken honestly, that summary needs three qualifications. First, the margin is small enough that a different random seed can erase it: across three seeds the hybrid beat persistence twice, not three times. Second, every model, baseline included, sits at a MAPE of about 1.16%, so no model offers a step change in predictive power; they all inherit the same hard limit from the data. Third, direction accuracy, arguably the number a trader would care about most, is below the trivial always-guess-up strategy for all three architectures. A model that tracks the price level beautifully but cannot call tomorrow's direction is of very limited economic use. The correct inference from this investigation is not "deep learning predicts stock prices"; it is that on a single price series, one step ahead, deep learning approximately reproduces persistence, and only a carefully constructed hybrid extracts a small additional signal.

**Why the results came out this way.** Daily returns on liquid stocks are close to serially uncorrelated, which is the empirical core of the efficient market view: if yesterday's price movement reliably predicted today's, traders would exploit the pattern until it disappeared. My models only see 60 days of past prices of one stock, so they cannot know about earnings, news, rates, or the pandemic that ends my test window. Given that input, learning "tomorrow is close to today, adjusted slightly for recent drift" may genuinely be the optimal answer, and that is what the loss curves suggest the models converged to, quickly and without overfitting. The design was sound; the ceiling belongs to the problem.

**Reliability and validity of my own process.** The strongest parts of the design are the chronological split, the leakage-free per-window normalisation, validation-only model selection, one identical protocol for every model, and a seeded, rerunnable notebook. The honest weaknesses are equally identifiable. I evaluated on a single train/validation/test split rather than walk-forward validation, so my test estimate rests on one particular slice of history that happens to end in the March 2020 crash. My multi-seed check used three seeds, enough to show the ranking is seed-sensitive, not enough to bound it precisely. And the models are univariate; adding volume, which the dataset already contains, would have been the natural next input to test. These limitations do not invalidate the comparison, LSTM against GRU under identical conditions, but they cap how far the absolute numbers should be trusted.

**The comparison verdict, inferred from the results.** GRU matched LSTM at around a quarter fewer parameters and often less training time, so for this scenario I would recommend GRU as the recurrent workhorse, consistent with published comparisons [S-007]. The hybrid result supports the wider trend in the field: convolution-plus-recurrence architectures repeatedly outperform standalone recurrent models on financial series [S-008, S-009], and my small but reproducible-on-average improvement fits that pattern. The logical next steps beyond this report, attention mechanisms and Transformer-based forecasters, extend the same idea of combining complementary inductive biases, though published results suggest they need substantially more tuning and data to pay off [S-007].

**Ethical implications.** A stock predictor is not an ethically neutral artefact, because people act on it with real money. The first risk is overtrust: Figure 3 looks compellingly accurate, and a user who does not understand that the model is essentially lagging the price by a day could take on serious financial risk; my own results show the honest predictive content is close to zero [S-005]. The second is automation bias, the documented tendency to defer to an algorithmic output even against one's own judgement, which is dangerous when the output is a confident-looking number with no uncertainty attached [S-006]. Third, accountability: if a firm deploys such a model and clients lose money, responsibility must sit with identifiable humans and documented validation, not be diffused onto "the algorithm" [S-004]. Fourth, at market scale, many actors running similar models trained on the same public data can correlate their behaviour and amplify moves, a systemic concern that individual model accuracy says nothing about. The mitigations follow from the failures: report performance against a naive baseline (as this report does), attach uncertainty and known failure modes to any forecast, keep a human decision-maker in the loop, and never present a research artefact like this one as investment advice. Documented, modest claims are themselves an ethical control [S-004].

**Reflection.** This assignment changed my working assumptions more than my toolkit. Keras and Colab made the engineering almost frictionless; the genuinely hard parts were experimental honesty, respecting time order, fitting nothing on the future, selecting only on validation data, and resisting the pull of a beautiful prediction plot. The single most useful habit I developed was evaluating against persistence from the start: it reframed every result and prevented me from writing a far more flattering, and wrong, report. If I repeated the work I would build walk-forward evaluation in from the beginning, add volume as a second feature, and predict returns or direction explicitly rather than price levels, because that is the question users actually need answered. What I take forward is a calibrated scepticism: deep learning is a powerful function approximator, but it cannot conjure signal that a market has already priced away, and recognising when a problem is signal-poor is as important a professional skill as building the model.

> **[Screenshot 21]** `_code/screenshots/21_step24_conclusion.png`  
> STEP 24: computed conclusion from the notebook (best model, baseline gap, LSTM vs GRU, hybrid verdict).

*(word count: ~900)*

## References

Not yet drafted (see `SOURCE_LEDGER.md`).

## Appendix: code

The final notebook is `_code/COM7019_25199053.ipynb`. Formatting for appendix inclusion happens at final assembly.

---

## Screenshot index (quick reference)

| # | File | Notebook step | Report section |
|---:|---|---|---|
| 1 | `01_step02_load_data.png` | STEP 2 | Section 2 |
| 2 | `02_step03_data_checks.png` | STEP 3 | Section 2 |
| 3 | `03_step04_split_fig1.png` | STEP 4 | Section 2 |
| 4 | `04_step05_windowing.png` | STEP 5 | Section 2 |
| 5 | `05_step07_baseline.png` | STEP 7 | Section 2 |
| 6 | `06_step09_lstm64_summary_arch_training.png` | STEP 9 | Section 3 |
| 7 | `07_step10_lstm64x32_arch_training.png` | STEP 10 | Section 3 |
| 8 | `08_step11_gru64_arch_training.png` | STEP 11 | Section 3 |
| 9 | `09_step12_gru64x32_arch_training.png` | STEP 12 | Section 3 |
| 10 | `10_step13_best_model_pick.png` | STEP 13 | Section 3 |
| 11 | `11_step14_lstm_no_dropout.png` | STEP 14 | Section 3 |
| 12 | `12_step15_gru_no_dropout.png` | STEP 15 | Section 3 |
| 13 | `13_step16_window30.png` | STEP 16 | Section 3 |
| 14 | `14_step17_low_learning_rate.png` | STEP 17 | Section 3 |
| 15 | `15_step18_cnn_lstm_summary_arch_training.png` | STEP 18 | Section 3 |
| 16 | `16_step19_finetune.png` | STEP 19 | Section 3 |
| 17 | `17_step20_final_results_table.png` | STEP 20 | Section 4 |
| 18 | `18_step21_fig2_loss_curves.png` | STEP 21 | Section 3 |
| 19 | `19_step22_fig3_predictions.png` | STEP 22 | Section 4 |
| 20 | `20_step23_summaries_saved.png` | STEP 23 | Section 4 |
| 21 | `21_step24_conclusion.png` | STEP 24 | Section 5 |

*Note: STEP 6 (train/val/test sample counts after windowing: 6,876 / 1,486 / 1,487) has no dedicated screenshot; the numbers are in the executed notebook and `RESULTS_LEDGER.md`.*
