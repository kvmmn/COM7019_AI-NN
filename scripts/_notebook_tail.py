# Tail cells for upgrade_notebook.py — model training through conclusion

def _c(s):
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in s.strip().split("\n")], "outputs": [], "execution_count": None}

def _m(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in s.strip().split("\n")]}

TAIL_CELLS = [
    _c('''
# STEP 8: LSTM single layer, 64 units, dropout 0.2
NAME = build_model_name("LSTM", "1Layer", "64Units", 0.2, PRIMARY_WINDOW_KEY)
model = keras.Sequential(name=NAME)
model.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dropout(DROPOUT_RATE))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
model.summary()
show_architecture(model, "arch_" + NAME + ".png")
model_lstm_1L, t = train_model(model, NAME, X_train, y_train, X_val, y_val)
evaluate_model(model_lstm_1L, NAME, t, X_test, y_test, base_test)
'''),
    _c('''
# STEP 9: LSTM stacked 64→32, dropout 0.2
NAME = build_model_name("LSTM", "2Layer", "64-32Units", 0.2, PRIMARY_WINDOW_KEY)
model = keras.Sequential(name=NAME)
model.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
model.add(keras.layers.LSTM(64, return_sequences=True))
model.add(keras.layers.Dropout(DROPOUT_RATE))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dropout(DROPOUT_RATE))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
show_architecture(model, "arch_" + NAME + ".png")
model_lstm_2L, t = train_model(model, NAME, X_train, y_train, X_val, y_val)
evaluate_model(model_lstm_2L, NAME, t, X_test, y_test, base_test)
'''),
    _c('''
# STEP 10: GRU single layer, 64 units, dropout 0.2
NAME = build_model_name("GRU", "1Layer", "64Units", 0.2, PRIMARY_WINDOW_KEY)
model = keras.Sequential(name=NAME)
model.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
model.add(keras.layers.GRU(64))
model.add(keras.layers.Dropout(DROPOUT_RATE))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
show_architecture(model, "arch_" + NAME + ".png")
model_gru_1L, t = train_model(model, NAME, X_train, y_train, X_val, y_val)
evaluate_model(model_gru_1L, NAME, t, X_test, y_test, base_test)
'''),
    _c('''
# STEP 11: GRU stacked 64→32, dropout 0.2
NAME = build_model_name("GRU", "2Layer", "64-32Units", 0.2, PRIMARY_WINDOW_KEY)
model = keras.Sequential(name=NAME)
model.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
model.add(keras.layers.GRU(64, return_sequences=True))
model.add(keras.layers.Dropout(DROPOUT_RATE))
model.add(keras.layers.GRU(32))
model.add(keras.layers.Dropout(DROPOUT_RATE))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
show_architecture(model, "arch_" + NAME + ".png")
model_gru_2L, t = train_model(model, NAME, X_train, y_train, X_val, y_val)
evaluate_model(model_gru_2L, NAME, t, X_test, y_test, base_test)
'''),
    _m("### Dropout ablation (validation-only selection)"),
    _c('''
# STEP 12: Pick best LSTM and GRU by validation RMSE
lstm_models = {
    build_model_name("LSTM", "1Layer", "64Units", 0.2, PRIMARY_WINDOW_KEY): model_lstm_1L,
    build_model_name("LSTM", "2Layer", "64-32Units", 0.2, PRIMARY_WINDOW_KEY): model_lstm_2L,
}
gru_models = {
    build_model_name("GRU", "1Layer", "64Units", 0.2, PRIMARY_WINDOW_KEY): model_gru_1L,
    build_model_name("GRU", "2Layer", "64-32Units", 0.2, PRIMARY_WINDOW_KEY): model_gru_2L,
}

def pick_best(models_dict, family):
    best_name, best_rmse, best_model = None, None, None
    print("Validation RMSE —", family)
    for name, m in models_dict.items():
        v = val_rmse_for_model(m, X_val, y_val, base_val)
        print(" ", name, ":", round(v, 4))
        if best_rmse is None or v < best_rmse:
            best_rmse, best_name, best_model = v, name, m
    print("Best", family, ":", best_name, "\\n")
    return best_name, best_model

best_lstm_name, best_lstm_model = pick_best(lstm_models, "LSTM")
best_gru_name, best_gru_model = pick_best(gru_models, "GRU")
'''),
    _c('''
# STEP 13: LSTM dropout ablation
if "1Layer" in best_lstm_name:
    NAME = build_model_name("LSTM", "1Layer", "64Units", None, PRIMARY_WINDOW_KEY)
    model = keras.Sequential(name=NAME)
    model.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(1))
else:
    NAME = build_model_name("LSTM", "2Layer", "64-32Units", None, PRIMARY_WINDOW_KEY)
    model = keras.Sequential(name=NAME)
    model.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
    model.add(keras.layers.LSTM(64, return_sequences=True))
    model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
model, t = train_model(model, NAME, X_train, y_train, X_val, y_val)
evaluate_model(model, NAME, t, X_test, y_test, base_test)
'''),
    _c('''
# STEP 14: GRU dropout ablation
NAME = build_model_name("GRU", "1Layer", "64Units", None, PRIMARY_WINDOW_KEY)
model_gru_nodrop = keras.Sequential(name=NAME)
model_gru_nodrop.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
model_gru_nodrop.add(keras.layers.GRU(64))
model_gru_nodrop.add(keras.layers.Dense(1))
model_gru_nodrop.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
model_gru_nodrop, t = train_model(model_gru_nodrop, NAME, X_train, y_train, X_val, y_val)
evaluate_model(model_gru_nodrop, NAME, t, X_test, y_test, base_test)
'''),
    _m("### Window ablation: 1W (5d), 1M (21d), 1Y (252d) trading days"),
    _c('''
# STEP 15: Compare all three finance-standard windows (same GRU, no dropout)
window_ablation_results = {}
for wkey in ["1W", "1M", "1Y"]:
    wsize = WINDOWS[wkey]["days"]
    Xw, yw, basew = build_windowed_data(wsize)
    (Xtr, ytr, btr), (Xva, yva, bva), (Xte, yte, bte) = split_windowed(Xw, yw, basew, wsize)
    wname = build_model_name("GRU", "1Layer", "64Units", None, wkey)
    m = keras.Sequential(name=wname)
    m.add(keras.layers.Input(shape=(wsize, 1)))
    m.add(keras.layers.GRU(64))
    m.add(keras.layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
    m, elapsed = train_model(m, wname, Xtr, ytr, Xva, yva)
    rmse = evaluate_model(m, wname, elapsed, Xte, yte, bte)
    window_ablation_results[f"{wkey} ({wsize}d)"] = rmse

plt.figure(figsize=(6, 4))
plt.bar(list(window_ablation_results.keys()), list(window_ablation_results.values()),
        color=["steelblue", "seagreen", "darkorange"])
plt.ylabel("Test RMSE (USD)")
plt.title("Figure 5: Window length ablation (GRU, no dropout)")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig5_window_ablation.png")
plt.show()
'''),
    _c('''
# STEP 16: Lower learning rate on primary window
NAME = build_model_name("GRU", "1Layer", "64Units", None, PRIMARY_WINDOW_KEY, learning_rate=0.0001)
model = keras.Sequential(name=NAME)
model.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
model.add(keras.layers.GRU(64))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])
model, t = train_model(model, NAME, X_train, y_train, X_val, y_val, lr=0.0001)
evaluate_model(model, NAME, t, X_test, y_test, base_test)
'''),
    _m("### CNN-LSTM hybrid (extension)"),
    _c('''
# STEP 17: Hybrid CNN-LSTM
NAME = build_model_name("Hybrid", "CNN32-LSTM64", "Conv1D+LSTM", 0.2, PRIMARY_WINDOW_KEY)
model_cnn_lstm = keras.Sequential(name=NAME)
model_cnn_lstm.add(keras.layers.Input(shape=(WINDOW_SIZE, 1)))
model_cnn_lstm.add(keras.layers.Conv1D(32, 3, activation="relu", padding="causal"))
model_cnn_lstm.add(keras.layers.MaxPooling1D(2))
model_cnn_lstm.add(keras.layers.LSTM(64))
model_cnn_lstm.add(keras.layers.Dropout(DROPOUT_RATE))
model_cnn_lstm.add(keras.layers.Dense(1))
model_cnn_lstm.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
model_cnn_lstm.summary()
show_architecture(model_cnn_lstm, "arch_" + NAME + ".png")
model_cnn_lstm, t = train_model(model_cnn_lstm, NAME, X_train, y_train, X_val, y_val)
evaluate_model(model_cnn_lstm, NAME, t, X_test, y_test, base_test)
'''),
    _c('''
# STEP 18: Fine-tune hybrid
FT_NAME = build_model_name("Hybrid", "CNN32-LSTM64", "Conv1D+LSTM", 0.2, PRIMARY_WINDOW_KEY, finetuned=True)
model_cnn_lstm.optimizer.learning_rate.assign(float(model_cnn_lstm.optimizer.learning_rate.numpy()) / 10)
model_cnn_lstm, t = train_model(model_cnn_lstm, FT_NAME, X_train, y_train, X_val, y_val)
evaluate_model(model_cnn_lstm, FT_NAME, t, X_test, y_test, base_test)
'''),
    _c('''
# STEP 19: Results table
results_df = pd.DataFrame(all_results).sort_values("Test RMSE").reset_index(drop=True)
print("\\n=== FINAL RESULTS ===")
print(results_df[["Model", "Test RMSE", "Test MAE", "Test MAPE %", "Parameters", "Train time (s)"]].to_string(index=False))
results_df.to_csv(OUTPUT_FOLDER + "/tables/results_all_runs.csv", index=False)
results_df
'''),
    _c('''
# STEP 20: Graphical model comparison (bar charts)
plot_df = results_df[results_df["Model"] != "Baseline (persistence)"].head(12)
metrics = ["Test RMSE", "Test MAE", "Test MAPE %"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
x = np.arange(len(plot_df))
for ax, metric in zip(axes, metrics):
    ax.bar(x, plot_df[metric], color=plt.cm.tab20(np.linspace(0, 1, len(plot_df))))
    ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=65, ha="right", fontsize=6)
fig.suptitle("Figure 4: Model comparison (test metrics)")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig4_model_comparison_bars.png")
plt.show()
'''),
    _c('''
# STEP 20b: Regularisation matrix (heatmap)
rows = []
for _, row in results_df.iterrows():
    n = row["Model"]
    if n == "Baseline (persistence)":
        continue
    if n.startswith("LSTM"):
        fam = "LSTM"
    elif n.startswith("GRU"):
        fam = "GRU"
    elif n.startswith("Hybrid"):
        fam = "Hybrid"
    else:
        continue
    if "FineTuned" in n:
        reg = "Fine-tuned"
    elif "NoDropout" in n:
        reg = "No dropout"
    elif "Dropout0.2" in n:
        reg = "Dropout 0.2"
    elif "LR0.0001" in n:
        reg = "LR 1e-4"
    else:
        reg = "Other"
    rows.append({"Family": fam, "Setting": reg, "Test RMSE": row["Test RMSE"]})
reg_df = pd.DataFrame(rows)
if len(reg_df):
    pivot = reg_df.pivot_table(index="Family", columns="Setting", values="Test RMSE", aggfunc="min")
    plt.figure(figsize=(8, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu_r")
    plt.title("Figure 6: Regularisation matrix (test RMSE, lower is better)")
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "/figures/fig6_regularisation_matrix.png")
    plt.show()
'''),
    _c('''
# STEP 21: Loss curves
plot_names = [best_lstm_name, best_gru_name,
              build_model_name("Hybrid", "CNN32-LSTM64", "Conv1D+LSTM", 0.2, PRIMARY_WINDOW_KEY)]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, name in zip(axes, plot_names):
    if name not in training_histories:
        continue
    h = training_histories[name]
    ax.plot(h["loss"], label="train", color="steelblue")
    ax.plot(h["val_loss"], label="val", color="seagreen")
    ax.set_title(name[:40], fontsize=8)
    ax.legend(fontsize=7)
fig.suptitle("Figure 2: Training vs validation loss")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig2_loss_curves.png")
plt.show()
'''),
    _c('''
# STEP 22: Predictions vs actual (last year of test)
test_dates = dates[val_end : val_end + len(y_test)]
actual_usd = (y_test + 1.0) * base_test
n_show = 250
sl = slice(-n_show, None)
plt.figure(figsize=(11, 4))
plt.plot(test_dates[sl], actual_usd[sl], color="gray", linewidth=1.5, label="Actual")
for m, col, lab in [(best_lstm_model, "steelblue", best_lstm_name),
                    (best_gru_model, "seagreen", best_gru_name),
                    (model_cnn_lstm, "darkorange", "Hybrid")]:
    pred_usd = (m.predict(X_test, verbose=0).flatten() + 1.0) * base_test
    plt.plot(test_dates[sl], pred_usd[sl], color=col, linewidth=1, alpha=0.85, label=lab[:30])
plt.ylabel("Adj Close (USD)")
plt.title("Figure 3: Predicted vs actual (final test year)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig3_predictions_vs_actual.png")
plt.show()
'''),
    _c('''
# STEP 23: Save model summaries
with open(OUTPUT_FOLDER + "/models/model_summaries.txt", "w") as f:
    for m in [best_lstm_model, best_gru_model, model_cnn_lstm]:
        m.summary(print_fn=lambda line: f.write(line + "\\n"))
        f.write("\\n" + "=" * 60 + "\\n\\n")
print("Summaries saved. Baseline RMSE:", round(baseline_rmse, 4))
'''),
    _c('''
# STEP 24: Conclusion (computed from results table)
best = results_df.iloc[0]
gap = (best["Test RMSE"] / baseline_rmse - 1) * 100
print("=== CONCLUSION ===")
print("Best model:", best["Model"], "| Test RMSE:", round(best["Test RMSE"], 4))
print("Baseline RMSE:", round(baseline_rmse, 4), "| Gap:", round(gap, 2), "%")
print("LSTM vs GRU val RMSE:", round(val_rmse_for_model(best_lstm_model, X_val, y_val, base_val), 4),
      "vs", round(val_rmse_for_model(best_gru_model, X_val, y_val, base_val), 4))
print("All numbers from this notebook run only.")
'''),
]
