import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import RESULTS_DIR

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save(name):
    return os.path.join(RESULTS_DIR, name)


def nn_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


def to_dollar(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_mean = np.mean(y_true)
    nrmse_pct = (rmse / y_mean) * 100
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE%": nrmse_pct,
        "MAPE%": mape,
        "R2": r2,
    }


# ── Plotting ──────────────────────────────────────────────────
def plot_training_curves(histories, filename):
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, (name, h) in enumerate(histories.items()):
        ax = axes[0, i]
        ax.plot(h["train_loss"], label="Train", alpha=0.8)
        ax.plot(h["val_loss"], label="Val", alpha=0.8)
        be = h["best_epoch"]
        ax.axvline(be, color="red", ls="--", alpha=0.5, label=f"Best ep={be}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save(filename), dpi=150, bbox_inches="tight")
    plt.close()


def plot_overlay(histories, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, h in histories.items():
        ax.plot(h["val_loss"], label=f"{name} (best ep={h['best_epoch']})", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save(filename), dpi=150, bbox_inches="tight")
    plt.close()


def plot_diagnostics(y_actual, y_pred, title, filename):
    residuals = y_actual - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(y_actual, y_pred, alpha=0.5, s=30, c="#4338CA")
    lims = [y_actual.min(), y_actual.max()]
    axes[0].plot(lims, lims, "r--", lw=2, label="Perfect")
    axes[0].set_xlabel("Actual ($)")
    axes[0].set_ylabel("Predicted ($)")
    axes[0].set_title(f"{title}\nR²={r2_score(y_actual, y_pred):.4f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30, c="#F59E0B")
    axes[1].axhline(0, color="r", lw=2, ls="--")
    axes[1].set_xlabel("Predicted ($)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(residuals, bins=40, color="#7C3AED", alpha=0.85, density=True, edgecolor="k")
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    axes[2].plot(xr, sp_stats.norm.pdf(xr, residuals.mean(), residuals.std()), "r-", lw=2)
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(save(filename), dpi=150, bbox_inches="tight")
    plt.close()


def plot_importance(importances, names, filename):
    idx = np.argsort(importances)[::-1]
    n = min(15, len(names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(n), importances[idx[:n]], color="#0D9488")
    ax.set_yticks(range(n))
    ax.set_yticklabels([names[i] for i in idx[:n]])
    ax.set_xlabel("RMSE increase when feature shuffled ($)")
    ax.set_title("Neural Network Permutation Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save(filename), dpi=150, bbox_inches="tight")
    plt.close()


def permutation_importance(model, X, y_s, scaler_y, n_repeats=10):
    y_true_log = to_dollar(y_s, scaler_y)
    y_true = np.expm1(y_true_log)
    base_pred_log = to_dollar(nn_predict(model, X), scaler_y)
    base_pred = np.expm1(base_pred_log)
    base_rmse = np.sqrt(mean_squared_error(y_true, base_pred))

    imp = np.zeros(X.shape[1])
    for col in range(X.shape[1]):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            np.random.shuffle(Xp[:, col])
            pp_log = to_dollar(nn_predict(model, Xp), scaler_y)
            pp = np.expm1(pp_log)
            deltas.append(np.sqrt(mean_squared_error(y_true, pp)) - base_rmse)
        imp[col] = np.mean(deltas)
    return imp


def evaluate_all(data, nn_models, nn_hists, nn_configs, linears, bl_results):
    y_test = data["y_test"]
    sc_y = data["sc_y"]
    X_test = data["X_test"]
    feature_names = data["feature_names"]

    # ── Step 4: Training Diagnostics ──
    print("\n" + "=" * 70)
    print("  Step 4: Training Diagnostics")
    print("=" * 70)

    arch = {k: nn_hists[k] for k in ["A: Shallow [64]", "B: Medium [128-64]", "C: Deep [256-128-64]"]}
    plot_training_curves(arch, "step4_architecture_curves.png")
    plot_overlay(arch, "Architecture Comparison (Val Loss)", "step4_architecture_overlay.png")
    print("  Saved architecture comparison curves")

    act = {k: nn_hists[k] for k in ["C: Deep [256-128-64]", "D: Deep+Sigmoid"]}
    plot_overlay(act, "Activation: ReLU vs Sigmoid (Val Loss)", "step4_activation_comparison.png")
    print("  Saved activation comparison")

    reg = {k: nn_hists[k] for k in ["C: Deep [256-128-64]", "E: Deep+Dropout0.3", "F: Deep+L2"]}
    plot_overlay(reg, "Regularization Comparison (Val Loss)", "step4_regularization_comparison.png")
    print("  Saved regularization comparison")

    # ── Step 5: Final Comparison Table ──
    print("\n" + "=" * 70)
    print("  Step 5: All Models Comparison")
    print("=" * 70)

    nn_results = {}
    for name, model in nn_models.items():
        pred_log = to_dollar(nn_predict(model, X_test), sc_y)
        pred = np.expm1(pred_log)
        m = metrics(y_test, pred)
        m["y_pred"] = pred
        nn_results[name] = m

    all_rows = []
    for name, r in bl_results.items():
        all_rows.append({"Model": name, "RMSE": r["RMSE"], "MAE": r["MAE"], "NRMSE%": r["NRMSE%"], "MAPE%": r["MAPE%"], "R2": r["R2"], "Type": "Linear"})
    for name, r in nn_results.items():
        all_rows.append({"Model": name, "RMSE": r["RMSE"], "MAE": r["MAE"], "NRMSE%": r["NRMSE%"], "MAPE%": r["MAPE%"], "R2": r["R2"], "Type": "NN"})

    comp_df = pd.DataFrame(all_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    comp_df.to_csv(save("step5_comparison_table.csv"), index=False)
    print(comp_df.to_string(index=False))

    best_nn_name = max(nn_results, key=lambda k: nn_results[k]["R2"])
    best_nn = nn_results[best_nn_name]
    best_bl = max(bl_results, key=lambda k: bl_results[k]["R2"])
    print(f"\n  Best NN:       {best_nn_name}  R²={best_nn['R2']:.4f}  RMSE=${best_nn['RMSE']:,.0f}")
    print(f"  Best Baseline: {best_bl}  R²={bl_results[best_bl]['R2']:.4f}  RMSE=${bl_results[best_bl]['RMSE']:,.0f}")
    delta = best_nn["R2"] - bl_results[best_bl]["R2"]
    if delta > 0:
        print(f"  => NN wins by R² +{delta:.4f}")
    else:
        print(f"  => Baseline wins by R² +{-delta:.4f}")

    plot_diagnostics(y_test, best_nn["y_pred"], f"Best NN: {best_nn_name}", "step5_nn_diagnostics.png")
    plot_diagnostics(y_test, bl_results[best_bl]["y_pred"], f"Best Baseline: {best_bl}", "step5_baseline_diagnostics.png")
    print("  Saved diagnostic plots")

    # ── Step 6: Feature Importance & Business Insights ──
    print("\n" + "=" * 70)
    print("  Step 6: Feature Importance & Business Insights")
    print("=" * 70)

    best_model = nn_models[best_nn_name]
    y_test_log = data["y_test_log"]

    print("  Computing permutation importance ...")
    imp = permutation_importance(best_model, X_test, y_test_log, data["sc_y"])
    plot_importance(imp, feature_names, "step6_permutation_importance.png")

    imp_sorted = np.argsort(imp)[::-1]
    print("\n  NN Permutation Importance:")
    imp_rows = []
    for rank, idx in enumerate(imp_sorted):
        print(f"    {rank + 1:2d}. {feature_names[idx]:28s}  ${imp[idx]:>10,.0f}")
        imp_rows.append({"rank": rank + 1, "feature": feature_names[idx], "importance_rmse": round(imp[idx], 2)})
    pd.DataFrame(imp_rows).to_csv(save("step6_nn_importance.csv"), index=False)

    lasso = linears["Lasso"]
    lc = np.abs(lasso.coef_)
    lc_sorted = np.argsort(lc)[::-1]
    print("\n  Lasso |coef| (for comparison):")
    lasso_rows = []
    for rank, idx in enumerate(lc_sorted):
        print(f"    {rank + 1:2d}. {feature_names[idx]:28s}  {lc[idx]:>10.4f}")
        lasso_rows.append({"rank": rank + 1, "feature": feature_names[idx], "abs_coef": round(lc[idx], 4)})
    pd.DataFrame(lasso_rows).to_csv(save("step6_lasso_importance.csv"), index=False)

    res_nn = y_test - best_nn["y_pred"]
    abs_res = np.abs(res_nn)
    worst_idx = np.argsort(abs_res)[::-1][:10]

    worst = pd.DataFrame({
        "Actual": y_test[worst_idx],
        "NN_Predicted": best_nn["y_pred"][worst_idx],
        "NN_Residual": res_nn[worst_idx],
        "Baseline_Predicted": bl_results[best_bl]["y_pred"][worst_idx],
        "Baseline_Residual": (y_test - bl_results[best_bl]["y_pred"])[worst_idx],
    })
    worst.to_csv(save("step6_worst_predictions.csv"), index=False)
    print(f"\n  Top 10 worst NN predictions:")
    print(worst.to_string())

    print("\n" + "=" * 70)
    print(f"  All results saved to: {RESULTS_DIR}")
    print("=" * 70)
