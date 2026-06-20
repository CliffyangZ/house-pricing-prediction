#!/usr/bin/env python3
"""
Ames Housing: Neural Network vs Regularized Linear Models
期末專題 — PyTorch MLP 實驗

覆蓋 architecture.md 六大步驟：
  Step 1: 期中迴歸結果回顧（OLS / Ridge / Lasso / ElasticNet baseline）
  Step 2: 神經網路資料前處理
  Step 3: 建立神經網路（至少 3 種架構）
  Step 4: 訓練過程診斷（loss curves, overfitting）
  Step 5: 模型比較
  Step 6: 商業洞察（permutation importance, worst predictions）

Usage:
    uv run python experiments/train.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sp_stats
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp import MLP

# ── Configuration ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 64
MAX_EPOCHS = 500
EARLY_STOP_PATIENCE = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "AmesHousing.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "neural_network")
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ── Step 2: Data Loading & Preprocessing ──────────────────────
def load_and_preprocess():
    """沿用期中前處理流程：缺失值補值、outlier 刪除、類別編碼、log1p、相關性選特徵。"""
    raw_df = pd.read_csv(DATA_PATH)

    selected_features = [
        "Gr Liv Area", "Overall Qual", "Year Built", "Total Bsmt SF",
        "Garage Cars", "Full Bath", "TotRms AbvGrd", "Neighborhood",
        "Kitchen Qual", "Sale Condition", "Garage Area", "Overall Cond",
        "Fireplaces", "Bsmt Full Bath", "1st Flr SF",
    ]

    df = raw_df[selected_features + ["SalePrice"]].copy()

    # 缺失值處理
    num_cols = df[selected_features].select_dtypes(include=["number"]).columns
    cat_cols = df[selected_features].select_dtypes(include=["object", "str"]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")

    # Outlier 刪除
    m1 = (df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 300000)
    m2 = (df["Total Bsmt SF"] > 5000) & (df["SalePrice"] < 300000)
    df = df[~(m1 | m2)].copy()

    # Kitchen Qual ordinal encoding
    quality_map = {"Missing": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    df["Kitchen Qual"] = df["Kitchen Qual"].map(quality_map).astype(int)

    # Frequency encoding（沿用期中：在全資料上計算頻率，維持比較一致性）
    for col in ["Neighborhood", "Sale Condition"]:
        freq = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_frequency"] = df[col].map(freq).fillna(0)
        df = df.drop(columns=col)

    # 以 abs(corr) 排序選特徵（期中選 top 15 即全部）
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)
    top_features = corr.head(15).index.tolist()

    # 面積特徵 log1p 轉換
    for c in ["Gr Liv Area", "Total Bsmt SF", "Garage Area", "1st Flr SF"]:
        if c in df.columns:
            df[c] = np.log1p(df[c])

    return df[top_features].values, df["SalePrice"].values, top_features


# ── Early Stopping ────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.stopped = False

    def step(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True


# ── Step 3: NN Training ──────────────────────────────────────
def train_nn(config, X_train, X_val, y_train, y_val, input_dim):
    torch.manual_seed(RANDOM_STATE)
    model = MLP(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        activation=config.get("activation", "relu"),
        dropout=config.get("dropout", 0.0),
    ).to(DEVICE)

    wd = config.get("weight_decay", 0.0)
    lr = config.get("lr", 0.001)
    if config.get("optimizer", "adam") == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_X = torch.FloatTensor(X_val).to(DEVICE)
    val_y = torch.FloatTensor(y_val).to(DEVICE)

    es = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    hist = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for _ in range(MAX_EPOCHS):
        model.train()
        batch_losses = []
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            vl = criterion(model(val_X), val_y).item()

        hist["train_loss"].append(np.mean(batch_losses))
        hist["val_loss"].append(vl)
        es.step(vl, model)
        if es.stopped:
            break

    elapsed = time.time() - t0
    if es.best_state:
        model.load_state_dict(es.best_state)

    best_ep = len(hist["val_loss"]) - es.counter
    hist["best_epoch"] = best_ep
    hist["total_epochs"] = len(hist["val_loss"])
    hist["elapsed"] = elapsed
    return model, hist


# ── Evaluation helpers ────────────────────────────────────────
def nn_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


def to_dollar(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ── Plotting ──────────────────────────────────────────────────
def save(name):
    return os.path.join(RESULTS_DIR, name)


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


# ── Step 6: Permutation Importance ────────────────────────────
def permutation_importance(model, X, y_s, scaler_y, n_repeats=10):
    y_true = to_dollar(y_s, scaler_y)
    base_pred = to_dollar(nn_predict(model, X), scaler_y)
    base_rmse = np.sqrt(mean_squared_error(y_true, base_pred))

    imp = np.zeros(X.shape[1])
    for col in range(X.shape[1]):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            np.random.shuffle(Xp[:, col])
            pp = to_dollar(nn_predict(model, Xp), scaler_y)
            deltas.append(np.sqrt(mean_squared_error(y_true, pp)) - base_rmse)
        imp[col] = np.mean(deltas)
    return imp


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Ames Housing: Neural Network vs Regularized Linear Models")
    print("=" * 70)

    # ── Load & preprocess ──
    X, y, feature_names = load_and_preprocess()
    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features")

    # ── Split: same test as midterm, val from train ──
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    print(f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

    # Scalers for NN (fit on train only)
    sc_X = StandardScaler().fit(X_train)
    sc_y = StandardScaler().fit(y_train.reshape(-1, 1))
    Xtr = sc_X.transform(X_train)
    Xva = sc_X.transform(X_val)
    Xte = sc_X.transform(X_test)
    ytr = sc_y.transform(y_train.reshape(-1, 1)).ravel()
    yva = sc_y.transform(y_val.reshape(-1, 1)).ravel()

    # Scalers for linear baselines (fit on full trainval, scale X only — raw y)
    sc_X_full = StandardScaler().fit(X_tv)
    Xtv_s = sc_X_full.transform(X_tv)
    Xte_s = sc_X_full.transform(X_test)

    # ════════════════════════════════════════════════════════════
    # Step 1: Baseline Linear Models
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Step 1: Baseline (OLS / Ridge / Lasso / ElasticNet)")
    print("=" * 70)

    linears = {
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(cv=5),
        "Lasso": LassoCV(cv=5, max_iter=10000),
        "ElasticNet": ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0], max_iter=10000),
    }

    bl_results = {}
    for name, mdl in linears.items():
        mdl.fit(Xtv_s, y_tv)
        pred = mdl.predict(Xte_s)
        m = metrics(y_test, pred)
        m["y_pred"] = pred
        bl_results[name] = m
        alpha_str = ""
        if hasattr(mdl, "alpha_"):
            alpha_str = f"  alpha={mdl.alpha_:.2f}"
        if hasattr(mdl, "l1_ratio_"):
            alpha_str += f" l1_ratio={mdl.l1_ratio_:.2f}"
        print(f"  {name:12s}  RMSE=${m['RMSE']:>10,.0f}  MAE=${m['MAE']:>10,.0f}  R²={m['R2']:.4f}{alpha_str}")

    bl_df = pd.DataFrame({n: {k: v for k, v in r.items() if k != "y_pred"} for n, r in bl_results.items()}).T
    bl_df.to_csv(save("step1_baseline_metrics.csv"))

    best_bl = max(bl_results, key=lambda k: bl_results[k]["R2"])
    print(f"\n  Best baseline: {best_bl}  R²={bl_results[best_bl]['R2']:.4f}")

    # ════════════════════════════════════════════════════════════
    # Step 3: Neural Network Experiments (≥3 architectures)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Step 3: Neural Network Experiments")
    print("=" * 70)

    configs = {
        "A: Shallow [64]": dict(hidden_dims=[64]),
        "B: Medium [128-64]": dict(hidden_dims=[128, 64]),
        "C: Deep [256-128-64]": dict(hidden_dims=[256, 128, 64]),
        "D: Deep+Sigmoid": dict(hidden_dims=[256, 128, 64], activation="sigmoid"),
        "E: Deep+Dropout0.3": dict(hidden_dims=[256, 128, 64], dropout=0.3),
        "F: Deep+L2": dict(hidden_dims=[256, 128, 64], weight_decay=0.01),
    }

    nn_models = {}
    nn_hists = {}
    nn_results = {}
    input_dim = Xtr.shape[1]

    for name, cfg in configs.items():
        print(f"\n  Training {name} ...")
        model, hist = train_nn(cfg, Xtr, Xva, ytr, yva, input_dim)
        pred = to_dollar(nn_predict(model, Xte), sc_y)
        m = metrics(y_test, pred)
        m["y_pred"] = pred

        nn_models[name] = model
        nn_hists[name] = hist
        nn_results[name] = m

        print(f"    epochs={hist['total_epochs']} (best={hist['best_epoch']})  "
              f"time={hist['elapsed']:.1f}s  "
              f"RMSE=${m['RMSE']:,.0f}  R²={m['R2']:.4f}")

    # Save experiment table
    rows = []
    for name, cfg in configs.items():
        h = nn_hists[name]
        r = nn_results[name]
        rows.append({
            "name": name,
            "hidden_dims": str(cfg.get("hidden_dims")),
            "activation": cfg.get("activation", "relu"),
            "dropout": cfg.get("dropout", 0.0),
            "weight_decay": cfg.get("weight_decay", 0.0),
            "best_epoch": h["best_epoch"],
            "total_epochs": h["total_epochs"],
            "train_time_s": round(h["elapsed"], 1),
            "test_RMSE": round(r["RMSE"], 2),
            "test_MAE": round(r["MAE"], 2),
            "test_R2": round(r["R2"], 4),
        })
    pd.DataFrame(rows).to_csv(save("step3_nn_experiments.csv"), index=False)

    # ════════════════════════════════════════════════════════════
    # Step 4: Training Diagnostics
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Step 4: Training Diagnostics")
    print("=" * 70)

    # 4a. Architecture comparison (A vs B vs C)
    arch = {k: nn_hists[k] for k in ["A: Shallow [64]", "B: Medium [128-64]", "C: Deep [256-128-64]"]}
    plot_training_curves(arch, "step4_architecture_curves.png")
    plot_overlay(arch, "Architecture Comparison (Val Loss)", "step4_architecture_overlay.png")
    print("  Saved architecture comparison curves")

    # 4b. Activation: ReLU vs Sigmoid
    act = {k: nn_hists[k] for k in ["C: Deep [256-128-64]", "D: Deep+Sigmoid"]}
    plot_overlay(act, "Activation: ReLU vs Sigmoid (Val Loss)", "step4_activation_comparison.png")
    print("  Saved activation comparison")

    # 4c. Regularization: None vs Dropout vs L2
    reg = {k: nn_hists[k] for k in ["C: Deep [256-128-64]", "E: Deep+Dropout0.3", "F: Deep+L2"]}
    plot_overlay(reg, "Regularization Comparison (Val Loss)", "step4_regularization_comparison.png")
    print("  Saved regularization comparison")

    # ════════════════════════════════════════════════════════════
    # Step 5: Final Comparison Table
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Step 5: All Models Comparison")
    print("=" * 70)

    all_rows = []
    for name, r in bl_results.items():
        all_rows.append({"Model": name, "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"], "Type": "Linear"})
    for name, r in nn_results.items():
        all_rows.append({"Model": name, "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"], "Type": "NN"})

    comp_df = pd.DataFrame(all_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    comp_df.to_csv(save("step5_comparison_table.csv"), index=False)
    print(comp_df.to_string(index=False))

    best_nn_name = max(nn_results, key=lambda k: nn_results[k]["R2"])
    best_nn = nn_results[best_nn_name]
    print(f"\n  Best NN:       {best_nn_name}  R²={best_nn['R2']:.4f}  RMSE=${best_nn['RMSE']:,.0f}")
    print(f"  Best Baseline: {best_bl}  R²={bl_results[best_bl]['R2']:.4f}  RMSE=${bl_results[best_bl]['RMSE']:,.0f}")
    delta = best_nn["R2"] - bl_results[best_bl]["R2"]
    if delta > 0:
        print(f"  => NN wins by R² +{delta:.4f}")
    else:
        print(f"  => Baseline wins by R² +{-delta:.4f}")

    # Diagnostic plots
    plot_diagnostics(y_test, best_nn["y_pred"], f"Best NN: {best_nn_name}", "step5_nn_diagnostics.png")
    plot_diagnostics(y_test, bl_results[best_bl]["y_pred"], f"Best Baseline: {best_bl}", "step5_baseline_diagnostics.png")
    print("  Saved diagnostic plots")

    # ════════════════════════════════════════════════════════════
    # Step 6: Feature Importance & Business Insights
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Step 6: Feature Importance & Business Insights")
    print("=" * 70)

    best_model = nn_models[best_nn_name]
    yte_s = sc_y.transform(y_test.reshape(-1, 1)).ravel()

    print("  Computing permutation importance ...")
    imp = permutation_importance(best_model, Xte, yte_s, sc_y)
    plot_importance(imp, feature_names, "step6_permutation_importance.png")

    imp_sorted = np.argsort(imp)[::-1]
    print("\n  NN Permutation Importance:")
    imp_rows = []
    for rank, idx in enumerate(imp_sorted):
        print(f"    {rank + 1:2d}. {feature_names[idx]:28s}  ${imp[idx]:>10,.0f}")
        imp_rows.append({"rank": rank + 1, "feature": feature_names[idx], "importance_rmse": round(imp[idx], 2)})
    pd.DataFrame(imp_rows).to_csv(save("step6_nn_importance.csv"), index=False)

    # Lasso coefficient comparison
    lasso = linears["Lasso"]
    lc = np.abs(lasso.coef_)
    lc_sorted = np.argsort(lc)[::-1]
    print("\n  Lasso |coef| (for comparison):")
    lasso_rows = []
    for rank, idx in enumerate(lc_sorted):
        print(f"    {rank + 1:2d}. {feature_names[idx]:28s}  {lc[idx]:>10.4f}")
        lasso_rows.append({"rank": rank + 1, "feature": feature_names[idx], "abs_coef": round(lc[idx], 4)})
    pd.DataFrame(lasso_rows).to_csv(save("step6_lasso_importance.csv"), index=False)

    # Worst predictions
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


if __name__ == "__main__":
    main()
