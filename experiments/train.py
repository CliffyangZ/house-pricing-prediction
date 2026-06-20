#!/usr/bin/env python3
"""
Ames Housing: Neural Network vs Regularized Linear Models

Usage:
    uv run python experiments/train.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import RANDOM_STATE, prepare_data
from evaluation import evaluate_all, metrics, nn_predict, save, to_dollar
from mlp import MLP

BATCH_SIZE = 64
MAX_EPOCHS = 500
EARLY_STOP_PATIENCE = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


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


def train_baselines(data):
    linears = {
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(cv=5),
        "Lasso": LassoCV(cv=5, max_iter=10000),
        "ElasticNet": ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0], max_iter=10000),
    }

    bl_results = {}
    for name, mdl in linears.items():
        mdl.fit(data["Xtv_s"], data["y_tv"])
        pred = mdl.predict(data["Xte_s"])
        m = metrics(data["y_test"], pred)
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

    return linears, bl_results


def train_neural_networks(data):
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
    input_dim = data["X_train"].shape[1]

    for name, cfg in configs.items():
        print(f"\n  Training {name} ...")
        model, hist = train_nn(cfg, data["X_train"], data["X_val"], data["y_train"], data["y_val"], input_dim)
        pred = to_dollar(nn_predict(model, data["X_test"]), data["sc_y"])
        m = metrics(data["y_test"], pred)

        nn_models[name] = model
        nn_hists[name] = hist

        print(f"    epochs={hist['total_epochs']} (best={hist['best_epoch']})  "
              f"time={hist['elapsed']:.1f}s  "
              f"RMSE=${m['RMSE']:,.0f}  R²={m['R2']:.4f}")

    rows = []
    for name, cfg in configs.items():
        h = nn_hists[name]
        pred = to_dollar(nn_predict(nn_models[name], data["X_test"]), data["sc_y"])
        r = metrics(data["y_test"], pred)
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

    return nn_models, nn_hists, configs


def main():
    print("=" * 70)
    print("  Ames Housing: Neural Network vs Regularized Linear Models")
    print("=" * 70)

    data = prepare_data()
    print(f"\nDataset: {data['n_samples']} samples × {data['n_features']} features")
    print(f"Split: train={data['n_train']}, val={data['n_val']}, test={data['n_test']}")

    print("\n" + "=" * 70)
    print("  Step 1: Baseline (OLS / Ridge / Lasso / ElasticNet)")
    print("=" * 70)
    linears, bl_results = train_baselines(data)

    print("\n" + "=" * 70)
    print("  Step 3: Neural Network Experiments")
    print("=" * 70)
    nn_models, nn_hists, nn_configs = train_neural_networks(data)

    evaluate_all(data, nn_models, nn_hists, nn_configs, linears, bl_results)


if __name__ == "__main__":
    main()
