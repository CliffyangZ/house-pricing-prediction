from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "gb_lr_engineering_analysis"
OUT_DIR.mkdir(exist_ok=True)


def setup_plot_style():
    sns.set_theme(style="whitegrid")
    font_path = ROOT / "tutorial" / "ChineseFont.ttf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        mpl.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        mpl.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "STHeiti", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False


def markdown_table(df, floatfmt=",.4f"):
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: format(value, floatfmt))
    rows = []
    headers = [str(col) for col in display_df.columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display_df.iterrows():
        rows.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(rows)


def load_preprocessed_data():
    raw_df = pd.read_csv(ROOT / "practice" / "data" / "AmesHousing.csv")
    selected_features = [
        "Gr Liv Area",
        "Overall Qual",
        "Year Built",
        "Total Bsmt SF",
        "Garage Cars",
        "Full Bath",
        "TotRms AbvGrd",
        "Neighborhood",
        "Kitchen Qual",
        "Sale Condition",
        "Garage Area",
        "Overall Cond",
        "Fireplaces",
        "Bsmt Full Bath",
        "1st Flr SF",
    ]

    clean_df = raw_df[selected_features + ["SalePrice"]].copy()
    numeric_cols = clean_df[selected_features].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = clean_df[selected_features].select_dtypes(include=["object", "str"]).columns.tolist()

    for col in numeric_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())
    for col in categorical_cols:
        clean_df[col] = clean_df[col].fillna("Missing")

    outlier_mask = (
        ((clean_df["Gr Liv Area"] > 4000) & (clean_df["SalePrice"] < 300000))
        | ((clean_df["Total Bsmt SF"] > 5000) & (clean_df["SalePrice"] < 300000))
    )
    clean_df = clean_df.loc[~outlier_mask].copy()

    quality_order = {"Missing": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    clean_df["Kitchen Qual"] = clean_df["Kitchen Qual"].map(quality_order).astype(int)

    for col in ["Neighborhood", "Sale Condition"]:
        frequency_map = clean_df[col].value_counts(normalize=True).to_dict()
        clean_df[f"{col}_frequency"] = clean_df[col].map(frequency_map).fillna(0)
        clean_df = clean_df.drop(columns=col)
        selected_features = [f"{col}_frequency" if feature == col else feature for feature in selected_features]

    numeric_clean_df = clean_df.select_dtypes(include=["number"])
    corr_matrix = numeric_clean_df.corr()
    model_features = corr_matrix["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(15).index.tolist()

    feature_df = clean_df[model_features].copy()
    target = clean_df["SalePrice"].copy()
    for col in ["Gr Liv Area", "Total Bsmt SF", "Garage Area", "1st Flr SF"]:
        if col in feature_df.columns:
            feature_df[col] = np.log1p(feature_df[col])

    return feature_df, target


def inverse_target(scaler_y, values):
    return scaler_y.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def evaluate_model(name, model, X_train, X_test, y_train_scaled, y_test_scaled, y_train, y_test, scaler_y):
    model.fit(X_train, y_train_scaled)
    train_pred_scaled = model.predict(X_train)
    test_pred_scaled = model.predict(X_test)
    train_pred = inverse_target(scaler_y, train_pred_scaled)
    test_pred = inverse_target(scaler_y, test_pred_scaled)
    residuals = y_test.values - test_pred

    train_mse_scaled = mean_squared_error(y_train_scaled, train_pred_scaled)
    test_mse_scaled = mean_squared_error(y_test_scaled, test_pred_scaled)
    train_rmse_scaled = np.sqrt(train_mse_scaled)
    test_rmse_scaled = np.sqrt(test_mse_scaled)
    train_mae_scaled = mean_absolute_error(y_train_scaled, train_pred_scaled)
    test_mae_scaled = mean_absolute_error(y_test_scaled, test_pred_scaled)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    gap_pct = (test_rmse_scaled - train_rmse_scaled) / train_rmse_scaled
    abs_resid_corr = np.corrcoef(test_pred, np.abs(residuals))[0, 1]

    return {
        "model_name": name,
        "model": model,
        "train_pred_scaled": train_pred_scaled,
        "test_pred_scaled": test_pred_scaled,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "residuals": residuals,
        "train_mse_scaled": train_mse_scaled,
        "test_mse_scaled": test_mse_scaled,
        "train_rmse_scaled": train_rmse_scaled,
        "test_rmse_scaled": test_rmse_scaled,
        "train_mae_scaled": train_mae_scaled,
        "test_mae_scaled": test_mae_scaled,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae_original": mean_absolute_error(y_train, train_pred),
        "test_mae_original": mean_absolute_error(y_test, test_pred),
        "gap_pct": gap_pct,
        "abs_resid_corr": abs_resid_corr,
    }


def binned_residual_std(y_pred, residuals, bins=5):
    df = pd.DataFrame({"prediction": y_pred, "abs_residual": np.abs(residuals), "residual": residuals})
    df["prediction_bin"] = pd.qcut(df["prediction"], q=bins, duplicates="drop")
    return (
        df.groupby("prediction_bin", observed=False)
        .agg(
            bin_mid=("prediction", "mean"),
            residual_std=("residual", "std"),
            mean_abs_residual=("abs_residual", "mean"),
            count=("residual", "size"),
        )
        .reset_index(drop=True)
    )


def plot_model_diagnostics(results, y_test):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=160)
    for row, result in enumerate(results):
        name = result["model_name"]
        pred = result["test_pred"]
        residuals = result["residuals"]
        min_price = min(y_test.min(), pred.min())
        max_price = max(y_test.max(), pred.max())

        sns.scatterplot(x=y_test, y=pred, ax=axes[row, 0], alpha=0.55, s=35, color="#2563eb")
        axes[row, 0].plot([min_price, max_price], [min_price, max_price], color="#dc2626", linewidth=2)
        axes[row, 0].set_title(f"{name}: 真實 vs 預測")
        axes[row, 0].set_xlabel("真實 SalePrice")
        axes[row, 0].set_ylabel("預測 SalePrice")

        sns.scatterplot(x=pred, y=residuals, ax=axes[row, 1], alpha=0.55, s=35, color="#16a34a")
        axes[row, 1].axhline(0, color="#dc2626", linewidth=2)
        axes[row, 1].set_title(f"{name}: 殘差圖")
        axes[row, 1].set_xlabel("預測 SalePrice")
        axes[row, 1].set_ylabel("殘差（真實 - 預測）")

        bin_df = binned_residual_std(pred, residuals)
        sns.lineplot(data=bin_df, x="bin_mid", y="residual_std", marker="o", linewidth=2.5, ax=axes[row, 2])
        axes[row, 2].set_title(f"{name}: 異方差檢查")
        axes[row, 2].set_xlabel("預測房價分箱中心")
        axes[row, 2].set_ylabel("分箱殘差標準差")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_model_residual_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=160)
    rmse_long = results_df.melt(id_vars="Model", value_vars=["Train RMSE (scaled)", "Test RMSE (scaled)"], var_name="Split", value_name="RMSE")
    r2_long = results_df.melt(id_vars="Model", value_vars=["Train R²", "Test R²"], var_name="Split", value_name="R²")

    sns.barplot(data=rmse_long, x="Model", y="RMSE", hue="Split", ax=axes[0])
    axes[0].set_title("Step 1: 模型訓練 RMSE")
    axes[0].set_ylabel("RMSE（標準化目標單位）")

    sns.barplot(data=r2_long, x="Model", y="R²", hue="Split", ax=axes[1])
    axes[1].set_title("Step 1: 模型訓練 R²")
    axes[1].set_ylim(0, 1)

    sns.barplot(data=results_df, x="Model", y="Overfit Gap", ax=axes[2], color="#f97316")
    axes[2].set_title("Step 2: Overfitting Gap")
    axes[2].set_ylabel("(Test RMSE - Train RMSE) / Train RMSE")
    axes[2].axhline(0.15, color="#dc2626", linestyle="--", linewidth=2, label="15% 參考線")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_train_test_overfitting_summary.png", bbox_inches="tight")
    plt.close(fig)


def fit_linear_regularization(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test, scaler_y, feature_names):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "OLS": LinearRegression(),
        "Ridge(L2)": RidgeCV(alphas=np.logspace(-3, 4, 120), cv=cv, scoring="neg_root_mean_squared_error"),
        "Lasso(L1)": LassoCV(alphas=np.logspace(-4, 1, 120), cv=cv, random_state=42, max_iter=200000),
        "ElasticNet": ElasticNetCV(
            alphas=np.logspace(-4, 1, 90),
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=cv,
            random_state=42,
            max_iter=200000,
        ),
    }
    rows = []
    coef_rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train_scaled)
        train_pred_scaled = model.predict(X_train_scaled)
        test_pred_scaled = model.predict(X_test_scaled)
        rows.append(
            {
                "Model": name,
                "Train MSE (scaled)": mean_squared_error(y_train_scaled, train_pred_scaled),
                "Test MSE (scaled)": mean_squared_error(y_test_scaled, test_pred_scaled),
                "Train RMSE (scaled)": np.sqrt(mean_squared_error(y_train_scaled, train_pred_scaled)),
                "Test RMSE (scaled)": np.sqrt(mean_squared_error(y_test_scaled, test_pred_scaled)),
                "Train MAE (scaled)": mean_absolute_error(y_train_scaled, train_pred_scaled),
                "Test MAE (scaled)": mean_absolute_error(y_test_scaled, test_pred_scaled),
                "Train R²": r2_score(y_train_scaled, train_pred_scaled),
                "Test R²": r2_score(y_test_scaled, test_pred_scaled),
            }
        )
        for feature, coef in zip(feature_names, model.coef_):
            coef_rows.append({"Model": name, "feature": feature, "coefficient": coef})
        fitted[name] = model

    metrics_df = pd.DataFrame(rows)
    coef_df = pd.DataFrame(coef_rows)
    zeroed_features = coef_df[(coef_df["Model"] == "Lasso(L1)") & (coef_df["coefficient"].abs() < 1e-8)]["feature"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=160)
    metric_long = metrics_df.melt(id_vars="Model", value_vars=["Train RMSE (scaled)", "Test RMSE (scaled)"], var_name="Split", value_name="RMSE")
    sns.barplot(data=metric_long, x="Model", y="RMSE", hue="Split", ax=axes[0])
    axes[0].set_title("Linear Regression 正則化後效能比較")
    axes[0].set_ylabel("RMSE（標準化目標單位）")
    axes[0].tick_params(axis="x", rotation=18)

    coef_plot = coef_df.copy()
    coef_order = (
        coef_plot[coef_plot["Model"] == "OLS"]
        .assign(abs_coef=lambda df: df["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)["feature"]
    )
    sns.barplot(data=coef_plot, y="feature", x="coefficient", hue="Model", order=coef_order, ax=axes[1])
    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set_title("OLS / Ridge / Lasso / ElasticNet 係數比較")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_linear_regularization_comparison.png", bbox_inches="tight")
    plt.close(fig)

    ridge_alpha = fitted["Ridge(L2)"].alpha_
    lasso_alpha = fitted["Lasso(L1)"].alpha_
    enet_alpha = fitted["ElasticNet"].alpha_
    enet_l1_ratio = fitted["ElasticNet"].l1_ratio_

    return metrics_df, coef_df, zeroed_features, ridge_alpha, lasso_alpha, enet_alpha, enet_l1_ratio


def fit_gradient_boosting_regularization(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test, scaler_y, feature_names):
    gb_complex = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=1.0,
        min_samples_leaf=1,
        random_state=42,
    )
    gb_regularized = GradientBoostingRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=8,
        min_samples_split=20,
        validation_fraction=0.15,
        n_iter_no_change=25,
        tol=1e-4,
        random_state=42,
    )

    rows = []
    fitted = {}
    for name, model in {"GB Complex": gb_complex, "GB Regularized": gb_regularized}.items():
        model.fit(X_train_scaled, y_train_scaled)
        train_pred_scaled = model.predict(X_train_scaled)
        test_pred_scaled = model.predict(X_test_scaled)
        rows.append(
            {
                "Model": name,
                "Train MSE (scaled)": mean_squared_error(y_train_scaled, train_pred_scaled),
                "Test MSE (scaled)": mean_squared_error(y_test_scaled, test_pred_scaled),
                "Train RMSE (scaled)": np.sqrt(mean_squared_error(y_train_scaled, train_pred_scaled)),
                "Test RMSE (scaled)": np.sqrt(mean_squared_error(y_test_scaled, test_pred_scaled)),
                "Train MAE (scaled)": mean_absolute_error(y_train_scaled, train_pred_scaled),
                "Test MAE (scaled)": mean_absolute_error(y_test_scaled, test_pred_scaled),
                "Train R²": r2_score(y_train_scaled, train_pred_scaled),
                "Test R²": r2_score(y_test_scaled, test_pred_scaled),
                "Trees Used": model.n_estimators_,
                "Overfit Gap": (
                    np.sqrt(mean_squared_error(y_test_scaled, test_pred_scaled))
                    - np.sqrt(mean_squared_error(y_train_scaled, train_pred_scaled))
                )
                / np.sqrt(mean_squared_error(y_train_scaled, train_pred_scaled)),
            }
        )
        fitted[name] = model

    metrics_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=160)
    rmse_long = metrics_df.melt(id_vars="Model", value_vars=["Train RMSE (scaled)", "Test RMSE (scaled)"], var_name="Split", value_name="RMSE")
    sns.barplot(data=rmse_long, x="Model", y="RMSE", hue="Split", ax=axes[0])
    axes[0].set_title("Gradient Boosting 防過擬合前後 RMSE")
    axes[0].set_ylabel("RMSE（標準化目標單位）")

    for name, model in fitted.items():
        test_loss = [mean_squared_error(y_test_scaled, pred) for pred in model.staged_predict(X_test_scaled)]
        axes[1].plot(np.arange(1, len(test_loss) + 1), test_loss, linewidth=2, label=name)
    axes[1].set_title("Gradient Boosting staged Test MSE")
    axes[1].set_xlabel("Trees / Iterations")
    axes[1].set_ylabel("Test MSE（標準化目標）")
    axes[1].legend()

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": fitted["GB Regularized"].feature_importances_}
    ).sort_values("importance", ascending=False)
    sns.barplot(data=importance_df.head(12), y="feature", x="importance", ax=axes[2], color="#2563eb")
    axes[2].set_title("Regularized GB 特徵重要性")
    axes[2].set_ylabel("")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_gradient_boosting_regularization.png", bbox_inches="tight")
    plt.close(fig)
    return metrics_df, importance_df, fitted


def write_report(base_metrics, linear_metrics, gb_metrics, zeroed_features, ridge_alpha, lasso_alpha, enet_alpha, enet_l1_ratio):
    lines = [
        "# Gradient Boosting 與 Linear Regression 工程分析",
        "",
        "本分析沿用 `practice/housing_price.ipynb` 的資料流程：缺失值處理、outlier 移除、類別編碼、面積類特徵 `log1p`、train/test split、特徵與目標標準化。",
        "指標說明：模型 loss 相關的 `MSE`、`RMSE`、`MAE` 都使用標準化後的目標值計算；殘差圖與真實 vs 預測圖保留原始 SalePrice 尺度，方便判讀房價區間的誤差型態。",
        "",
        "## 流程總覽",
        "",
        "1. 先訓練 baseline model，取得標準化 Train/Test MSE、RMSE、MAE 與 R²。",
        "2. 做殘差分析：看殘差是否隨預測值變大而擴散，作為異方差性線索。",
        "3. 比較 Train/Test gap：若 test error 明顯高於 train error，判斷有 overfitting 風險。",
        "4. Linear Regression 用 L2 / L1 / ElasticNet 在 loss function 加懲罰項。",
        "5. Gradient Boosting 不使用 L1/L2/ElasticNet 係數懲罰，而是用樹模型正則化：限制 `max_depth`、提高 `min_samples_leaf`、降低 `learning_rate`、使用 `subsample`、控制 `n_estimators` 與 early stopping。",
        "",
        "## Step 1: Baseline 訓練結果",
        "",
        markdown_table(base_metrics),
        "",
        "圖表：`02_train_test_overfitting_summary.png`",
        "",
        "## Step 2: 殘差分析與異方差性",
        "",
        "殘差圖若呈現漏斗狀，代表預測值越高，誤差變異越大，這是異方差性的常見訊號。房價資料通常會有這個現象，因為高價房受地段、裝潢、稀有條件影響更大，單一模型較難用同樣誤差尺度描述所有價格區間。",
        "",
        "Linear Regression 的殘差較明顯在高價區擴散，且高價房容易被低估，代表線性假設不足。Gradient Boosting 的殘差較集中，但高價區仍有低估與變異擴大，表示非線性模型改善了 fit，但資料本身仍有高價區不確定性。",
        "",
        "圖表：`01_model_residual_diagnostics.png`",
        "",
        "## Step 3: Overfitting 判斷",
        "",
        "Overfitting 不是只看 train score 高，而是看 train 與 test 的差距。若標準化 Train RMSE 很低但標準化 Test RMSE 明顯較高，模型可能記住訓練資料細節，泛化能力不足。",
        "",
        "在 baseline 中，Gradient Boosting 的 train/test gap 通常比 Linear Regression 更值得注意，因為樹模型容量較大；Linear Regression 容量較低，主要問題通常是 underfit 或線性假設不足，而不一定是 overfit。",
        "",
        "## Step 4A: Linear Regression 正則化",
        "",
        f"Ridge(L2) 最佳 λ：`{ridge_alpha:.6g}`。L2 懲罰項會把係數往 0 收縮，降低共線性造成的係數不穩定，但通常不做特徵淘汰。",
        f"Lasso(L1) 最佳 λ：`{lasso_alpha:.6g}`。L1 懲罰項可能把係數壓成 0，因此可做特徵選擇。",
        f"ElasticNet 最佳 λ：`{enet_alpha:.6g}`，最佳 `l1_ratio`：`{enet_l1_ratio:.2f}`。ElasticNet 混合 L1 與 L2，適合特徵相關性高但又希望有特徵選擇的情境。",
        "",
        "Linear Regression 正則化比較：",
        "",
        markdown_table(linear_metrics),
        "",
        "Lasso 歸零特徵：",
        "",
        "\n".join([f"- `{feature}`" for feature in zeroed_features]) if zeroed_features else "無特徵被歸零。這代表在目前 CV 選出的懲罰強度下，15 個特徵仍都有保留價值，或懲罰不需要強到淘汰特徵。",
        "",
        "圖表：`03_linear_regularization_comparison.png`",
        "",
        "## Step 4B: Gradient Boosting 防過擬合處理",
        "",
        "Gradient Boosting 的 loss 是逐棵樹加總下降；過多樹、太深的樹、葉節點樣本太少，都會讓模型貼合訓練資料噪音。這裡用較保守的樹深、較小 learning rate、subsample 與 early stopping 來正則化。",
        "",
        "Gradient Boosting 調整前後：",
        "",
        markdown_table(gb_metrics),
        "",
        "圖表：`04_gradient_boosting_regularization.png`",
        "",
        "## 結論",
        "",
        "Linear Regression 的核心診斷是線性假設與殘差結構；正則化能穩定係數與降低 overfit，但無法解決明顯非線性。Gradient Boosting 對非線性房價關係較有效，但需要用樹模型正則化與 early stopping 控制泛化誤差。",
    ]
    (OUT_DIR / "engineering_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    setup_plot_style()
    X, y = load_preprocessed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    baseline_models = [
        (
            "Linear Regression",
            LinearRegression(),
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.04,
                max_depth=5,
                subsample=0.8,
                min_samples_split=5,
                random_state=42,
            ),
        ),
    ]

    baseline_results = [
        evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test, scaler_y)
        for name, model in baseline_models
    ]
    base_metrics = pd.DataFrame(
        [
            {
                "Model": result["model_name"],
                "Train MSE (scaled)": result["train_mse_scaled"],
                "Test MSE (scaled)": result["test_mse_scaled"],
                "Train RMSE (scaled)": result["train_rmse_scaled"],
                "Test RMSE (scaled)": result["test_rmse_scaled"],
                "Train MAE (scaled)": result["train_mae_scaled"],
                "Test MAE (scaled)": result["test_mae_scaled"],
                "Train R²": result["train_r2"],
                "Test R²": result["test_r2"],
                "Overfit Gap": result["gap_pct"],
                "Abs Residual Corr": result["abs_resid_corr"],
            }
            for result in baseline_results
        ]
    )

    plot_model_diagnostics(baseline_results, y_test)
    plot_metric_summary(base_metrics)

    linear_metrics, linear_coef, zeroed_features, ridge_alpha, lasso_alpha, enet_alpha, enet_l1_ratio = fit_linear_regularization(
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test, scaler_y, X.columns.tolist()
    )
    gb_metrics, gb_importance, _ = fit_gradient_boosting_regularization(
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test, scaler_y, X.columns.tolist()
    )

    base_metrics.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)
    linear_metrics.to_csv(OUT_DIR / "linear_regularization_metrics.csv", index=False)
    linear_coef.to_csv(OUT_DIR / "linear_regularization_coefficients.csv", index=False)
    gb_metrics.to_csv(OUT_DIR / "gradient_boosting_regularization_metrics.csv", index=False)
    gb_importance.to_csv(OUT_DIR / "gradient_boosting_feature_importance.csv", index=False)

    write_report(base_metrics, linear_metrics, gb_metrics, zeroed_features, ridge_alpha, lasso_alpha, enet_alpha, enet_l1_ratio)

    print("Baseline metrics")
    print(base_metrics.to_string(index=False))
    print("\nLinear regularization metrics")
    print(linear_metrics.to_string(index=False))
    print("\nGradient Boosting regularization metrics")
    print(gb_metrics.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
