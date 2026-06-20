from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "regularized_linear_models"
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

    return feature_df, target, clean_df


def evaluate_original_scale(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
        "test_pred": test_pred,
        "model": model,
    }


def calculate_vif(X):
    rows = []
    for col in X.columns:
        y_col = X[col]
        X_other = X.drop(columns=col)
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_other, y_col)
        r2 = pipe.score(X_other, y_col)
        vif = np.inf if r2 >= 0.999999 else 1 / (1 - r2)
        rows.append({"feature": col, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def plot_metric_comparison(results):
    metrics_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Train RMSE": result["train_rmse"],
                "Test RMSE": result["test_rmse"],
                "Train R²": result["train_r2"],
                "Test R²": result["test_r2"],
            }
            for name, result in results.items()
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    rmse_long = metrics_df.melt(id_vars="Model", value_vars=["Train RMSE", "Test RMSE"], var_name="Split", value_name="RMSE")
    r2_long = metrics_df.melt(id_vars="Model", value_vars=["Train R²", "Test R²"], var_name="Split", value_name="R²")

    sns.barplot(data=rmse_long, x="Model", y="RMSE", hue="Split", ax=axes[0])
    axes[0].set_title("Train/Test RMSE 比較")
    axes[0].set_ylabel("RMSE（美元）")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=r2_long, x="Model", y="R²", hue="Split", ax=axes[1])
    axes[1].set_title("Train/Test R² 比較")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return metrics_df


def plot_ols_diagnostics(y_test, ols_pred):
    residuals = y_test - ols_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    sns.scatterplot(x=ols_pred, y=residuals, ax=axes[0], alpha=0.55, s=42, color="#2563eb")
    axes[0].axhline(0, color="#dc2626", linewidth=2)
    axes[0].set_title("OLS 殘差圖：預測值 vs 殘差")
    axes[0].set_xlabel("預測 SalePrice")
    axes[0].set_ylabel("殘差（真實 - 預測）")

    sns.scatterplot(x=y_test, y=ols_pred, ax=axes[1], alpha=0.55, s=42, color="#16a34a")
    min_price = min(y_test.min(), ols_pred.min())
    max_price = max(y_test.max(), ols_pred.max())
    axes[1].plot([min_price, max_price], [min_price, max_price], color="#dc2626", linewidth=2)
    axes[1].set_title("OLS：真實房價 vs 預測房價")
    axes[1].set_xlabel("真實 SalePrice")
    axes[1].set_ylabel("預測 SalePrice")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ols_residuals_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)


def plot_ridge_coefficients(X_train, y_train, feature_names, ridge_alpha, ols_model, ridge_model):
    alphas = np.logspace(-3, 5, 80)
    coefs = []
    for alpha in alphas:
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        ridge.fit(X_train, y_train)
        coefs.append(ridge.named_steps["ridge"].coef_)
    coefs = np.array(coefs)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=160)
    for idx, feature in enumerate(feature_names):
        axes[0].plot(alphas, coefs[:, idx], linewidth=1.6, label=feature)
    axes[0].axvline(ridge_alpha, color="#111827", linestyle="--", linewidth=2, label=f"最佳 λ={ridge_alpha:.4g}")
    axes[0].set_xscale("log")
    axes[0].set_title("Ridge 係數收縮路徑")
    axes[0].set_xlabel("λ / alpha（log scale）")
    axes[0].set_ylabel("標準化係數")
    axes[0].legend(fontsize=7, ncol=2)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "OLS": ols_model.named_steps["linearregression"].coef_,
            "Ridge": ridge_model.named_steps["ridge"].coef_,
        }
    )
    coef_long = coef_df.melt(id_vars="feature", var_name="Model", value_name="coefficient")
    order = coef_df.assign(abs_ols=coef_df["OLS"].abs()).sort_values("abs_ols", ascending=False)["feature"]
    sns.barplot(data=coef_long, y="feature", x="coefficient", hue="Model", order=order, ax=axes[1])
    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set_title("OLS vs Ridge 係數比較")
    axes[1].set_xlabel("標準化係數")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ridge_coefficient_shrinkage.png", bbox_inches="tight")
    plt.close(fig)
    return coef_df


def plot_lasso_coefficients(feature_names, lasso_model):
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": lasso_model.named_steps["lassocv"].coef_}
    )
    coef_df["selected"] = np.where(coef_df["coefficient"].abs() > 1e-8, "保留", "歸零淘汰")
    coef_df = coef_df.sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    sns.barplot(data=coef_df, y="feature", x="coefficient", hue="selected", dodge=False, ax=ax)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Lasso 係數：L1 正則化會把部分特徵壓成 0")
    ax.set_xlabel("標準化係數")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lasso_zeroed_features.png", bbox_inches="tight")
    plt.close(fig)
    return coef_df


def plot_elasticnet_cv(enet_cv):
    fitted_cv = enet_cv.named_steps["elasticnetcv"]
    mse_path = fitted_cv.mse_path_
    l1_ratios = np.atleast_1d(fitted_cv.l1_ratio)
    alphas = fitted_cv.alphas_

    mean_mse = mse_path.mean(axis=2)
    cv_rmse = np.sqrt(mean_mse)
    heatmap_df = pd.DataFrame(cv_rmse, index=[f"{ratio:.2f}" for ratio in l1_ratios], columns=alphas)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap="viridis_r",
        cbar_kws={"label": "CV RMSE（美元）"},
        xticklabels=10,
        yticklabels=True,
    )
    ax.set_title("ElasticNet CV RMSE：λ 與 l1_ratio 搜尋")
    ax.set_xlabel("λ / alpha（由大到小）")
    ax.set_ylabel("l1_ratio（0=Ridge, 1=Lasso）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "elasticnet_cv_rmse_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def markdown_table(df, floatfmt=".4f"):
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


def write_report(metrics_df, ridge_alpha, lasso_alpha, lasso_coef_df, enet_alpha, enet_l1_ratio, enet_cv_rmse, vif_df, ridge_coef_df):
    zeroed = lasso_coef_df.loc[lasso_coef_df["selected"] == "歸零淘汰", "feature"].tolist()
    ridge_coef_df = ridge_coef_df.copy()
    ridge_coef_df["shrink_pct"] = 1 - (ridge_coef_df["Ridge"].abs() / ridge_coef_df["OLS"].abs().replace(0, np.nan))
    top_shrink = ridge_coef_df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("shrink_pct", ascending=False).head(5)

    report = [
        "# 線性回歸正則化分析",
        "",
        "資料與前處理沿用 `practice/housing_price.ipynb`：缺失值補值、刪除極端 outlier、類別特徵轉數值、面積類特徵 `log1p`、`train_test_split(random_state=42)`。",
        "",
        "## 模型比較",
        "",
        markdown_table(metrics_df, ",.4f"),
        "",
        "## 線性迴歸（OLS）",
        "",
        "OLS 作為 baseline，完全不限制係數大小，因此可直接觀察線性模型在目前特徵工程下的表現。殘差圖顯示高價房區域的殘差擴散較大，且高價物件容易被低估，代表線性假設與固定誤差變異不完全成立。",
        "",
        "圖表：`ols_residuals_actual_vs_predicted.png`",
        "",
        "## Ridge（L2）",
        "",
        f"最佳 λ（RidgeCV）：`{ridge_alpha:.6g}`。Ridge 使用 L2 懲罰讓係數向 0 收縮，但通常不會把係數變成 0。",
        "",
        "係數收縮最大的特徵：",
        "",
        markdown_table(top_shrink[["feature", "OLS", "Ridge", "shrink_pct"]], ",.4f"),
        "",
        "多重共線性檢查（VIF 最高）：",
        "",
        markdown_table(vif_df.head(8), ",.4f"),
        "",
        "Ridge 不會讓特徵之間的共線性消失，因為共線性是資料本身的問題；它改善的是共線性造成的係數不穩定，讓相關特徵的權重較不容易暴衝。",
        "",
        "圖表：`ridge_coefficient_shrinkage.png`",
        "",
        "## Lasso（L1）",
        "",
        f"最佳 λ（LassoCV）：`{lasso_alpha:.6g}`。Lasso 使用 L1 懲罰，可把弱訊號或被其他特徵替代的係數壓成 0，達到特徵選擇效果。",
        "",
        "被歸零淘汰的特徵：",
        "",
        "\n".join([f"- `{feature}`" for feature in zeroed]) if zeroed else "無特徵被歸零。",
        "",
        "若被淘汰的是與面積、品質高度重疊的特徵，通常代表它們對 SalePrice 的額外解釋力有限；業務上不一定代表該特徵完全不重要，而是在線性模型與目前特徵組合下資訊已被其他變數吸收。",
        "",
        "圖表：`lasso_zeroed_features.png`",
        "",
        "## ElasticNet",
        "",
        f"最佳 λ：`{enet_alpha:.6g}`；最佳 `l1_ratio`：`{enet_l1_ratio:.2f}`；CV RMSE：`${enet_cv_rmse:,.0f}`。",
        "",
        "ElasticNet 同時混合 Ridge 的穩定係數與 Lasso 的特徵選擇。當資料有多重共線性，而且又希望淘汰部分弱特徵時，ElasticNet 通常比單純 Ridge 或 Lasso 更穩健。若 Ridge 和 Lasso 的 test RMSE 很接近，ElasticNet 的價值主要在於折衷可解釋性與穩定性。",
        "",
        "圖表：`elasticnet_cv_rmse_heatmap.png`",
    ]
    (OUT_DIR / "regularized_linear_models_report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    setup_plot_style()
    X, y, _ = load_preprocessed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_alphas = np.logspace(-3, 5, 120)
    lasso_alphas = np.logspace(1, 5, 120)
    elastic_alphas = np.logspace(1, 5, 80)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

    models = {
        "OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ridge_alphas, cv=cv, scoring="neg_root_mean_squared_error")),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=lasso_alphas, cv=cv, random_state=42, max_iter=200000)),
        "ElasticNet": make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                alphas=elastic_alphas,
                l1_ratio=l1_ratios,
                cv=cv,
                random_state=42,
                max_iter=200000,
            ),
        ),
    }

    results = {name: evaluate_original_scale(model, X_train, X_test, y_train, y_test) for name, model in models.items()}
    metrics_df = plot_metric_comparison(results)
    plot_ols_diagnostics(y_test, results["OLS"]["test_pred"])

    ridge_alpha = results["Ridge"]["model"].named_steps["ridgecv"].alpha_
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)).fit(X_train, y_train)
    ridge_coef_df = plot_ridge_coefficients(X_train, y_train, X.columns.tolist(), ridge_alpha, results["OLS"]["model"], ridge_model)

    lasso_model = results["Lasso"]["model"]
    lasso_alpha = lasso_model.named_steps["lassocv"].alpha_
    lasso_coef_df = plot_lasso_coefficients(X.columns.tolist(), lasso_model)

    enet_model = results["ElasticNet"]["model"]
    enet_alpha = enet_model.named_steps["elasticnetcv"].alpha_
    enet_l1_ratio = enet_model.named_steps["elasticnetcv"].l1_ratio_
    enet_cv_rmse = np.sqrt(enet_model.named_steps["elasticnetcv"].mse_path_.mean(axis=2).min())
    plot_elasticnet_cv(enet_model)

    cv_rmse = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        cv_rmse[name] = -scores.mean()
    metrics_df["CV RMSE"] = metrics_df["Model"].map(cv_rmse)
    metrics_df.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    vif_df = calculate_vif(X_train)
    vif_df.to_csv(OUT_DIR / "vif.csv", index=False)
    lasso_coef_df.to_csv(OUT_DIR / "lasso_coefficients.csv", index=False)
    ridge_coef_df.to_csv(OUT_DIR / "ridge_vs_ols_coefficients.csv", index=False)
    write_report(metrics_df, ridge_alpha, lasso_alpha, lasso_coef_df, enet_alpha, enet_l1_ratio, enet_cv_rmse, vif_df, ridge_coef_df)

    print(metrics_df.to_string(index=False))
    print(f"Ridge best lambda: {ridge_alpha:.6g}")
    print(f"Lasso best lambda: {lasso_alpha:.6g}")
    print(f"ElasticNet best lambda: {enet_alpha:.6g}, l1_ratio: {enet_l1_ratio:.2f}, CV RMSE: {enet_cv_rmse:,.0f}")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
