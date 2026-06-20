# 線性回歸正則化分析

資料與前處理沿用 `practice/housing_price.ipynb`：缺失值補值、刪除極端 outlier、類別特徵轉數值、面積類特徵 `log1p`、`train_test_split(random_state=42)`。

## 模型比較

| Model | Train RMSE | Test RMSE | Train R² | Test R² | CV RMSE |
| --- | --- | --- | --- | --- | --- |
| OLS | 33,084.5448 | 38,680.7538 | 0.8165 | 0.8144 | 33,393.0934 |
| Ridge | 33,090.7468 | 38,754.2662 | 0.8165 | 0.8136 | 33,379.9114 |
| Lasso | 33,086.1524 | 38,716.5651 | 0.8165 | 0.8140 | 33,390.6855 |
| ElasticNet | 33,086.3111 | 38,718.3754 | 0.8165 | 0.8140 | 33,390.1850 |

## 線性迴歸（OLS）

OLS 作為 baseline，完全不限制係數大小，因此可直接觀察線性模型在目前特徵工程下的表現。殘差圖顯示高價房區域的殘差擴散較大，且高價物件容易被低估，代表線性假設與固定誤差變異不完全成立。

圖表：`ols_residuals_actual_vs_predicted.png`

## Ridge（L2）

最佳 λ（RidgeCV）：`37.2759`。Ridge 使用 L2 懲罰讓係數向 0 收縮，但通常不會把係數變成 0。

係數收縮最大的特徵：

| feature | OLS | Ridge | shrink_pct |
| --- | --- | --- | --- |
| Full Bath | -1,630.4913 | -1,134.7941 | 0.3040 |
| Garage Area | -4,701.7756 | -4,219.0831 | 0.1027 |
| Gr Liv Area | 16,967.6752 | 16,265.7396 | 0.0414 |
| Garage Cars | 11,410.7854 | 11,040.0330 | 0.0325 |
| Overall Cond | 5,581.4276 | 5,411.9444 | 0.0304 |

多重共線性檢查（VIF 最高）：

| feature | vif |
| --- | --- |
| Gr Liv Area | 5.1799 |
| Garage Cars | 3.4398 |
| Overall Qual | 3.1563 |
| TotRms AbvGrd | 3.0759 |
| Year Built | 2.3985 |
| Garage Area | 2.2874 |
| Full Bath | 2.2385 |
| Kitchen Qual | 1.9855 |

Ridge 不會讓特徵之間的共線性消失，因為共線性是資料本身的問題；它改善的是共線性造成的係數不穩定，讓相關特徵的權重較不容易暴衝。

圖表：`ridge_coefficient_shrinkage.png`

## Lasso（L1）

最佳 λ（LassoCV）：`69.2367`。Lasso 使用 L1 懲罰，可把弱訊號或被其他特徵替代的係數壓成 0，達到特徵選擇效果。

被歸零淘汰的特徵：

無特徵被歸零。

若被淘汰的是與面積、品質高度重疊的特徵，通常代表它們對 SalePrice 的額外解釋力有限；業務上不一定代表該特徵完全不重要，而是在線性模型與目前特徵組合下資訊已被其他變數吸收。

圖表：`lasso_zeroed_features.png`

## ElasticNet

最佳 λ：`72.5704`；最佳 `l1_ratio`：`1.00`；CV RMSE：`$33,500`。

ElasticNet 同時混合 Ridge 的穩定係數與 Lasso 的特徵選擇。當資料有多重共線性，而且又希望淘汰部分弱特徵時，ElasticNet 通常比單純 Ridge 或 Lasso 更穩健。若 Ridge 和 Lasso 的 test RMSE 很接近，ElasticNet 的價值主要在於折衷可解釋性與穩定性。

圖表：`elasticnet_cv_rmse_heatmap.png`