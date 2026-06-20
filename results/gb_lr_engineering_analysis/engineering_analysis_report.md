# Gradient Boosting 與 Linear Regression 工程分析

本分析沿用 `practice/housing_price.ipynb` 的資料流程：缺失值處理、outlier 移除、類別編碼、面積類特徵 `log1p`、train/test split、特徵與目標標準化。
指標說明：模型 loss 相關的 `MSE`、`RMSE`、`MAE` 都使用標準化後的目標值計算；殘差圖與真實 vs 預測圖保留原始 SalePrice 尺度，方便判讀房價區間的誤差型態。

## 流程總覽

1. 先訓練 baseline model，取得標準化 Train/Test MSE、RMSE、MAE 與 R²。
2. 做殘差分析：看殘差是否隨預測值變大而擴散，作為異方差性線索。
3. 比較 Train/Test gap：若 test error 明顯高於 train error，判斷有 overfitting 風險。
4. Linear Regression 用 L2 / L1 / ElasticNet 在 loss function 加懲罰項。
5. Gradient Boosting 不使用 L1/L2/ElasticNet 係數懲罰，而是用樹模型正則化：限制 `max_depth`、提高 `min_samples_leaf`、降低 `learning_rate`、使用 `subsample`、控制 `n_estimators` 與 early stopping。

## Step 1: Baseline 訓練結果

| Model | Train MSE (scaled) | Test MSE (scaled) | Train RMSE (scaled) | Test RMSE (scaled) | Train MAE (scaled) | Test MAE (scaled) | Train R² | Test R² | Overfit Gap | Abs Residual Corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.1835 | 0.2508 | 0.4283 | 0.5008 | 0.2985 | 0.3195 | 0.8165 | 0.8144 | 0.1691 | 0.4136 |
| Gradient Boosting | 0.0200 | 0.0778 | 0.1415 | 0.2789 | 0.1087 | 0.1894 | 0.9800 | 0.9424 | 0.9717 | 0.5193 |

圖表：`02_train_test_overfitting_summary.png`

## Step 2: 殘差分析與異方差性

殘差圖若呈現漏斗狀，代表預測值越高，誤差變異越大，這是異方差性的常見訊號。房價資料通常會有這個現象，因為高價房受地段、裝潢、稀有條件影響更大，單一模型較難用同樣誤差尺度描述所有價格區間。

Linear Regression 的殘差較明顯在高價區擴散，且高價房容易被低估，代表線性假設不足。Gradient Boosting 的殘差較集中，但高價區仍有低估與變異擴大，表示非線性模型改善了 fit，但資料本身仍有高價區不確定性。

圖表：`01_model_residual_diagnostics.png`

## Step 3: Overfitting 判斷

Overfitting 不是只看 train score 高，而是看 train 與 test 的差距。若標準化 Train RMSE 很低但標準化 Test RMSE 明顯較高，模型可能記住訓練資料細節，泛化能力不足。

在 baseline 中，Gradient Boosting 的 train/test gap 通常比 Linear Regression 更值得注意，因為樹模型容量較大；Linear Regression 容量較低，主要問題通常是 underfit 或線性假設不足，而不一定是 overfit。

## Step 4A: Linear Regression 正則化

Ridge(L2) 最佳 λ：`38.7468`。L2 懲罰項會把係數往 0 收縮，降低共線性造成的係數不穩定，但通常不做特徵淘汰。
Lasso(L1) 最佳 λ：`0.000925522`。L1 懲罰項可能把係數壓成 0，因此可做特徵選擇。
ElasticNet 最佳 λ：`0.0092532`，最佳 `l1_ratio`：`0.10`。ElasticNet 混合 L1 與 L2，適合特徵相關性高但又希望有特徵選擇的情境。

Linear Regression 正則化比較：

| Model | Train MSE (scaled) | Test MSE (scaled) | Train RMSE (scaled) | Test RMSE (scaled) | Train MAE (scaled) | Test MAE (scaled) | Train R² | Test R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | 0.1835 | 0.2508 | 0.4283 | 0.5008 | 0.2985 | 0.3195 | 0.8165 | 0.8144 |
| Ridge(L2) | 0.1836 | 0.2518 | 0.4284 | 0.5018 | 0.2979 | 0.3189 | 0.8164 | 0.8136 |
| Lasso(L1) | 0.1835 | 0.2513 | 0.4284 | 0.5013 | 0.2983 | 0.3194 | 0.8165 | 0.8140 |
| ElasticNet | 0.1835 | 0.2518 | 0.4284 | 0.5018 | 0.2980 | 0.3192 | 0.8165 | 0.8136 |

Lasso 歸零特徵：

無特徵被歸零。這代表在目前 CV 選出的懲罰強度下，15 個特徵仍都有保留價值，或懲罰不需要強到淘汰特徵。

圖表：`03_linear_regularization_comparison.png`

## Step 4B: Gradient Boosting 防過擬合處理

Gradient Boosting 的 loss 是逐棵樹加總下降；過多樹、太深的樹、葉節點樣本太少，都會讓模型貼合訓練資料噪音。這裡用較保守的樹深、較小 learning rate、subsample 與 early stopping 來正則化。

Gradient Boosting 調整前後：

| Model | Train MSE (scaled) | Test MSE (scaled) | Train RMSE (scaled) | Test RMSE (scaled) | Train MAE (scaled) | Test MAE (scaled) | Train R² | Test R² | Trees Used | Overfit Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GB Complex | 0.0111 | 0.0784 | 0.1052 | 0.2799 | 0.0801 | 0.1901 | 0.9889 | 0.9420 | 500 | 1.6612 |
| GB Regularized | 0.0585 | 0.0864 | 0.2420 | 0.2939 | 0.1677 | 0.1972 | 0.9415 | 0.9361 | 397 | 0.2146 |

圖表：`04_gradient_boosting_regularization.png`

## 結論

Linear Regression 的核心診斷是線性假設與殘差結構；正則化能穩定係數與降低 overfit，但無法解決明顯非線性。Gradient Boosting 對非線性房價關係較有效，但需要用樹模型正則化與 early stopping 控制泛化誤差。