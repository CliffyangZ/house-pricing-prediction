# House Pricing Prediction

Ames Housing 房價預測 — 比較正則化線性模型與神經網路在小型表格資料上的表現。

> 模組基本分為 `data/`、`experiments/`、`results/`
> 在 `experiments/` 開發，結果 output 到 `results/` 中。

```
.
├── data/AmesHousing.csv          # Ames Housing 原始資料（2,930 筆）
├── experiments/
│   ├── mlp.py                    # PyTorch MLP 模型定義
│   └── train.py                  # 主實驗腳本（涵蓋六大步驟）
├── results/
│   ├── regularized_linear_models/  # 期中：正則化線性模型結果
│   └── neural_network/             # 期末：神經網路實驗結果
└── housing_price_demo.ipynb      # EDA 與特徵工程探索
```

執行方式：

```bash
uv sync
uv run python experiments/train.py
```

---

## 1. 資料前處理與特徵工程

### 1.1 候選特徵（15 個）

從 82 個原始欄位中，依據領域知識篩選出 15 個與房價最相關的特徵：

| 類型 | 特徵 |
|------|------|
| 面積 | Gr Liv Area, Total Bsmt SF, Garage Area, 1st Flr SF |
| 品質/狀態 | Overall Qual, Overall Cond, Kitchen Qual |
| 結構 | Garage Cars, Full Bath, Bsmt Full Bath, TotRms AbvGrd, Fireplaces |
| 時間 | Year Built |
| 類別 | Neighborhood, Sale Condition |

### 1.2 前處理流程

| 步驟 | 處理方式 |
|------|----------|
| 缺失值 | 數值欄位用中位數填充；類別欄位填入 `"Missing"` |
| Outlier | 刪除面積極大但售價偏低的不合理樣本（Gr Liv Area > 4000 且 SalePrice < 300k 等） |
| 類別編碼 | Kitchen Qual → Ordinal Encoding（Po=1 ~ Ex=5）；Neighborhood、Sale Condition → Frequency Encoding |
| 偏態修正 | 面積類特徵（Gr Liv Area, Total Bsmt SF, Garage Area, 1st Flr SF）做 `log1p` 轉換 |
| 特徵選擇 | 以 `abs(corr)` 排序取 top 15（即全部候選特徵） |

### 1.3 資料切割與縮放

| 項目 | 說明 |
|------|------|
| Train / Test | `train_test_split(test_size=0.2, random_state=42)`，與期中完全相同 |
| Validation | 僅從 Train 再切出 20%，用於 Early Stopping，不動用 Test |
| 最終比例 | Train 1,872 / Val 469 / Test 586 |
| 特徵縮放 | `StandardScaler` 只在 Train 上 `fit`，避免 Data Leakage |
| 目標變數 | 線性模型用原始 SalePrice；神經網路用 StandardScaler 標準化後訓練，評估時 inverse_transform 回美元 |

---

## 2. 實驗模型比較

總共比較 **10 個模型**：4 個線性 baseline + 6 個神經網路配置。

### 2.1 Baseline：正則化線性模型（期中）

| Model | Test RMSE ($) | Test R² |
|-------|------------:|--------:|
| OLS | 38,681 | 0.8144 |
| Ridge (CV) | 38,700 | 0.8142 |
| Lasso (CV) | 38,713 | 0.8140 |
| ElasticNet (CV) | 38,713 | 0.8140 |

### 2.2 神經網路實驗（期末）

6 組實驗覆蓋四個比較維度：

| 比較維度 | 實驗組合 |
|----------|----------|
| 網路深度與寬度 | A vs B vs C（淺窄 → 深寬） |
| 啟動函數 | C (ReLU) vs D (Sigmoid) |
| 正則化手段 | C (無) vs E (Dropout 0.3) vs F (L2 weight_decay=0.01) |

| Config | Architecture | Activation | Regularization | Best Epoch | Test RMSE ($) | Test R² |
|--------|-------------|-----------|----------------|----------:|-------------:|--------:|
| A | [64] | ReLU | — | 164 | 22,324 | 0.9382 |
| B | [128, 64] | ReLU | — | 33 | 22,651 | 0.9363 |
| C | [256, 128, 64] | ReLU | — | 21 | 22,235 | 0.9387 |
| D | [256, 128, 64] | Sigmoid | — | 183 | **21,614** | **0.9420** |
| E | [256, 128, 64] | ReLU | Dropout 0.3 | 34 | 21,931 | 0.9403 |
| F | [256, 128, 64] | ReLU | L2 (0.01) | 134 | 23,378 | 0.9322 |

### 2.3 結論

- 所有 NN 配置皆大幅超越線性 baseline（R² 提升約 +0.12）
- 最佳模型為 **D: Deep+Sigmoid**（R²=0.9420, RMSE=$21,614），但收斂速度較慢（183 epochs vs ReLU 的 21 epochs）
- 特徵重要性方面，NN Permutation Importance 與 Lasso 係數排名大致一致，Gr Liv Area、Overall Qual、Year Built 穩居前三
