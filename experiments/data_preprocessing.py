import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "AmesHousing.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "neural_network")
os.makedirs(RESULTS_DIR, exist_ok=True)


def feature_engineering(df):
    df["TotalSF"] = (
        df["1st Flr SF"] +
        df["2nd Flr SF"] +
        df["Total Bsmt SF"]
    )

    df["HouseAge"] = df["Yr Sold"] - df["Year Built"]
    df["RemodAge"] = df["Yr Sold"] - df["Year Remod/Add"]

    df["IsRemodeled"] = (df["Year Built"] != df["Year Remod/Add"]).astype(int)

    df["TotalBath"] = (
        df["Full Bath"] +
        0.5 * df["Half Bath"] +
        df["Bsmt Full Bath"] +
        0.5 * df["Bsmt Half Bath"]
    )

    df["TotalRooms"] = df["TotRms AbvGrd"] + df["Bedroom AbvGr"]

    df["AreaPerRoom"] = df["Gr Liv Area"] / (df["TotRms AbvGrd"] + 1)

    df["HasBasement"] = (df["Total Bsmt SF"] > 0).astype(int)
    df["HasGarage"] = (df["Garage Area"] > 0).astype(int)

    df["IsNew"] = (df["Year Built"] == df["Yr Sold"]).astype(int)

    df["Garage Area"] = df["Garage Area"].fillna(0)

    return df


ENGINEERED_FEATURES = [
    "TotalSF", "HouseAge", "RemodAge", "IsRemodeled",
    "TotalBath", "TotalRooms", "AreaPerRoom",
    "HasBasement", "HasGarage", "IsNew",
]


def load_and_preprocess():
    raw_df = pd.read_csv(DATA_PATH)

    raw_df = feature_engineering(raw_df)

    selected_features = [
        "Gr Liv Area", "Overall Qual", "Year Built", "Total Bsmt SF",
        "Garage Cars", "Full Bath", "TotRms AbvGrd", "Neighborhood",
        "Kitchen Qual", "Sale Condition", "Garage Area", "Overall Cond",
        "Fireplaces", "Bsmt Full Bath", "1st Flr SF",
    ] + ENGINEERED_FEATURES

    df = raw_df[selected_features + ["SalePrice"]].copy()

    num_cols = df[selected_features].select_dtypes(include=["number"]).columns
    cat_cols = df[selected_features].select_dtypes(include=["object"]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")

    m1 = (df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 300000)
    m2 = (df["Total Bsmt SF"] > 5000) & (df["SalePrice"] < 300000)
    df = df[~(m1 | m2)].copy()

    quality_map = {"Missing": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    df["Kitchen Qual"] = df["Kitchen Qual"].map(quality_map).astype(int)

    for col in ["Neighborhood", "Sale Condition"]:
        freq = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_frequency"] = df[col].map(freq).fillna(0)
        df = df.drop(columns=col)

    for c in ["Gr Liv Area", "Total Bsmt SF", "Garage Area", "1st Flr SF"]:
        if c in df.columns:
            df[c] = np.log1p(df[c])

    feature_cols = [c for c in df.columns if c != "SalePrice"]
    return df[feature_cols].values, df["SalePrice"].values, feature_cols


def prepare_data():
    X, y, feature_names = load_and_preprocess()

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    sc_X = StandardScaler().fit(X_train)
    sc_y = StandardScaler().fit(y_train.reshape(-1, 1))
    Xtr = sc_X.transform(X_train)
    Xva = sc_X.transform(X_val)
    Xte = sc_X.transform(X_test)
    ytr = sc_y.transform(y_train.reshape(-1, 1)).ravel()
    yva = sc_y.transform(y_val.reshape(-1, 1)).ravel()

    sc_X_full = StandardScaler().fit(X_tv)
    Xtv_s = sc_X_full.transform(X_tv)
    Xte_s = sc_X_full.transform(X_test)

    return {
        "feature_names": feature_names,
        "X_train": Xtr, "X_val": Xva, "X_test": Xte,
        "y_train": ytr, "y_val": yva, "y_test": y_test,
        "Xtv_s": Xtv_s, "Xte_s": Xte_s, "y_tv": y_tv,
        "sc_y": sc_y,
        "n_samples": X.shape[0], "n_features": X.shape[1],
        "n_train": X_train.shape[0], "n_val": X_val.shape[0], "n_test": X_test.shape[0],
    }
