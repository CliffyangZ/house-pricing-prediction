import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "AmesHousing.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def feature_engineering(df):
    # ======================
    # 🔥 Core Features
    # ======================

    df["TotalSF"] = (
        df["1stFlrSF"] +
        df["2ndFlrSF"] +
        df["TotalBsmtSF"]
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    # ======================
    # 🔥 Interaction Features
    # ======================

    df["TotalBath"] = (
        df["FullBath"] +
        0.5 * df["HalfBath"] +
        df["BsmtFullBath"] +
        0.5 * df["BsmtHalfBath"]
    )

    df["TotalRooms"] = df["TotRmsAbvGrd"] + df["BedroomAbvGr"]

    df["AreaPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)

    # ======================
    # 🔥 Binary Features
    # ======================

    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

    df["IsNew"] = (df["YearBuilt"] == df["YrSold"]).astype(int)

    # ======================
    # 🔥 Missing safety
    # ======================

    df["GarageArea"] = df["GarageArea"].fillna(0)

    return df


def preprocess(df):
    # ======================
    # 1. Feature Engineering
    # ======================
    df = feature_engineering(df)

    # ======================
    # 2. Missing values
    # ======================

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # ======================
    # 3. One-hot encoding
    # ======================

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def split_data(df):
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def main():
    df = load_data()

    print("Original shape:", df.shape)

    df = preprocess(df)

    print("After preprocessing:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, scaler = scale_data(X_train, X_test)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


if __name__ == "__main__":
    main()
