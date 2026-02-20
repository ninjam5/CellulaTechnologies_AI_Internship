import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].str.strip()

    df.drop(columns=["Booking_ID"], inplace=True)
    df.rename(columns={"average price ": "average price"}, inplace=True)

    df["date of reservation"] = pd.to_datetime(
        df["date of reservation"], format="mixed", dayfirst=False, errors="coerce"
    )
    invalid_dates = df["date of reservation"].isna().sum()
    if invalid_dates > 0:
        print(f"  {invalid_dates} rows with invalid dates removed.")
        df.dropna(subset=["date of reservation"], inplace=True)

    df["reservation_month"] = df["date of reservation"].dt.month
    df.drop(columns=["date of reservation"], inplace=True)

    print(f"  Data loaded and cleaned. Shape: {df.shape}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = len(df_clean)
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]
        after = len(df_clean)
        print(f"    {col}: removed {before - after} outliers "
              f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
    return df_clean.reset_index(drop=True)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df["booking status"] = df["booking status"].map(
        {"Canceled": 1, "Not_Canceled": 0}
    )
    return df


def build_pipeline(numerical_features: list, categorical_features: list) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="error"),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    return pipeline


def main():
    print("=" * 60)
    print("  Hotel Booking Classification - Model Training")
    print("=" * 60)

    print("\n[1/6] Loading and cleaning data...")
    df = load_and_clean_data("dataset.csv")

    print("\n[2/6] Removing outliers (IQR method)...")
    outlier_cols = ["lead time", "average price", "P-C", "special requests"]
    df = remove_outliers_iqr(df, outlier_cols)
    print(f"  Shape after outlier removal: {df.shape}")

    print("\n[3/6] Encoding target variable...")
    df = encode_target(df)
    print(f"  Target distribution:\n{df['booking status'].value_counts().to_string()}")

    print("\n[4/6] Splitting data (80/20)...")
    X = df.drop(columns=["booking status"])
    y = df["booking status"]

    categorical_features = ["type of meal", "room type", "market segment type"]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    print(f"  Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    print("\n[5/6] Building and training pipeline (Random Forest)...")
    pipeline = build_pipeline(numerical_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Canceled", "Canceled"]))

    print("[6/6] Saving pipeline...")
    joblib.dump(pipeline, "model_pipeline.pkl")
    print("  Pipeline saved as 'model_pipeline.pkl'")
    print("\n" + "=" * 60)
    print("  Training complete! Ready for Flask deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
