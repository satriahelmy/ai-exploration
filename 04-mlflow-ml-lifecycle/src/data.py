import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    """Prepare features and target for model training."""

    df = df.copy()

    # TotalCharges contains blank strings for new customers
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # customerID is an identifier, not a predictive feature
    df = df.drop(columns=["customerID"])

    # Convert target to binary
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1,
    })

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def split_data(X, y, val_size=0.2, test_size=0.2, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test