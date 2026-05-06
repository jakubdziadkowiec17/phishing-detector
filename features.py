import pandas as pd

from preprocessing import preprocess_df
from url_features import extract_basic_url_features


def prepare_features(
    df: pd.DataFrame,
    label_col: str = "label",
    url_col: str = "URL",
    min_ready_numeric_features: int = 10,
) -> tuple[pd.DataFrame, pd.Series]:
    # Base preprocessing (label mapping + numeric selection + imputing)
    X_num, y = preprocess_df(df, label_col=label_col, drop_text_columns=True, impute_missing=True)

    if len(X_num.columns) >= min_ready_numeric_features:
        print(
            f"\nUsing existing numeric URL features ({len(X_num.columns)} columns). "
            "No manual URL feature extraction needed."
        )
        return X_num, y

    # If dataset doesn't have enough ready numeric features, fall back to basic URL extraction
    if url_col not in df.columns:
        raise KeyError(
            f"Not enough numeric features found ({len(X_num.columns)}), "
            f"and missing `{url_col}` column for URL feature extraction."
        )

    print(
        f"\nNot enough ready numeric features ({len(X_num.columns)}). "
        f"Extracting basic URL features from `{url_col}`."
    )

    # Align rows to those used by preprocess_df (drops NaNs in label)
    df_aligned = df.dropna(subset=[label_col])
    url_feats = extract_basic_url_features(df_aligned[url_col])
    X = url_feats.copy()

    # Fill any remaining NaNs (shouldn't happen, but safe)
    if X.isna().any().any():
        X = X.fillna(0)

    return X, y

