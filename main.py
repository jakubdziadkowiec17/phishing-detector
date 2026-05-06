import os
from pathlib import Path

import pandas as pd

import kagglehub


DATASET = "ndarvind/phiusiil-phishing-url-dataset"
FILE_PATH = ""


def find_first_csv(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(".csv"):
                return Path(dirpath) / name
    return None


def load_df():
    cache_dir = Path("data") / "kagglehub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_first_csv(cache_dir)
    if csv_path:
        return pd.read_csv(csv_path)

    if FILE_PATH:
        from kagglehub import KaggleDatasetAdapter

        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            DATASET,
            FILE_PATH,
        )

    local_dir = Path(kagglehub.dataset_download(DATASET, output_dir=str(cache_dir)))
    csv_path = find_first_csv(local_dir)
    if not csv_path:
        raise FileNotFoundError(f"No CSV found in: {local_dir}")
    return pd.read_csv(csv_path)


def quick_eda(df: pd.DataFrame, label_col: str = "label", top_missing: int = 15) -> None:
    print("\n=== Quick data check ===")

    # Missing values
    missing = df.isna().sum()
    total_missing = int(missing.sum())
    print("\nMissing values:")
    print("total_missing:", total_missing)
    print("rows_with_any_missing:", int(df.isna().any(axis=1).sum()))
    if total_missing:
        missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
        top = (
            pd.DataFrame({"missing": missing, "missing_%": missing_pct})
            .query("missing > 0")
            .sort_values(["missing", "missing_%"], ascending=False)
            .head(top_missing)
        )
        print(f"top_{top_missing}_columns_with_missing:")
        print(top)
    else:
        print("No missing values detected.")

    # Data types
    print("\nDtypes:")
    print(df.dtypes.value_counts())
    print("\nColumns + dtypes:")
    print(df.dtypes)

    # Class distribution
    if label_col in df.columns:
        print(f"\nClass distribution (`{label_col}`):")
        vc = df[label_col].value_counts(dropna=False)
        vc_pct = df[label_col].value_counts(dropna=False, normalize=True).mul(100).round(3)
        print(pd.DataFrame({"count": vc, "percent": vc_pct}))

        uniques = set(df[label_col].dropna().unique().tolist())
        if uniques.issubset({0, 1}):
            mapped = df[label_col].map({0: "legal", 1: "phishing"})
            mapped_vc = mapped.value_counts(dropna=False)
            mapped_pct = mapped.value_counts(dropna=False, normalize=True).mul(100).round(3)
            print("\nClass distribution (mapped 0->legal, 1->phishing):")
            print(pd.DataFrame({"count": mapped_vc, "percent": mapped_pct}))
    else:
        print(f"\nNo `{label_col}` column found; cannot compute phishing/legal distribution.")


def preprocess_df(
    df: pd.DataFrame,
    label_col: str = "label",
    drop_text_columns: bool = True,
    impute_missing: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: `{label_col}`")

    out = df.copy()

    # Ensure label is {0,1} with 1=phishing, 0=legal
    out = out.dropna(subset=[label_col])
    y_raw = out[label_col]

    # If it's already numeric 0/1, keep. Otherwise map common string labels.
    if not pd.api.types.is_numeric_dtype(y_raw):
        y_norm = y_raw.astype(str).str.strip().str.lower()
        mapping = {
            "phishing": 1,
            "phish": 1,
            "1": 1,
            "true": 1,
            "yes": 1,
            "legal": 0,
            "legit": 0,
            "legitimate": 0,
            "0": 0,
            "false": 0,
            "no": 0,
        }
        y = y_norm.map(mapping)
        if y.isna().any():
            bad = sorted(set(y_norm[y.isna()].unique().tolist()))
            raise ValueError(
                f"Unrecognized labels in `{label_col}` (cannot map to 0/1): {bad[:20]}"
            )
        y = y.astype("int64")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        uniques = set(y.dropna().unique().tolist())
        if not uniques.issubset({0, 1}):
            raise ValueError(f"`{label_col}` must be binary 0/1. Found values: {sorted(uniques)}")
        y = y.astype("int64")

    # Build feature matrix
    X = out.drop(columns=[label_col])

    if drop_text_columns:
        # Keep only numeric columns (model-friendly without encoding)
        X = X.select_dtypes(include=["number"]).copy()

    if impute_missing:
        # Fill numeric NaNs with median (robust default)
        for col in X.columns:
            if X[col].isna().any():
                med = X[col].median()
                if pd.isna(med):
                    # all-NaN column -> fill with 0 then keep (or later drop if desired)
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(med)

    # Final safety check
    if X.isna().any().any():
        still_missing = X.isna().sum().sort_values(ascending=False)
        still_missing = still_missing[still_missing > 0].head(20)
        raise ValueError(f"Missing values remain after preprocessing (top columns):\n{still_missing}")

    return X, y


def extract_basic_url_features(urls: pd.Series) -> pd.DataFrame:
    s = urls.astype("string").fillna("")
    out = pd.DataFrame(index=urls.index)

    out["url_len"] = s.str.len().astype("int64")
    out["dot_count"] = s.str.count(r"\.").astype("int64")
    out["has_https"] = s.str.startswith("https://").astype("int64")

    # Very simple IPv4 detection anywhere in URL
    ipv4_re = r"(?:\d{1,3}\.){3}\d{1,3}"
    out["has_ip"] = s.str.contains(ipv4_re, regex=True).astype("int64")

    out["digit_count"] = s.str.count(r"\d").astype("int64")
    out["special_count"] = s.str.count(r"[^A-Za-z0-9]").astype("int64")

    return out


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


if __name__ == "__main__":
    df = load_df()
    print("head():")
    print(df.head())
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    quick_eda(df)

    X, y = prepare_features(df)
    print("\n=== Features ready ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X columns (first 30):", list(X.columns[:30]))
    print("y distribution:", y.value_counts().to_dict())
