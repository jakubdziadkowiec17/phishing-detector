import pandas as pd


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

