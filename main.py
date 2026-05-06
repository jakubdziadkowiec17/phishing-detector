from config import DEFAULT_LABEL_COL, DEFAULT_URL_COL
from data_io import load_df
from eda import quick_eda
from features import prepare_features


if __name__ == "__main__":
    df = load_df()
    print("head():")
    print(df.head())
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    quick_eda(df, label_col=DEFAULT_LABEL_COL)

    X, y = prepare_features(df, label_col=DEFAULT_LABEL_COL, url_col=DEFAULT_URL_COL)
    print("\n=== Features ready ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X columns (first 30):", list(X.columns[:30]))
    print("y distribution:", y.value_counts().to_dict())
