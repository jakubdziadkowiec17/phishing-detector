import numpy as np
import pandas as pd


def create_noisy_test_set(
    X_test: pd.DataFrame,
    noise_std: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Disturb the test set with Gaussian noise proportional to each feature's std.

    Used ONLY for robustness analysis — never for standard evaluation.
    noise_std=0.0 returns a clean copy; noise_std=1.0 adds noise equal to 1 std per feature.
    """
    rng = np.random.default_rng(random_state)
    X_noisy = X_test.copy()

    for col in X_noisy.select_dtypes(include=["number"]).columns:
        col_std = X_noisy[col].std()
        if col_std > 0:
            noise = rng.normal(0, noise_std * col_std, len(X_noisy))
        else:
            noise = np.zeros(len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise

    return X_noisy
