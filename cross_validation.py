import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler


def run_cross_validation(X, y, *, use_smote: bool = True, random_state: int = 42):
    """5-fold stratified cross-validation.

    Notes:
        If use_smote=True, SMOTE is applied *inside* each training fold via an imblearn
        pipeline. This avoids data leakage and is the recommended methodology.
    """
    print("\n=== Cross-Validation (5-fold Stratified) ===")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    smote = SMOTE(random_state=random_state) if use_smote else "passthrough"

    models = {
        "Logistic Regression": ImbPipeline(
            steps=[
                ("smote", smote),
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "Random Forest": ImbPipeline(
            steps=[
                ("smote", smote),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_phishing": make_scorer(precision_score, pos_label=0),
        "recall_phishing": make_scorer(recall_score, pos_label=0),
        "f1_phishing": make_scorer(f1_score, pos_label=0),
    }

    results = []

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

        for metric_name in scoring.keys():
            scores = cv_results[f"test_{metric_name}"]
            mean_score = scores.mean()
            std_score = scores.std()

            print(f"  {metric_name}:")
            for fold_idx, score in enumerate(scores):
                print(f"    Fold {fold_idx + 1}: {score:.4f}")
            print(f"    Mean: {mean_score:.4f} (+/- {std_score:.4f})")

            results.append({
                "model": model_name,
                "metric": metric_name,
                "mean": mean_score,
                "std": std_score,
            })

    results_df = pd.DataFrame(results)
    return results_df
