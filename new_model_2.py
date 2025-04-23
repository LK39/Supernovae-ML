import argparse
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn(
        "XGBoost not found; skipping that candidate.  Install via `pip install xgboost`."
    )

# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
NUMERIC_FEATURES = ["Gi", "GVhel", "mag"]


def _read_and_trim(path: str, label_value: int) -> pd.DataFrame:
    """Read one CSV and return only Gi/GVhel/mag + numeric `target`."""
    usecols = NUMERIC_FEATURES + [
        col for col in ("Type I Label", "Type II Label") if col
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)

    # keep numeric cols, drop rows with NaNs in numerics
    df_numeric = df[NUMERIC_FEATURES].copy()
    df_numeric.dropna(inplace=True)
    df_numeric["target"] = label_value
    return df_numeric


def load_binary_data(path_I: str, path_II: str) -> pd.DataFrame:
    """Merge the two CSVs into a single data frame with numeric target."""
    df_I = _read_and_trim(path_I, 0)  # Type I → 0
    df_II = _read_and_trim(path_II, 1)  # Type II → 1
    return pd.concat([df_I, df_II], ignore_index=True)


# ML helpers


def build_preprocessor():
    return make_column_transformer((StandardScaler(), NUMERIC_FEATURES))


def candidate_models():
    models = {
        "logreg": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "rf": (
            RandomForestClassifier(n_estimators=400, n_jobs=-1),
            {"clf__max_depth": [None, 6, 10]},
        ),
        "gb": (
            GradientBoostingClassifier(),
            {"clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [200, 400]},
        ),
    }
    if HAS_XGB:
        models["xgb"] = (
            XGBClassifier(tree_method="hist", eval_metric="logloss"),
            {"clf__eta": [0.05, 0.1], "clf__max_depth": [4, 6]},
        )
    return models


def grid_search_models(X, y, preproc, cv):
    best_estimator, best_score, best_name = None, -np.inf, None
    for name, (alg, grid) in candidate_models().items():
        pipe = Pipeline([("pre", preproc), ("clf", alg)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
        gs.fit(X, y)
        if gs.best_score_ > best_score:
            best_estimator, best_score, best_name = (
                gs.best_estimator_,
                gs.best_score_,
                name,
            )
    print(f"Best model: {best_name} | CV F1: {best_score:.3f}")
    return best_estimator


def evaluate(model, X_test, y_test, outdir: pathlib.Path):
    outdir.mkdir(exist_ok=True)
    y_pred = model.predict(X_test)
    (outdir / "metrics.txt").write_text(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap="Blues", colorbar=False
    )
    import matplotlib.pyplot as plt

    plt.savefig(outdir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    if len(np.unique(y_test)) == 2:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.savefig(outdir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Supernova Type‑I vs Type‑II classifier (Gi, GVhel, mag only)."
    )
    parser.add_argument(
        "--path_I",
        default="/Users/kajaloennecken/Documents/Supernovae-ML/Supernovae/filtered_specific_type_I_supernovae_with_label.csv",
        help="CSV containing Type‑I supernovae",
    )
    parser.add_argument(
        "--path_II",
        default="/Users/kajaloennecken/Documents/Supernovae-ML/Supernovae/filtered_specific_type_II_supernovae_with_label.csv",
        help="CSV containing Type‑II supernovae",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for the test split (default 0.2)",
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # 1 – load data
    df = load_binary_data(args.path_I, args.path_II)
    X, y = df[NUMERIC_FEATURES], df["target"]

    # 2 – split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # 3 – preprocess + model selection
    preproc = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    best_model = grid_search_models(X_train, y_train, preproc, cv)

    # 4 – evaluation
    outdir = pathlib.Path("outputs")
    evaluate(best_model, X_test, y_test, outdir)

    # 5 – save model
    model_path = outdir / "sn_type_classifier.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path.resolve()}")


if __name__ == "__main__":
    main()
