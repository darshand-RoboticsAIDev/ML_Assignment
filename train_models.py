"""
train_models.py
---------------
Trains five classifiers on the UCI Dry Bean dataset, evaluates them on a
held-out stratified test split, and saves:

    model/<model_key>.joblib   - fitted sklearn Pipeline (scaler + classifier)
    model/label_encoder.joblib - maps bean class names <-> integer labels
    model/metadata.json        - feature list, class list, chosen hyper-params
    model/metrics_summary.csv  - comparison table (Accuracy, AUC, Prec, Rec, F1, MCC)
    test_data.csv              - the held-out test split (features + Class)

Run:  python train_models.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from bean_metrics import compute_metrics

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "dry_bean_dataset.csv"
MODEL_DIR = ROOT / "model"
TEST_FILE = ROOT / "test_data.csv"

TARGET = "Class"
SEED = 2026

# sklearn >=1.9 flags SVC(probability=True) as deprecated; it still works and
# gives the probability scores needed for AUC, so silence the notice.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
TEST_FRACTION = 0.20


def load_beans() -> pd.DataFrame:
    beans = pd.read_csv(DATA_FILE)
    n_before = len(beans)
    beans = beans.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_before} rows, dropped {n_before - len(beans)} exact duplicates "
          f"-> {len(beans)} rows, {beans.shape[1] - 1} features")
    return beans


def candidate_models():
    """Each entry: key -> (display name, estimator, small hyper-param grid).

    Every estimator sits behind a StandardScaler: the 16 shape features live on
    very different scales (Area ~ 10^4-10^5 vs ShapeFactor2 ~ 10^-3), which
    matters for LR, KNN and SVM. The tree ignores scaling, and GaussianNB is
    unaffected by it, but keeping one pipeline shape keeps the app simple.
    """
    return {
        "logistic_regression": (
            "Logistic Regression",
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "decision_tree": (
            "Decision Tree",
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [6, 8, 10, 14], "clf__min_samples_leaf": [1, 5, 10]},
        ),
        "knn": (
            "KNN",
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 9, 15], "clf__weights": ["uniform", "distance"]},
        ),
        "naive_bayes": (
            "Naive Bayes (Gaussian)",
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        "svm": (
            "Support Vector Machine",
            SVC(kernel="rbf", probability=True, random_state=SEED),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.05]},
        ),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    beans = load_beans()

    feature_cols = [c for c in beans.columns if c != TARGET]
    encoder = LabelEncoder().fit(beans[TARGET])
    X = beans[feature_cols]
    y = encoder.transform(beans[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Save the untouched held-out split for the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = encoder.inverse_transform(y_test)
    test_df.to_csv(TEST_FILE, index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, chosen_params = [], {}

    for key, (label, estimator, grid) in candidate_models().items():
        pipe = Pipeline([("scale", StandardScaler()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)
        scores = compute_metrics(y_test, y_pred, y_proba)

        joblib.dump(best, MODEL_DIR / f"{key}.joblib", compress=3)
        chosen_params[key] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        rows.append({"Model": label, **scores})
        print(f"{label:<26} best={chosen_params[key]}  "
              + "  ".join(f"{m}={v:.4f}" for m, v in scores.items()))

    summary = pd.DataFrame(rows).set_index("Model").round(4)
    summary.to_csv(MODEL_DIR / "metrics_summary.csv")

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    meta = {
        "dataset": "UCI Dry Bean Dataset (id 602)",
        "target": TARGET,
        "features": feature_cols,
        "classes": encoder.classes_.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_seed": SEED,
        "averaging": "macro (one-vs-rest for AUC)",
        "best_params": chosen_params,
        "models": {k: v[0] for k, v in candidate_models().items()},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n", summary.to_string())


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
