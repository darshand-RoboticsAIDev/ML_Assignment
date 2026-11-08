"""
Bean Sorter - Streamlit front-end for the Dry Bean variety classifiers.

The app never retrains anything: it loads the five fitted pipelines saved in
model/ by train_models.py, scores whatever labelled CSV the user uploads, and
shows metrics, a confusion matrix and a classification report.
"""

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from bean_metrics import compute_metrics

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"

METRIC_HELP = {
    "Accuracy": "Share of beans whose variety was predicted correctly.",
    "AUC": "Macro one-vs-rest ROC AUC from predicted probabilities.",
    "Precision": "Macro average: of beans labelled X, how many really are X.",
    "Recall": "Macro average: of real X beans, how many were found.",
    "F1": "Macro average harmonic mean of precision and recall.",
    "MCC": "Matthews correlation: balanced score in [-1, 1], robust to class imbalance.",
}

BEAN_NOTES = {
    "BARBUNYA": "Large, oval, speckled",
    "BOMBAY": "Very large, round (easiest class)",
    "CALI": "Large, kidney-shaped, white",
    "DERMASON": "Small, rounded, most common",
    "HOROZ": "Long, slender",
    "SEKER": "Medium, very round",
    "SIRA": "Small-medium, often mixed up with DERMASON",
}

st.set_page_config(page_title="Bean Sorter", page_icon="🫘", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .bean-hero {background: linear-gradient(100deg, #5b3a1e 0%, #a0673a 55%, #d9b48f 100%);
                  color: #fff8ef; padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;}
      .bean-hero h1 {color: #fff8ef; margin: 0; font-size: 1.9rem;}
      .bean-hero p {margin: .25rem 0 0 0; opacity: .92;}
      div[data-testid="stMetric"] {background: rgba(160,103,58,.08); border: 1px solid rgba(160,103,58,.25);
                                   border-radius: 10px; padding: .55rem .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Loading helpers (cached so the joblib files are read once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    models = {key: joblib.load(MODEL_DIR / f"{key}.joblib") for key in meta["models"]}
    return meta, encoder, models


@st.cache_data
def read_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def validate(df: pd.DataFrame, meta: dict, known_classes):
    """Return (clean_df, problems). Problems is a list of human-readable errors."""
    problems = []
    missing = [c for c in meta["features"] + [meta["target"]] if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s): {', '.join(missing)}")
        return None, problems

    df = df[meta["features"] + [meta["target"]]].copy()
    df[meta["target"]] = df[meta["target"]].astype(str).str.strip().str.upper()
    for col in meta["features"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_rows = df[meta["features"]].isna().any(axis=1)
    if bad_rows.any():
        st.warning(f"Dropped {int(bad_rows.sum())} row(s) with missing / non-numeric feature values.")
        df = df[~bad_rows]

    unknown = sorted(set(df[meta["target"]]) - set(known_classes))
    if unknown:
        problems.append(f"Unknown class label(s) in '{meta['target']}': {', '.join(unknown)}. "
                        f"Expected one of: {', '.join(known_classes)}")
    if len(df) == 0:
        problems.append("No usable rows left after cleaning.")
    return df, problems


def plot_confusion(cm, labels, normalise):
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalise else cm
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(data, annot=True, fmt=".2f" if normalise else "d", cmap="YlOrBr",
                xticklabels=labels, yticklabels=labels, cbar=False, linewidths=.5,
                linecolor="white", ax=ax, annot_kws={"size": 9})
    ax.set_xlabel("Predicted variety")
    ax.set_ylabel("Actual variety")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def score_model(model, X, y_true, labels):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba, compute_metrics(y_true, y_pred, y_proba, labels=labels)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
meta, encoder, models = load_artifacts()
class_names = encoder.classes_.tolist()
label_ids = np.arange(len(class_names))
name_by_key = meta["models"]

st.markdown(
    """
    <div class="bean-hero">
      <h1>🫘 Bean Sorter</h1>
      <p>Seven dry-bean varieties, sixteen shape measurements, five pre-trained classifiers.
         Upload labelled test beans and see how each model sorts them.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1 · Test data")
    upload = st.file_uploader("Upload a labelled CSV (test data only)", type=["csv"])
    use_sample = st.toggle("Fall back to bundled test_data.csv", value=True,
                           help="Used only when no file is uploaded. It is the 20% stratified "
                                "hold-out split saved during training.")
    st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                       file_name="test_data.csv", mime="text/csv", use_container_width=True)

    st.header("2 · Model")
    model_key = st.selectbox("Classifier", list(name_by_key), format_func=name_by_key.get)
    params = meta["best_params"].get(model_key, {})
    if params:
        st.caption("Tuned hyper-parameters: " + ", ".join(f"`{k}={v}`" for k, v in params.items()))

    st.header("3 · Display")
    normalise = st.checkbox("Normalise confusion matrix by row", value=False)

    with st.expander("Required columns"):
        st.code("\n".join(meta["features"] + [meta["target"] + "  (target)"]), language=None)

# Pick the data source
if upload is not None:
    raw_df, source = read_csv(upload.getvalue()), upload.name
elif use_sample:
    raw_df, source = read_csv(SAMPLE_CSV.read_bytes()), "test_data.csv (bundled)"
else:
    st.info("Upload a CSV in the sidebar, or switch on the bundled test data to get started.")
    st.stop()

df, problems = validate(raw_df, meta, class_names)
if problems:
    for p in problems:
        st.error(p)
    st.stop()

X = df[meta["features"]]
y_true = encoder.transform(df[meta["target"]])

tab_eval, tab_compare, tab_preds, tab_about = st.tabs(
    ["📊 Model evaluation", "🏁 Compare all models", "🔎 Predictions", "ℹ️ About the data"]
)

# ---- Tab 1: selected model ------------------------------------------------- #
with tab_eval:
    y_pred, y_proba, scores = score_model(models[model_key], X, y_true, label_ids)
    st.subheader(f"{name_by_key[model_key]} on {source}")
    st.caption(f"{len(df):,} beans scored · {df[meta['target']].nunique()} varieties present · "
               "metrics use macro averaging")

    cols = st.columns(6)
    for col, (name, value) in zip(cols, scores.items()):
        col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}", help=METRIC_HELP[name])

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)
        st.pyplot(plot_confusion(cm, class_names, normalise), use_container_width=True)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y_true, y_pred, labels=label_ids, target_names=class_names,
                                       output_dict=True, zero_division=0)
        report.pop("accuracy", None)  # already shown in the metric cards above
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                    "f1-score": "{:.3f}", "support": "{:,}"})
            .background_gradient(subset=["f1-score"], cmap="YlOrBr", vmin=0.8, vmax=1.0),
            use_container_width=True,
        )

        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        if off_diag.sum():
            a, b = np.unravel_index(off_diag.argmax(), off_diag.shape)
            st.info(f"Most frequent mistake: **{class_names[a]}** predicted as "
                    f"**{class_names[b]}** ({off_diag[a, b]} beans).")

# ---- Tab 2: all models side by side --------------------------------------- #
with tab_compare:
    st.subheader("All five classifiers on the same data")
    rows = []
    for key, model in models.items():
        _, _, s = score_model(model, X, y_true, label_ids)
        rows.append({"Model": name_by_key[key], **s})
    table = pd.DataFrame(rows).set_index("Model")
    best = table["MCC"].idxmax()

    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#e8c9a6"),
                 use_container_width=True)
    st.success(f"Best by MCC on this data: **{best}** (MCC {table.loc[best, 'MCC']:.4f}).")

    metric_pick = st.radio("Chart metric", list(METRIC_HELP), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ordered = table[metric_pick].sort_values()
    bars = ax.barh(ordered.index, ordered.values,
                   color=["#a0673a" if m == best else "#d9b48f" for m in ordered.index])
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    lo = np.nanmin(ordered.values)
    ax.set_xlim(max(0, lo - 0.05), 1.0)
    ax.set_xlabel(metric_pick)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---- Tab 3: row-level predictions ----------------------------------------- #
with tab_preds:
    st.subheader(f"Row-level predictions · {name_by_key[model_key]}")
    out = df.copy()
    out["Predicted"] = encoder.inverse_transform(y_pred)
    out["Confidence"] = y_proba.max(axis=1).round(4)
    out["Correct"] = out["Predicted"] == out[meta["target"]]
    only_wrong = st.checkbox("Show misclassified beans only")
    view = out[~out["Correct"]] if only_wrong else out
    st.caption(f"Showing {len(view):,} of {len(out):,} rows")
    st.dataframe(view[[meta["target"], "Predicted", "Confidence", "Correct"] + meta["features"]],
                 use_container_width=True, height=420)
    st.download_button("Download predictions as CSV", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_key}.csv", mime="text/csv")

# ---- Tab 4: dataset info -------------------------------------------------- #
with tab_about:
    st.subheader("UCI Dry Bean dataset")
    st.write(
        "13,611 images of seven registered dry-bean varieties were captured with a "
        "high-resolution camera; each bean was segmented and described by 12 dimension "
        "features and 4 shape factors. The models were trained on 80% of the "
        f"de-duplicated data ({meta['train_rows']:,} rows) and the remaining "
        f"{meta['test_rows']:,} rows form test_data.csv."
    )
    counts = df[meta["target"]].value_counts().rename("Beans in uploaded data")
    info = pd.DataFrame({"Description": pd.Series(BEAN_NOTES)}).join(counts).fillna(0)
    info["Beans in uploaded data"] = info["Beans in uploaded data"].astype(int)
    st.dataframe(info, use_container_width=True)
    st.caption("Source: Koklu, M. & Ozkan, I.A. (2020). Multiclass classification of dry beans "
               "using computer vision and machine learning techniques. Computers and "
               "Electronics in Agriculture, 174, 105507.")
