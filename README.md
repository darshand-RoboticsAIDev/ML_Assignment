# Bean Sorter: Dry Bean Variety Classification

An end-to-end machine learning project that trains five classifiers on the UCI Dry Bean dataset, saves them, and serves them through an interactive Streamlit app where you can upload labelled test data and evaluate each model without retraining.

## a. Problem statement

Dry beans are the most widely produced pulse crop, and seed quality strongly affects yield and market price. Sorting beans by variety by hand is slow and inconsistent. The goal of this project is to predict the **variety of a dry bean (7 classes)** from 16 geometric shape features extracted from a photograph of the bean, and to compare how five classical classification algorithms perform on this multi-class problem.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository, *Dry Bean Dataset* (id 602), Koklu & Ozkan (2020) |
| Task | Multi-class classification (7 classes) |
| Instances | 13,611 (13,543 after removing 68 exact duplicate rows) |
| Features | 16 numeric, no missing values |
| Target | `Class`: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |

**Features.** 12 dimension features (`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, `Compactness`) and 4 shape factors (`ShapeFactor1` to `ShapeFactor4`). Column names are kept exactly as in the original UCI file (including the original spelling `AspectRation`).

**Class balance.** The classes are imbalanced: DERMASON 3,546 · SIRA 2,636 · SEKER 2,027 · HOROZ 1,928 · CALI 1,630 · BARBUNYA 1,322 · BOMBAY 522.

**Preprocessing.**

1. Removed 68 exact duplicate rows.
2. Stratified 80/20 train/test split (`random_state=2026`): 10,834 training rows and 2,709 test rows. The test split is saved as `test_data.csv`.
3. `StandardScaler` inside every model pipeline. Features range from about 10⁻³ (`ShapeFactor2`) to 10⁵ (`Area`), so scaling matters for distance- and margin-based models.
4. Hyper-parameters chosen with 5-fold stratified cross-validation on the **training set only**, using macro F1.

## c. GitHub repository link

**Repository:** `<paste your GitHub repository URL here>`

**Live Streamlit app:** `<paste your Streamlit Community Cloud URL here>`

```
bean_classifier/
├── app.py                  # Streamlit web application
├── train_models.py         # Trains, tunes, evaluates and saves all five models
├── bean_metrics.py         # Shared metric function (used by training and the app)
├── requirements.txt
├── README.md
├── test_data.csv           # Held-out test split (16 features + Class), 2,709 rows
├── data/
│   └── dry_bean_dataset.csv   # Full original dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── label_encoder.joblib
    ├── metadata.json          # features, classes, tuned hyper-parameters
    └── metrics_summary.csv    # comparison table below
```

## d. Models used

All five models are scikit-learn `Pipeline(StandardScaler → classifier)` objects, trained on the same training split and evaluated on the same test split.

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | `C=1.0`, `max_iter=3000` |
| Decision Tree | `max_depth=10`, `min_samples_leaf=10` |
| K-Nearest Neighbors | `n_neighbors=9`, `weights="distance"` |
| Naive Bayes (Gaussian) | `var_smoothing=1e-9` |
| Support Vector Machine | RBF kernel, `C=10`, `gamma="scale"`, `probability=True` |

**Metric definitions.** Because the problem is multi-class, Precision, Recall and F1 are **macro-averaged** (every variety counts equally, so the small BOMBAY class is not swamped by DERMASON). AUC is the **macro one-vs-rest ROC AUC** computed from predicted class probabilities. MCC is the multi-class Matthews Correlation Coefficient.

### Comparison table (test set, 2,709 beans)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9251 | **0.9947** | 0.9371 | 0.9373 | 0.9371 | 0.9094 |
| Decision Tree | 0.9110 | 0.9821 | 0.9252 | 0.9232 | 0.9241 | 0.8924 |
| KNN | 0.9228 | 0.9882 | 0.9374 | 0.9364 | 0.9369 | 0.9067 |
| Naive Bayes | 0.8989 | 0.9914 | 0.9065 | 0.9120 | 0.9089 | 0.8781 |
| Support Vector Machine | **0.9280** | 0.9946 | **0.9404** | **0.9398** | **0.9400** | **0.9129** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A very strong linear baseline: 92.5% accuracy and the highest AUC (0.9947), so its probability ranking of classes is excellent. Most of the shape features separate the varieties almost linearly once they are scaled. Its main weakness is the DERMASON ↔ SIRA boundary (about 90 of its ~200 errors), which is the hardest pair for every model. |
| Decision Tree | The weakest model apart from Naive Bayes (91.1% accuracy, MCC 0.892). Axis-aligned splits approximate the curved class boundaries in steps, and its probability estimates are coarse (leaf frequencies), which is why its AUC (0.982) is the lowest of the five. Limiting depth to 10 and leaves to 10 samples was needed to stop it overfitting. It is still the most interpretable model. |
| KNN | Close to Logistic Regression (92.3% accuracy, macro F1 0.937) and the best on the CALI and HOROZ classes. Distance weighting with k = 9 worked best. It depends completely on scaling (Area is roughly 10⁷ times larger than ShapeFactor2), and it is the heaviest model to store (the whole training set) and the slowest to predict. Its AUC is lower than the probabilistic models because its "probabilities" come from neighbour votes. |
| Naive Bayes | Lowest accuracy (89.9%) and MCC (0.878). Many features are almost perfectly correlated (Area, ConvexArea, EquivDiameter, Perimeter), which violates the independence assumption and double-counts the same evidence. This hurts BARBUNYA ↔ CALI and DERMASON ↔ SIRA in particular. Its AUC is still high (0.991), so the ranking of classes is reasonable even when the top-1 prediction is wrong. It is the fastest model to train. |
| Support Vector Machine | Best on 5 of the 6 metrics: accuracy 92.8%, macro F1 0.940, MCC 0.913, with AUC (0.9946) essentially tied with Logistic Regression. The RBF kernel captures the slightly non-linear boundaries between the medium-sized varieties (SEKER, SIRA, DERMASON) better than the linear model. Costs: slower training, and probability outputs need an internal calibration step. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF).** It has the highest MCC, the most reliable single score under class imbalance, and leads on accuracy, precision, recall and F1. Logistic Regression is a very close second and would be a good choice if a simpler, faster, more explainable model were preferred. Every model classifies BOMBAY almost perfectly; the remaining errors for all models are concentrated in DERMASON vs SIRA, which look very similar in shape. |

## Streamlit app features

- **CSV upload** of labelled test data (with a fallback to the bundled `test_data.csv` and a button to download it).
- **Model selection dropdown** for the five saved classifiers, showing each model's tuned hyper-parameters.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC as metric cards.
- **Confusion matrix** (counts or row-normalised) and a **classification report** with per-class precision, recall and F1, plus the most frequent mistake.
- **Compare all models** tab: all five models scored on the uploaded file, with the best model highlighted and a bar chart for any metric.
- **Predictions** tab: row-level predicted variety and confidence, a filter for misclassified beans, and CSV download.

The app only loads the saved `.joblib` pipelines and calls `predict` / `predict_proba`. It never retrains on uploaded data. Uploaded files must contain the 16 feature columns and the `Class` column. Extra columns are ignored, and rows with non-numeric values are dropped with a warning.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # optional: rebuilds model/ and test_data.csv
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this folder to a public GitHub repository.
2. On https://streamlit.io/cloud choose **New app**, then select the repository, branch `main` and file `app.py`.
3. Under **Advanced settings**, choose **Python 3.12**. The pinned package versions need Python 3.11 or newer, and the saved models must be loaded with the same scikit-learn version they were trained with (1.9.1).
4. Click **Deploy**.

## Reference

Koklu, M. and Ozkan, I.A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507.
