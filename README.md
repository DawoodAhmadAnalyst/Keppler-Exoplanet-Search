# Kepler Exoplanet Candidate Classifier — Streamlit Dashboard

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## What's inside

- **app.py** — the full dashboard (single file)
- **best_model.joblib** — your tuned Gradient Boosting pipeline from `Modelling.ipynb`
- **cumulative_cleaned.csv** — your cleaned KOI dataset
- **.streamlit/config.toml** — dark base theme (backup to the custom CSS in `app.py`)

## How the "Classify a Candidate" tab stays faithful to your notebook

Your saved `best_model.joblib` expects input that has already been through
`log1p` on all 13 features, then `StandardScaler`. The scaler itself wasn't
saved separately, so the app reconstructs it deterministically on startup:
same 70/30 stratified split, same `random_state=42`, same `log1p` + `StandardScaler`
fit on the training fold — verified to reproduce your notebook's exact reported
test-set numbers (Accuracy 0.9264 / F1-macro 0.9169 / ROC-AUC 0.9729) before
this was built. Any new input you enter goes through that same reconstructed
pipeline before hitting the model.

## Tabs

1. **Mission Overview** — what a transit is, class balance, feature glossary
2. **Explore the Data** — the period-radius diagram, per-feature distributions, correlation heatmap
3. **Model Comparison** — all 5 pipelines (test-set + 5-fold CV), confusion matrix, feature importance, ROC/PR curves
4. **Classify a Candidate** — live prediction, with buttons to load a real confirmed / false-positive example from the test set

## Extending it

- Swap in a new `best_model.joblib` at any time — the app introspects `FEATURES`
  and `model.feature_importances_`, so most model swaps need no code changes
  (as long as the new model is a tree-based sklearn estimator with the same 13 features).
- The static comparison tables (`TEST_SET_COMPARISON`, `CV_COMPARISON`) are hardcoded
  from your notebook's cell outputs — update them if you retrain.
