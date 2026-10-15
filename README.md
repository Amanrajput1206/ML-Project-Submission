# ML Project-I — Team #12 Savage (AIT 511, IIIT Bangalore)

Code for two Kaggle competitions:

| Task | Metric | Best model | CV | Public LB |
|---|---|---|---|---|
| Domestic Flight Price Prediction (regression) | R² on price | Equal-weight average of Ridge + Random Forest + XGBoost | R² 0.9325 | 0.96080 |
| Vehicle Type Classification (Sedan / SUV / Sports Car) | Accuracy | Noise-aware LDA and logistic regression + noise augmentation | 0.887 (noise-averaged) | 0.902 / 0.900 |

Team: Swapan Aggarwal (MT2026103), Anshuman Joshi (MT2026009), Vijay Kumar S (MT2026114), Aman Rajput (MT2026006).

## Repository layout

```
dataset/<task>/          train.csv, test.csv, sample_submission.csv (from Kaggle)
src/config.py            random seed (42), number of CV folds (5), paths
src/evaluate.py          one CV routine for every model, experiment log, submission writer
src/regression/          preprocessing (airline recovery, duplicates, duration imputer, encoders) and custom models
src/classification/      data loading, noise-augmented training, noise-aware LDA
notebooks/regression/    01_eda → 02_preprocessing → 03_models → 04_tuning → 05_ensembles
notebooks/classification/ 01_eda → 02_preprocessing → 03_models
results/<task>/          experiments.csv (every model's CV scores) and figures
submissions/<task>/      the CSV files submitted to Kaggle
```

## Setup

Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt        # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m ipykernel install --user --name ml-project --display-name "ML Project (.venv)"
```

## Reproducing the results

Run the notebooks in order with the **ML Project (.venv)** kernel: regression `01 → 05`, then classification `01 → 03`. From the command line:

```bash
.venv/Scripts/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ml-project notebooks/regression/01_eda.ipynb
# ... same for the other notebooks, in order
```

- Every random step uses the seed in `src/config.py` (42): CV splits, trees, Random Forest, XGBoost and the simulated noise.
- All models are scored with the same 5-fold CV (stratified for classification). Preprocessing that learns from data is re-fitted inside each fold.
- Each run updates `results/<task>/experiments.csv` and rewrites the files in `submissions/<task>/` with the same names that were uploaded to Kaggle.
- A full run takes about 25 minutes on a laptop (mostly `regression/05_ensembles` and `classification/03_models`).
