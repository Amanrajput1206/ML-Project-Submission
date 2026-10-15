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

The repository does not include a Python environment. Create one with Python 3.11 and install the pinned packages from `requirements.txt`. The commands below create it in a folder named `.venv` (ignored by git) and register it as a Jupyter kernel called `ml-project`.

Windows (Command Prompt or PowerShell):

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name ml-project --display-name "ML Project (.venv)"
```

Linux / macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name ml-project --display-name "ML Project (.venv)"
```

## Reproducing the results

Open the notebooks in Jupyter or VS Code, select the **ML Project (.venv)** kernel, and run them top to bottom in this order: regression `01 → 05`, then classification `01 → 03`. The notebooks find `src/` and `dataset/` on their own, so no paths need changing.

Or run them all from a bash shell (Git Bash on Windows). On Linux/macOS replace `.venv/Scripts/python` with `.venv/bin/python`:

```bash
for nb in regression/01_eda regression/02_preprocessing regression/03_models regression/04_tuning regression/05_ensembles classification/01_eda classification/02_preprocessing classification/03_models; do
  .venv/Scripts/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ml-project --ExecutePreprocessor.timeout=-1 notebooks/$nb.ipynb
done
```

- Every random step uses the seed in `src/config.py` (42): CV splits, trees, Random Forest, XGBoost and the simulated noise.
- All models are scored with the same 5-fold CV (stratified for classification). Preprocessing that learns from data is re-fitted inside each fold.
- Each run updates `results/<task>/experiments.csv` and rewrites the files in `submissions/<task>/` with the same names that were uploaded to Kaggle.
- A full run takes about 25 minutes on a laptop (mostly `regression/05_ensembles` and `classification/03_models`).
