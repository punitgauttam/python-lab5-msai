# Lab 5: Titanic Machine Learning Pipeline

This project uses the Titanic dataset to practise a complete machine-learning
workflow: data exploration, cleaning decisions, train/test splitting, preprocessing,
feature engineering, model selection, tuning, and final evaluation.

The main goal is to avoid data leakage. All imputation, scaling, and encoding are
inside a scikit-learn `Pipeline`, so they are fitted only on the training data in each
cross-validation fold.

## Project workflow

### Day 1: Exploration and cleaning

`day1_exploration_starter.ipynb` performs the data-quality review:

- loads and profiles the Titanic data;
- checks data types, missing values, class balance, and unusual values;
- compares survival rates for missing versus recorded `Age` and `Cabin` values;
- documents cleaning decisions and checks for target leakage;
- creates a stratified train/test split with a fixed random state; and
- saves the split in `split.joblib` for use on Day 2.

### Day 2: Pipeline and modelling

`day2_pipeline_modeling_starter.ipynb` continues from the saved split:

- demonstrates why fitting a scaler on the full dataset causes leakage;
- builds separate numeric and categorical preprocessing pipelines;
- imputes numeric values with the median and categorical values with the most common value;
- one-hot encodes categorical variables with `handle_unknown="ignore"`;
- uses F1 score because the survival classes are imbalanced;
- tests family-size, cabin-availability, and title features;
- compares logistic regression, random forest, and gradient boosting with cross-validation;
- tunes the best model with `GridSearchCV`; and
- evaluates the held-out test set once at the end.

## Repository structure

| File | Purpose |
| --- | --- |
| `data.py` | Loads the Titanic data from OpenML, with a synthetic fallback if the dataset cannot be downloaded. |
| `pipeline.py` | Contains the train/test split, column grouping, preprocessing pipeline, model pipeline, and feature engineering functions. |
| `day1_exploration_starter.ipynb` | Day 1 exploration, quality report, cleaning decisions, and saved split. |
| `day2_pipeline_modeling_starter.ipynb` | Day 2 pipeline construction, modelling, tuning, and final evaluation. |
| `split.joblib` | Saved stratified train/test split used by Day 2. |
| `requirements.txt` | Required Python packages. |

## Installation

Use Python 3.10 or newer. Create and activate a virtual environment, then install the
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell prevents activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## Running the notebooks

1. Start Jupyter Notebook or open the folder in VS Code.
2. Open `day1_exploration_starter.ipynb`.
3. Restart its kernel and run every cell from top to bottom. This creates `split.joblib`.
4. Open `day2_pipeline_modeling_starter.ipynb`.
5. Restart its kernel and run every cell from top to bottom.
6. Review the printed cross-validation and final test F1 scores before submitting.

Run Day 1 before Day 2 if `split.joblib` does not already exist.

## Key modelling choices

- The split is stratified so the survival-class proportion is similar in training and test data.
- `Pclass` is treated as categorical because passenger class is a label, not a continuous measurement.
- Missing numeric values are imputed before scaling.
- Missing categorical values are imputed before one-hot encoding.
- `handle_unknown="ignore"` prevents prediction failures for previously unseen categories.
- F1 is reported with mean and standard deviation across cross-validation folds.
- The held-out test set is evaluated only after model selection and tuning are complete.

## Submission

Before submitting, restart both notebook kernels and run all cells again. Push this
folder to a public GitHub repository, then submit the repository link on Moodle.
