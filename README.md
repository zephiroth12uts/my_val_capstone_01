# myVal Next-Category Recommendation

Machine learning prototype developed for the myVal Capstone Project.

The objective of this project is to recommend the **next household-content category** that may be relevant for a customer to document, using the information already available within myVal.

The recommendation is sequential: as the customer adds more assets and the household state evolves, the model can generate a new recommendation.

The repository also contains a second model, the **household contents-value regression**, which estimates the total value of a household's contents from the items already documented. It is described in its own sections below and can optionally feed the recommendation model with five additional columns.

---

## Project Objective

The modelling task is defined as:

> Given the information already documented and analysed for a household, predict the next previously undocumented category that may be relevant for the customer to capture.

The model combines:

- documented household assets
- existing myVal AI-analysis outputs
- customer and household information
- property context
- optional property-video information

The system is designed as a recommendation tool, not as a claim that a category is definitely missing.

The second task is defined as:

> Given the items a household has documented so far, estimate the total contents value of the whole household and give a 70% interval around the estimate.

The regression excludes vehicles, because a vehicle is not a contents-insurance item and is worth many times an ordinary item.

---

## Main Workflow

```text
Customer documents an asset
        ↓
myVal AI analyses the asset
        ↓
Household state is updated
        ↓
Next-category recommendation model
        ↓
Top-1 / Top-2 / Top-3 recommendations
```

A single property can generate multiple prediction opportunities as more assets are documented.

The regression follows the same household state:

```text
Household state at the prediction time
        ↓
Earliest 10 / 20 / 30 / 50 documented non-vehicle items (the basket)
        ↓
Household-value regression (XGBoost and LightGBM blend)
        ↓
Estimated household total with a 70% interval
        ↓
Five optional regression columns for the recommendation model
```

---

## Repository Structure

The project follows a Cookiecutter Data Science structure. Every folder that holds generated files has one subfolder for the regression and one for the classification (next-category) model, and the files that existed before the regression was added stay in the folder roots.

```text
.
├── data/
│   ├── interim/
│   │   └── regression/
│   ├── processed/
│   │   ├── model_state.csv
│   │   ├── classification/
│   │   │   └── model_state_with_reg.csv
│   │   └── regression/
│   └── raw/
│       └── myVal_Synthetic_Datasets_release_v2.xlsx
│
├── models/
│   ├── next_category_top1_random_forest.joblib
│   ├── next_category_ranking_random_forest.joblib
│   ├── classification/
│   └── regression/
│
├── my_val_capstone_01/
│   ├── modeling/
│   │   ├── predict.py
│   │   └── train.py
│   └── regression/
│       ├── config.py
│       ├── dataset.py
│       ├── features.py
│       ├── plots.py
│       └── modeling/
│           ├── predict.py
│           └── train.py
│
├── notebooks/
│   ├── classification/
│   │   ├── 01_fm_data_inventory.ipynb
│   │   ├── 02_fm_target_definition.ipynb
│   │   ├── 03_ky_regression_feature_merging.ipynb
│   │   ├── 04_fm_modelling.ipynb
│   │   ├── 05_fm_inference.ipynb
│   │   └── 06_old_vs_new_comparison.md
│   └── regression/
│       ├── 01_ky_eda_dataset.ipynb
│       ├── 02_ky_data_preparation.ipynb
│       ├── 03_ky_modelling_xgboost_lightgbm.ipynb
│       ├── 04_ky_ensemble_and_intervals.ipynb
│       ├── 05_ky_test_evaluation.ipynb
│       ├── 06_ky_split_and_sampling_audit.ipynb
│       ├── 07_ky_prediction_table.ipynb
│       └── reports/
│           ├── figures/
│           └── (result tables of the regression notebooks)
│
├── reports/
│   └── figures/
│
├── docs/
│   └── docs/
│       └── regression.md
│
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

The regression code is the subpackage `my_val_capstone_01/regression/`, so it is part of the same Python package as the team's front end and inference code. The team's own files in `my_val_capstone_01/` are not changed.

### Which folder belongs to which model

| Folder                                     | Regression                                                   | Classification (next category)                              | Both                                                         |
| ------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| `data/raw/`                              |                                                              |                                                             | the source workbook                                          |
| `data/interim/regression/`               | sheet copies, property split, predictions, state diagnostics |                                                             |                                                              |
| `data/processed/model_state.csv`         | read to rebuild the split                                    | built by`02_fm_target_definition`                         | shared input                                                 |
| `data/processed/regression/`             | modelling tables and prediction table                        |                                                             |                                                              |
| `data/processed/classification/`         |                                                              | `model_state_with_reg.csv`                                |                                                              |
| `models/regression/`                     | two models, blend weight, margins, hyperparameters           |                                                             |                                                              |
| `models/classification/`                 |                                                              | Random Forest models fitted with the regression columns     |                                                              |
| `models/*.joblib` in the root            |                                                              | the original Top-1 and ranking models used by the front end |                                                              |
| `notebooks/regression/`                  | `01_ky` to `07_ky`                                       |                                                             |                                                              |
| `notebooks/classification/`              |                                                              | `01_fm`, `02_fm`, `04_fm`, `05_fm`                  | `03_ky` merges the regression into the classification data |
| `my_val_capstone_01/regression/`         | the reusable regression code                                 |                                                             |                                   |
| `my_val_capstone_01/` (other files)      |                                                              | front end and inference of the recommendation model         |                                   |
| `notebooks/regression/reports/`          | result tables and figures of the regression                  |                                                             |                                   |
| `tests/`                                 | `test_regression.py`                                       | `test_data.py` (placeholder)                              |                                                              |
| `pyproject.toml`, `uv.lock`, `docs/` |                                                              |                                                             | shared                                                       |

---

## Notebook Overview

### Classification notebooks (`notebooks/classification/`)

#### `01_fm_data_inventory.ipynb`

Initial exploration of the source workbook.

Main tasks:

- workbook and table inspection
- key and relationship validation
- category distribution analysis
- temporal sequence exploration
- AI-analysis and video coverage checks

#### `02_fm_target_definition.ipynb`

Construction of the supervised learning target.

The target was defined as the **next previously undocumented category** for a property.

This notebook also:

- reconstructs historical household states
- handles simultaneous category events
- prevents future information from entering the predictors
- combines asset, AI, customer, property and video information
- creates the final processed modelling dataset

Final processed dataset:

```text
431 prediction states
234 properties
60 columns
```

#### `03_ky_regression_feature_merging.ipynb`

Bridge between the two models. It applies the saved regression models to every classification state and adds five columns to `model_state.csv`, without any training:

| Column                  | Meaning                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| `documented_value_nv` | value documented at the prediction time for non-vehicle items                                               |
| `reg_pred_total_nv`   | regression estimate of the household total, never below the documented value (-1 when there is no estimate) |
| `reg_pred_gap_nv`     | estimated value not yet documented (-1 when there is no estimate)                                           |
| `reg_completeness_nv` | documented value divided by the estimated total (-1 when there is no estimate)                              |
| `reg_available`       | 1 if at least 10 non-vehicle items with a value are documented, otherwise 0                                 |

The estimate uses the earliest 10, 20, 30 or 50 items documented by the prediction time, so no item from the future enters a state. The result is saved as `data/processed/classification/model_state_with_reg.csv` (431 rows and 65 columns).

#### `04_fm_modelling.ipynb`

Model development and evaluation. This notebook is `03_fm_modelling.ipynb` of the team's original numbering. In this repository it reads `model_state_with_reg.csv`, so the five regression columns are predictors of every model, and it saves its models to `models/classification/`.

The dataset was split by `Property_ID` to ensure that historical states from the same household did not appear across train, validation and test partitions.

Models evaluated included:

- Majority-Class Baseline
- Logistic Regression
- Random Forest
- Tuned CatBoost
- Regularised CatBoost
- Regularised Random Forest

The notebook also includes:

- overfitting analysis
- grouped cross-validation
- hyperparameter tuning
- AI-feature ablation
- feature importance
- permutation importance
- final holdout evaluation

#### `05_fm_inference.ipynb`

Validation of the persisted models and reusable inference function. This notebook is `04_fm_inference.ipynb` of the team's original numbering.

The inference pipeline returns:

- Top-1 recommended category
- Top-2 ranked recommendations
- Top-3 ranked recommendations
- predicted probabilities

Reusable inference logic is located in:

```text
my_val_capstone_01/modeling/predict.py
```

#### `06_old_vs_new_comparison.md`

Comparison of the classification results with and without the regression columns, including the controlled check and the model choice (see below).

### Regression notebooks (`notebooks/regression/`)

The regression notebooks estimate the total contents value of a household from the first N documented items (N = 10, 20, 30 or 50), and they are run in the numbered order. Each notebook reads the files saved by the one before it.

#### `01_ky_eda_dataset.ipynb`

Exploratory analysis of the source workbook from the point of view of the household value.

Main tasks:

- data-quality checks of the four core tables (customers, customer profiles, properties and assets)
- households per customer and the structure of the properties table
- vehicles, which are examined separately because they are excluded from the target
- value of the contents of a home compared with property and customer attributes
- inspection of every tab in the workbook

#### `02_ky_data_preparation.ipynb`

Construction of the modelling table.

This notebook:

- converts the slow `.xlsx` sheets to parquet once, so that later runs reload them quickly
- builds the target, which is the total value of the non-vehicle items of a household, and the household features
- simulates a customer who has documented N items, with 5 random baskets per household and per N
- turns every basket into a row of 26 features
- applies the same grouped split by property as the classification model (70% train, 15% validation and 15% test, `GroupShuffleSplit` with seed 42)
- checks how much of the home each basket covers and compares the distribution of the total value across the three sets

Final data sets:

```text
1,840 rows from 159 properties
train: 81 properties and 1,030 rows
validation: 18 properties and 230 rows
test: 13 properties and 150 rows
unlisted: 47 properties and 430 rows
```

The households that are not in the classification states are kept in an unlisted set, which is not used for training, tuning or testing.

#### `03_ky_modelling_xgboost_lightgbm.ipynb`

Model development on the training set, with every choice made on the validation set.

The notebook includes:

- the reasoning for choosing the number of trees on the validation curve and not on the training data
- an XGBoost baseline and a LightGBM baseline without tuning
- an 80-candidate random search for each family, fitted on train and scored on validation
- tuned models fitted on all of train, with the number of trees chosen on the validation curve averaged over 8 seeds
- validation curves, validation metrics in dollars and the final hyperparameters

The target is modelled on the `log1p` scale and all metrics are converted back to dollars. The test set is not used in this notebook. The models are saved to `models/regression/`.

#### `04_ky_ensemble_and_intervals.ipynb`

Combination of the two models and construction of the 70% interval, using the validation set only.

The notebook:

- compares the two models and chooses the blend weight (0.28 for XGBoost and 0.72 for LightGBM)
- learns one interval margin for each N group (0.828, 0.511, 0.426 and 0.590)
- explains why the margins are not learned on the training set
- shows the success rate per N on the training rows and the validation rows side by side
- saves the frozen weight and margins in `models/regression/ensemble_weight_and_margins.joblib`

The interval on the validation set is in-sample, so it is a description of the fit and not a result.

#### `05_ky_test_evaluation.ipynb`

The only place where the test set is scored.

The notebook applies the frozen weight and margins to the test set and reports:

- the accuracy of the ensemble in dollars (MAE and WAPE)
- the success rate of the 70% interval for each N and overall
- the success rate at row level and at household level, with definitions and an example
- train, validation and test side by side

Nothing is tuned or changed after this notebook has been run.

#### `06_ky_split_and_sampling_audit.ipynb`

Audit of the differences between the success rates of the three sets.

The notebook examines:

- what is in each set and the bias and error by set
- which homes determine the validation margin
- the uncertainty of the margin itself, using a bootstrap
- the success rate of the test set under other random splits
- the uncertainty of the test success rate
- how far the success rate of each basket can be believed

It shows that the validation rate of about 70% is by construction, that the training rate is high because it is in-sample, and that the high test rate comes mostly from easier test households and a favourable draw of homes. A 31-household cross-fitted estimate gives a real rate of about 70%.

#### `07_ky_prediction_table.ipynb`

Creation of the prediction table of all baskets.

The notebook predicts every basket row of the four data sets and saves the estimate and the 70% interval, with a label of train, validation, test or unlisted. The interval lower bound is never below the documented value. The checks in this notebook cover consistency and labels only and do not compute accuracy. The table is saved as `data/processed/regression/regression_predictions_all_splits.csv`.

---

## Run Order

The two pipelines depend on each other, so the notebooks must be run in this order:

1. `notebooks/classification/02_fm_target_definition.ipynb` creates `data/processed/model_state.csv` from the raw workbook.
2. `notebooks/regression/01_ky` to `07_ky` build and evaluate the regression. Notebook `02_ky` reads `model_state.csv` to reproduce the classification split.
3. `notebooks/classification/03_ky_regression_feature_merging.ipynb` creates `model_state_with_reg.csv` from the saved regression models.
4. `notebooks/classification/04_fm_modelling.ipynb` (about 50 minutes) and then `05_fm_inference.ipynb`.

The regression code can also be run from the project root:

```text
python -m my_val_capstone_01.regression.dataset
python -m my_val_capstone_01.regression.modeling.train
python -m my_val_capstone_01.regression.features
python -m my_val_capstone_01.regression.plots
```

The same four steps are available as Makefile rules (`make regression_dataset`, `make regression_train`, `make regression_features` and `make regression_plots`), and each rule runs the ones it depends on.

The data folder is not tracked by git, so the raw workbook has to be placed in `data/raw/` first.

---

## Data Leakage Prevention

Data leakage prevention was a key part of the modelling design.

The main controls were:

- features only use information available at the historical prediction point
- future asset and category information is excluded
- simultaneous category events are not artificially ordered
- preprocessing is fitted only on training data
- train, validation and test sets are grouped by `Property_ID`
- the final test set remained untouched during model selection and tuning

The grouped split produced:

```text
Train:       309 rows | 163 properties
Validation:   67 rows |  35 properties
Test:         55 rows |  36 properties
```

Property overlap between all partitions was zero.

The regression follows the same rules:

- the regression uses the same grouped split as the classification model, rebuilt from `model_state.csv` with the same code, so a household is never in different partitions in the two models
- the five regression columns of a state only use items documented by the prediction time
- no model is trained on the validation or test households, and the test set is scored in one notebook only
- the 53 training states with a regression estimate are in-sample, because the regression models were fitted on those households

---

## Modelling Results

The strongest Top-1 validation result came from the standard Random Forest.

However, the model showed substantial overfitting, so a regularised Random Forest was also evaluated for ranking recommendations.

### Final Holdout Results

**Top-1 Random Forest**

```text
Accuracy:    0.5091
Macro F1:    0.3223
Weighted F1: 0.4431
```

**Regularised Random Forest Ranking**

```text
Top-2 Accuracy: 0.7091
Top-3 Accuracy: 0.8364
```

These results suggest that the problem works particularly well as a **ranked recommendation task**, where the model provides several relevant category suggestions instead of relying only on one prediction.

These are the results of the team's original run without the regression columns. The results with the columns are reported in the section on the regression columns below.

---

## AI Contribution

An ablation experiment was performed to test whether the existing myVal AI outputs added predictive value.

With AI features:

```text
Validation Macro F1: 0.3676
```

Without AI features:

```text
Validation Macro F1: 0.3324
```

This suggests that the existing AI-analysis outputs provide useful additional information for the recommendation model.

---

## Feature Interpretation

Random Forest feature importance and validation permutation importance were used to understand which variables contributed most strongly to the predictions.

Important signals included:

- documented category composition
- number of assets already documented
- documented household value
- AI-detected category counts
- AI estimated asset values
- AI suggestion acceptance
- AI category agreement
- property valuation
- customer and household context
- selected video-derived information

The strongest validation permutation signal was the existing documented category profile.

The analysis was used for interpretation only and should not be treated as evidence of causal relationships.

---

## Final Model Selection

Two final models were retained because the project supports both a primary recommendation and ranked alternatives.

### Top-1 Recommendation

**Standard Random Forest**

Selected because it achieved the strongest Top-1 validation performance.

### Top-2 / Top-3 Ranking

**Regularised Random Forest**

Selected because it provided stronger ranking behaviour and reduced the generalisation gap.

### Household Value

**Blend of XGBoost and LightGBM**

Selected with a weight of 0.28 for XGBoost and 0.72 for LightGBM, which gave the lowest validation error. The weight is not sharply determined, because every weight between 0.12 and 0.56 is within $100 of the best.

---

## Per-Class Performance

Performance was not uniform across categories.

The strongest Top-1 performance was observed for:

- Electronics
- Furniture
- Other Items
- Appliances

The most difficult categories included:

- Clothings
- Jewellery and Personal Items
- Vehicles

These categories had substantially fewer observations, which limited the model's ability to generalise reliably.

---

## Inference

Two trained models are persisted in:

```text
models/
├── next_category_top1_random_forest.joblib
└── next_category_ranking_random_forest.joblib
```

Reusable inference logic is implemented in:

```text
my_val_capstone_01/modeling/predict.py
```

The prediction function returns:

```python
{
    "top_1": {
        "category": "...",
        "probability": ...
    },
    "top_2": [...],
    "top_3": [...]
}
```

An end-to-end inference test was successfully performed using an unseen holdout observation.

Example:

```text
Actual category: Other Items

Top-1:
Electronics

Top-2:
1. Electronics
2. Other Items

Top-3:
1. Electronics
2. Other Items
3. Clothings
```

This example illustrates the value of the ranked recommendation formulation: although the first recommendation was incorrect, the true next category appeared in the second position.

---

## Household Contents-Value Regression

### Method

A customer may have documented only part of a household. The regression predicts the total non-vehicle contents value of the whole household from a basket of the documented items.

- **Data:** the same workbook, with vehicles excluded and items without a value dropped. Only the 159 of 795 properties that own at least 10 valued items can be used, which gives 1,840 basket rows.
- **Baskets:** for every household, 10, 20, 30 and 50 items are drawn at random (5 draws each) and turned into 26 features, such as summary statistics of the values, the value per category and the household context.
- **Split:** train 81 properties (1,030 rows), validation 18 (230 rows), test 13 (150 rows), and 47 properties (430 rows) that never form a classification state are kept out.
- **Models:** XGBoost and LightGBM on the log of the total, with an 80-candidate random search on validation. The number of trees is chosen on the validation curve (20 for XGBoost and 70 for LightGBM).
- **Interval:** the two models are blended, and one margin per basket size is learned on the validation set (0.83 for N = 10, 0.51 for N = 20, 0.43 for N = 30 and 0.59 for N = 50). The interval is the estimate multiplied by one minus and one plus the margin, and its lower bound is never below the documented value.

### Results

Validation error of the models, in dollars:

```text
XGBoost baseline:   MAE 35,671
XGBoost tuned:      MAE 37,535
LightGBM baseline:  MAE 37,191
LightGBM tuned:     MAE 36,793
Blend (0.28 / 0.72): MAE 36,514
```

Test set, scored once with the frozen weight and margins:

```text
Ensemble:               MAE 19,544 | WAPE 35.4%
Household-level hit rate of the 70% interval: 87.6%
Row-level hit rate:                           86.0% (129 of 150 rows)
```

These numbers need to be read with care:

- the test set has only 13 households (5 at N = 30 and 3 at N = 50), so every figure is noisy
- the test homes are smaller than the validation homes (largest $138,593 against $360,420), which explains the lower error
- the margins come from only 18 validation households and two of them set most of the width, so the intervals are wide (median width about $61,600)
- over 2,000 random re-splits of the 31 validation and test households, the average hit rate was 69.1%, so the 70% target can be neither confirmed nor rejected with the current data

### Inference

Reusable inference logic is implemented in:

```text
my_val_capstone_01/regression/modeling/predict.py
```

`predict_household_value` receives one basket and returns the estimate, the two model predictions and the 70% interval. `checkpoint_for` returns the basket size that a number of documented items reaches.

---

## Regression Columns In The Classification Model

The five regression columns were added to the recommendation dataset (`model_state_with_reg.csv`) and the modelling notebook was run again without any other change.

Only 71 of the 431 states (16.5%) have a regression estimate, because an estimate needs at least 10 documented items. The other states carry -1. By split, 53 of 309 training states, 10 of 67 validation states and 8 of 55 test states have an estimate.

Final holdout results of the two runs:

| Measure        | Original columns | With the five regression columns |
| -------------- | ---------------- | -------------------------------- |
| Top-1 accuracy | 0.5091           | 0.5091                           |
| Macro F1       | 0.3223           | 0.3257                           |
| Top-2 accuracy | 0.7091           | 0.7818                           |
| Top-3 accuracy | 0.8364           | 0.8545                           |

The differences are small compared with the noise of a test set of 55 states, and the two runs also used different library versions. A controlled check in the same environment, with the same seeds and only the five columns different, showed no consistent benefit over 20 seeds. Validation differences between the two versions were smaller than the variation between seeds and had mixed signs.

**Recommendation:** keep the original recommendation model as the model to use, and keep the household-value regression as a separate estimate. The regression columns should be reconsidered when more household histories are available. The full comparison is in `notebooks/classification/06_old_vs_new_comparison.md`.

---

## Limitations

The application and dataset are still at an early stage.

The current modelling dataset contains a limited number of historical prediction opportunities, and some categories are underrepresented.

The models also showed noticeable overfitting, especially when model complexity increased.

For this reason, the current implementation should be considered a **technical proof of concept**, rather than a production-ready recommendation system.

Future improvements would benefit from:

- more household histories
- more real customer interaction data
- better representation of minority categories
- additional behavioural signals
- repeated retraining as the application grows

The regression has its own limitations:

- only 159 households have 10 or more valued items, and only 13 of them are in the test set
- the baskets are simulated by random draws, because real logs of partial documentation do not exist
- the random search selects different candidates with different XGBoost versions, so the results can only be reproduced with the pinned version
- the margins are estimated on 18 households and are very uncertain, especially for N = 30 and N = 50
- the regression estimates are available for a small share of the classification states, and the estimates of the training states are in-sample

---

## Environment

The project uses:

- Python
- pandas
- scikit-learn
- CatBoost
- XGBoost (pinned to version 3.2.0)
- LightGBM
- pyarrow
- matplotlib
- Jupyter
- joblib
- uv

Dependencies are managed through:

```text
pyproject.toml
uv.lock
```

XGBoost is pinned because the random search of the regression selects other candidates with other XGBoost versions. After changing `pyproject.toml`, the lock file has to be regenerated with `uv lock`.

The regression tests are run with `pytest tests/test_regression.py`, and the slow check of the modelling tables is included when the environment variable `RUN_SLOW_TESTS` is set to 1.

---

## Summary

This project demonstrates that myVal's historical household information and existing AI-analysis outputs can be combined to generate useful next-category recommendations.

The strongest results were obtained when the problem was treated as a ranked recommendation task.

The final ranking model placed the observed next category within the Top-2 recommendations in **70.91%** of unseen test cases and within the Top-3 recommendations in **83.64%**.

At the same time, the experiments exposed important limitations related to sample size, class imbalance and overfitting.

The current implementation should therefore be considered a machine learning proof of concept that establishes technical feasibility and provides a foundation for future improvement as more real customer behaviour becomes available.

The household contents-value regression adds a second capability. From the first 10, 20, 30 or 50 documented items, an XGBoost and LightGBM blend estimates the total contents value of a household and gives a 70% interval whose lower bound is never below the documented value. On the 13 test households, which were scored once with a weight and margins frozen on the validation set, the test error is $19,544 (WAPE 35.4%) and the interval contains the true value for 87.6% of the households (86.0% of the rows).

The audit in `06_ky_split_and_sampling_audit` shows that this test rate should not be read as the expected rate. The test households are easier than the validation households and the draw of homes was favourable, so a cross-fitted estimate on 31 out-of-sample households gives a rate of about 70%, which matches the target of the interval. The results also reproduce only with XGBoost 3.2.0, which is pinned in `pyproject.toml`.

The two models are connected through five regression columns, built in `03_ky_regression_feature_merging`. Adding them to the recommendation model did not give a consistent benefit in a controlled 20-seed check, so the original recommendation model remains the model to use and the regression is kept as a separate estimate. Both parts are proofs of concept, and the regression columns should be reconsidered when more household histories are available.
