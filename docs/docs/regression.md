# Household contents-value regression

This model sits next to the next-category recommendation model. It estimates the **total contents value of a household** from the N items that the customer has documented (N = 10, 20, 30 or 50) and returns a **70% interval** around the estimate. Vehicles are excluded from the regression.

## Folder layout

Every folder that holds generated files has one subfolder for the regression and one for the classification model. The files of the classification team in the folder roots are left where they are.

| Folder | `regression/` | `classification/` |
|---|---|---|
| `notebooks/` | `01_ky` to `07_ky`, the regression pipeline | `01_fm`, `02_fm`, `04_fm` and `05_fm` of the classification team, and `03_ky` that merges the regression into them |
| `data/interim/` | sheet copies, the property split, predictions and diagnostics | none |
| `data/processed/` | the modelling tables and the prediction table | `model_state_with_reg.csv` |
| `models/` | the two models, the blend weight, the margins and the hyperparameters | the Random Forest models that were fitted with the regression columns |

## Regression notebooks (`notebooks/regression/`, run in this order)

| Notebook | Purpose | Reads | Writes |
|---|---|---|---|
| `01_ky_eda_dataset` | Exploratory analysis of the four core tables | the workbook in `data/raw/` | parquet copies of the sheets in `data/interim/regression/` |
| `02_ky_data_preparation` | Builds the target, simulates the baskets, creates the features and applies the grouped split, which is rebuilt from `data/processed/model_state.csv` with the same code as `03_fm_modelling` | the workbook and `model_state.csv` | `train_df`, `val_df`, `test_df` and `unlisted_df` in `data/processed/regression/`, and `property_split.csv` in `data/interim/regression/` |
| `03_ky_modelling_xgboost_lightgbm` | Fits XGBoost and LightGBM with an 80-candidate random search and chooses the number of trees on validation, without using the test set | `train_df` and `val_df` | the two models and `final_hyperparameters.csv` in `models/regression/`, the result tables in `notebooks/regression/reports/` and the validation predictions in `data/interim/regression/` |
| `04_ky_ensemble_and_intervals` | Learns the blend weight and the 70% margins on the validation set | the two models and their validation predictions | `ensemble_weight_and_margins.joblib` in `models/regression/` and the validation intervals in `data/interim/regression/` |
| `05_ky_test_evaluation` | Scores the test set once with the frozen weight and margins | the two models, the frozen weight and margins, and `test_df` | the result tables in `notebooks/regression/reports/` and the test predictions in `data/interim/regression/` |
| `06_ky_split_and_sampling_audit` | Explains why train, validation and test give different success rates, using bootstrap and random re-split experiments, without tuning anything | the models, the frozen weight and margins, and the three modelling tables | no files, the results are shown in the notebook |
| `07_ky_prediction_table` | Saves the prediction and the 70% interval of every basket row with a train, validation, test or unlisted label | the models, the frozen weight and margins, the modelling tables and `property_split.csv` | `regression_predictions_all_splits.csv` in `data/processed/regression/` |

## Classification notebooks (`notebooks/classification/`)

The notebooks of the classification team were copied from the working copy of the classification model. Their code and outputs are unchanged. Only the relative paths were adjusted to the deeper folder (`../` became `../../`), the data and model paths of notebooks 04 and 05 point to the `classification` subfolders, and two description cells were updated.

| Notebook | Content |
|---|---|
| `01_fm_data_inventory` and `02_fm_target_definition` | the data inventory and the construction of `data/processed/model_state.csv` |
| `03_ky_regression_feature_merging` | applies the saved regression models to every classification state and adds five regression columns to `model_state.csv`, without any training. It writes `model_state_with_reg.csv` to `data/processed/classification/` and a diagnostics file to `data/interim/regression/` |
| `04_fm_modelling` | the modelling notebook of the classification team on `model_state_with_reg.csv` (about 50 minutes to run), which saves its models to `models/classification/` |
| `05_fm_inference` | the inference notebook on the same models. Its package validation section imports `my_val_capstone_01` and reads the models of `models/classification/` |
| `06_old_vs_new_comparison.md` | the comparison of the classification results with and without the regression columns |

The notebook `02_fm_target_definition` creates `data/processed/model_state.csv`, which regression notebook 02 and classification notebook 03 read.

## Reusable code

The folder `my_val_capstone_01/regression/` holds the Python code of the regression. It is a subpackage of `my_val_capstone_01`, and the front end files of that package are not touched. The files follow the Cookiecutter layout of the template, and each module that creates files has a command line.

| File | Content | Command | Notebook |
|---|---|---|---|
| `config.py` | the folder paths of the regression | none | all |
| `dataset.py` | the household target, the basket simulation and the split of the classification model | `python -m my_val_capstone_01.regression.dataset` | regression 02 |
| `features.py` | the basket features and the five regression columns for the classification states | `python -m my_val_capstone_01.regression.features` | regression 02 and classification 03 |
| `modeling/train.py` | the search, the choice of the number of trees, the blend weight and the margins | `python -m my_val_capstone_01.regression.modeling.train` | regression 03 and 04 |
| `modeling/predict.py` | `predict_household_value` returns the estimate and its 70% interval for one basket, and `checkpoint_for` returns the basket size that a number of documented items reaches | none | new baskets |
| `plots.py` | the validation curves, the blend weight curve and the success rates per basket size | `python -m my_val_capstone_01.regression.plots` | regression 03, 04 and 05 |

Run the commands from the project root. The modules reproduce the outputs of the notebooks exactly, which is checked by `tests/test_regression.py` (the check of the modelling tables is slow and runs when the environment variable `RUN_SLOW_TESTS` is set to 1). The test set is only scored in regression notebook 05.

## Environment

The random search selects different candidates with different XGBoost versions, so XGBoost is pinned to 3.2.0 in `pyproject.toml`. The saved results were produced with XGBoost 3.2.0 and LightGBM 4.7.0.
