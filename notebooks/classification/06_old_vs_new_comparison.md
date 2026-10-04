# 06. Old versus New: Modelling and Inference Comparison

This document compares the original next-category notebooks of the classification team (**old**) with the same notebooks after the household-value regression features were merged into the modelling dataset and the notebooks were re-run (**new**). All numbers are taken from the saved cell outputs of the notebooks listed below. Nothing was re-estimated for the tables.

|                                    | Old (classification team's original)                                | New (this working copy)                                                              |
| ---------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Modelling notebook                 | `my_val_capstone_01-develop/notebooks/03_fm_modelling.ipynb`      | `notebooks/classification/04_fm_modelling.ipynb`                |
| Inference notebook                 | `my_val_capstone_01-develop/notebooks/04_fm_inference.ipynb`      | `notebooks/classification/05_fm_inference.ipynb`                |
| Modelling dataset                  | `model_state.csv`: 431 rows x 60 columns, 55 predictors per model | `model_state_with_reg.csv`: 431 rows x 65 columns, 60 predictors per model         |
| Environment of the saved outputs   | another machine (`C:\Users\cmcjj\...`, pandas 3 warnings)         | the working copy's Anaconda kernel (Python 3.12.8, pandas 2.2.2, scikit-learn 1.5.1) |
| Run time of the modelling notebook | not recorded                                                        | about 50 minutes (9:28 to 10:18 AM, mostly the two CatBoost searches)                |

## 1. What changed

### 1.1 Data

The new dataset keeps the 431 historical states, the key (`Property_ID`, `Prediction_Time`) and all 60 original columns unchanged, and appends five columns created in `03_ky_regression_feature_merging` from the household-value regression:

| column                  | meaning                                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `documented_value_nv` | documented value of the non-vehicle items at the prediction time (filled for all 431 states)              |
| `reg_pred_total_nv`   | regression estimate of the household's total non-vehicle contents value, never below the documented value |
| `reg_pred_gap_nv`     | estimated value not yet documented                                                                        |
| `reg_completeness_nv` | documented value divided by the estimated total (0 to 1)                                                  |
| `reg_available`       | 1 if at least ten non-vehicle items with a value were documented, else 0                                  |

The estimate is available for **71 of 431 states (16.5%)**: train 53 of 309, validation 10 of 67, test 8 of 55. The other 360 states carry -1 in the three estimate columns. The estimate of a state uses only the earliest 10, 20, 30 or 50 items documented at that time (never a later item), and vehicles are excluded from the regression and from `documented_value_nv`. The classification team's own `documented_value_total` (which includes vehicles) is unchanged.

### 1.2 Split

The train, validation and test partitions are **identical** in both versions: 309 / 67 / 55 rows and 163 / 35 / 36 properties (checked state by state against `state_split_index.csv`: 431 of 431 states match, no property appears in two sets).

### 1.3 Code

Methodology, parameters and model settings are unchanged. The code differences are:

- `04_fm_modelling`: the data path points to `model_state_with_reg.csv` (one line). Two cells were added: a packages cell that installs `catboost` only if it is missing, and a check cell that confirms the five new columns are inputs of every model (60 predictors, 13 of them AI features). The 20 other original code cells are identical.
- `05_fm_inference`: the data path points to `model_state_with_reg.csv` (one line). Two cells were added: a cell that installs `loguru` only if it is missing and a cell that puts the project folder on the import path. The other original code cells are identical.
- In the run, the saved models kept their original paths (`models/next_category_top1_random_forest.joblib`, `models/next_category_ranking_random_forest.joblib`) and were overwritten by the re-run. In this folder structure the same two models are stored in `models/classification/`, and the notebook saves them there.
- Markdown: only result sentences were updated (modelling conclusion, final conclusion). Titles, structure and style are unchanged, and "Final Deliverables" is exactly the original.

## 2. Modelling comparison (validation set, 67 states)

### 2.1 All experiments

Majority-class baseline (validation accuracy / Macro F1 / Weighted F1): old 0.2239 / 0.0610 / 0.0819, new 0.2239 / 0.0610 / 0.0819 (identical, as expected).

**Validation accuracy**

| Model                             | Old    | New    | Difference |
| --------------------------------- | ------ | ------ | ---------- |
| Logistic Regression               | 0.3582 | 0.3582 | +0.0000    |
| CatBoost Tuned                    | 0.4478 | 0.4776 | +0.0298    |
| CatBoost Regularised              | 0.4328 | 0.4776 | +0.0448    |
| Random Forest Regularised         | 0.4328 | 0.4776 | +0.0448    |
| Random Forest Regularised - No AI | 0.3881 | 0.3582 | -0.0299    |

**Validation Macro F1**

| Model                             | Old    | New    | Difference |
| --------------------------------- | ------ | ------ | ---------- |
| Logistic Regression               | 0.3353 | 0.3360 | +0.0007    |
| Random Forest                     | 0.4205 | 0.4132 | -0.0073    |
| CatBoost Tuned                    | 0.3261 | 0.4288 | +0.1027    |
| CatBoost Regularised              | 0.3384 | 0.4190 | +0.0806    |
| Random Forest Regularised         | 0.3676 | 0.4245 | +0.0569    |
| Random Forest Regularised - No AI | 0.3324 | 0.2954 | -0.0370    |

**Validation Top-2 accuracy**

| Model                             | Old    | New    | Difference |
| --------------------------------- | ------ | ------ | ---------- |
| Logistic Regression               | 0.5672 | 0.5672 | +0.0000    |
| Random Forest                     | 0.6567 | 0.6119 | -0.0448    |
| CatBoost Tuned                    | 0.7015 | 0.7313 | +0.0298    |
| CatBoost Regularised              | 0.7015 | 0.7313 | +0.0298    |
| Random Forest Regularised         | 0.7164 | 0.6716 | -0.0448    |
| Random Forest Regularised - No AI | 0.7313 | 0.6567 | -0.0746    |

**Validation Top-3 accuracy**

| Model                             | Old    | New    | Difference |
| --------------------------------- | ------ | ------ | ---------- |
| Logistic Regression               | 0.8060 | 0.7910 | -0.0150    |
| Random Forest                     | 0.8060 | 0.7612 | -0.0448    |
| CatBoost Tuned                    | 0.8209 | 0.8358 | +0.0149    |
| CatBoost Regularised              | 0.8060 | 0.8358 | +0.0298    |
| Random Forest Regularised         | 0.8358 | 0.8060 | -0.0298    |
| Random Forest Regularised - No AI | 0.8209 | 0.7910 | -0.0299    |

**Macro F1 gap between train and validation (smaller is better)**

| Model                             | Old    | New    | Difference |
| --------------------------------- | ------ | ------ | ---------- |
| Logistic Regression               | 0.5997 | 0.5990 | -0.0007    |
| Random Forest                     | 0.5795 | 0.5868 | +0.0073    |
| CatBoost Tuned                    | 0.5403 | 0.3963 | -0.1440    |
| CatBoost Regularised              | 0.3605 | 0.3546 | -0.0059    |
| Random Forest Regularised         | 0.4372 | 0.3884 | -0.0488    |
| Random Forest Regularised - No AI | 0.4792 | 0.5311 | +0.0519    |

**Training Macro F1 (how much each model fits the training states)**

| Model                             | Old    | New    |
| --------------------------------- | ------ | ------ |
| Logistic Regression               | 0.9350 | 0.9350 |
| Random Forest                     | 1.0000 | 1.0000 |
| CatBoost Tuned                    | 0.8664 | 0.8251 |
| CatBoost Regularised              | 0.6989 | 0.7736 |
| Random Forest Regularised         | 0.8047 | 0.8129 |
| Random Forest Regularised - No AI | 0.8116 | 0.8264 |

### 2.2 Hyperparameters chosen by the searches

The searches are randomised (40 candidates, grouped 5-fold cross-validation), so the selected settings differ between runs.

| Search                    | Best CV Macro F1 (old / new) | Old parameters                                                                                            | New parameters                                                                                           |
| ------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Tuned CatBoost            | 0.4302 / 0.4217              | random_strength 2.0, learning_rate 0.03, l2_leaf_reg 5, iterations 300, depth 8, bagging_temperature 0.5  | random_strength 0.5, learning_rate 0.01, l2_leaf_reg 5, iterations 800, depth 6, bagging_temperature 0.5 |
| Regularised CatBoost      | 0.4155 / 0.4259              | random_strength 5.0, learning_rate 0.03, l2_leaf_reg 10, iterations 600, depth 4, bagging_temperature 5.0 | random_strength 1.0, learning_rate 0.03, l2_leaf_reg 5, iterations 600, depth 3, bagging_temperature 1.0 |
| Regularised Random Forest | 0.4127 / 0.4149              | n_estimators 800, min_samples_split 15, min_samples_leaf 1, max_features sqrt, max_depth None             | n_estimators 500, min_samples_split 5, min_samples_leaf 4, max_features 0.5, max_depth 7                 |

### 2.3 Feature importance

Top five by validation permutation importance:

| Rank | Old                                        | New                                   |
| ---- | ------------------------------------------ | ------------------------------------- |
| 1    | `documented_count_CAT-001` (0.1333)      | `documented_count_CAT-001` (0.1774) |
| 2    | `documented_count_CAT-006` (0.0704)      | `documented_count_CAT-005` (0.1340) |
| 3    | `documented_count_CAT-005` (0.0392)      | `documented_count_CAT-006` (0.1295) |
| 4    | `ai_suggestion_acceptance_rate` (0.0355) | `documented_count_CAT-003` (0.1281) |
| 5    | `documented_count_CAT-003` (0.0354)      | `documented_count_CAT-002` (0.0968) |

Position of the new columns in the new run (top 25 lists only): `reg_pred_total_nv`: permutation rank 20 (0.0268), Random Forest importance rank outside the top 25, `reg_pred_gap_nv`: permutation rank 21 (0.0268), Random Forest importance rank outside the top 25, `reg_completeness_nv`: permutation rank outside the top 25, Random Forest importance rank outside the top 25, `reg_available`: permutation rank outside the top 25, Random Forest importance rank outside the top 25, `documented_value_nv`: permutation rank outside the top 25, Random Forest importance rank 15 (0.0227).

The most important features are the same kind in both runs: the counts of documented assets per category (`documented_count_CAT-00x`) and the AI suggestion signals. The regression columns are well below them.

### 2.4 AI feature ablation

| Regularised Random Forest | Validation accuracy (old / new) | Validation Macro F1 (old / new) |
| ------------------------- | ------------------------------- | ------------------------------- |
| with all features         | 0.4328 / 0.4776                 | 0.3676 / 0.4245                 |
| without AI features       | 0.3881 / 0.3582                 | 0.3324 / 0.2954                 |

Removing the AI features lowered validation Macro F1 in both runs (old 0.3676 to 0.3324, new 0.4245 to 0.2954). The drop is larger in the new run.

## 3. Final holdout evaluation (test set, 55 states, scored once per run)

| Measure                                    | Old    | New    | Difference | States correct (old / new) |
| ------------------------------------------ | ------ | ------ | ---------- | -------------------------- |
| Top-1 accuracy (Random Forest)             | 0.5091 | 0.5091 | +0.0000    | 28 / 28                    |
| Top-1 Macro F1                             | 0.3223 | 0.3257 | +0.0034    | -                          |
| Top-1 Weighted F1                          | 0.4431 | 0.4516 | +0.0085    | -                          |
| Top-2 accuracy (regularised Random Forest) | 0.7091 | 0.7818 | +0.0727    | 39 / 43                    |
| Top-3 accuracy (regularised Random Forest) | 0.8364 | 0.8545 | +0.0181    | 46 / 47                    |

One state out of 55 equals 1.8 percentage points, so the Top-2 change corresponds to four states and the Top-3 change to one state.

### 3.1 Top-1 per category

| Category                     | Support | Recall old | Recall new | F1 old | F1 new | Correct old | Correct new |
| ---------------------------- | ------- | ---------- | ---------- | ------ | ------ | ----------- | ----------- |
| Appliances                   | 11      | 0.3636     | 0.2727     | 0.4000 | 0.3158 | 4           | 3           |
| Clothings                    | 3       | 0.0000     | 0.0000     | 0.0000 | 0.0000 | 0           | 0           |
| Electronics                  | 12      | 1.0000     | 1.0000     | 0.7742 | 0.8000 | 12          | 12          |
| Furniture                    | 12      | 0.7500     | 0.8333     | 0.6207 | 0.6897 | 9           | 10          |
| Jewellery and Personal Items | 8       | 0.0000     | 0.1250     | 0.0000 | 0.1667 | 0           | 1           |
| Other Items                  | 7       | 0.4286     | 0.2857     | 0.4615 | 0.3077 | 3           | 2           |
| Vehicles                     | 2       | 0.0000     | 0.0000     | 0.0000 | 0.0000 | 0           | 0           |

Electronics (all 12 correct in both runs) and Furniture are predicted most reliably. Clothings (3 states) and Vehicles (2 states) are never predicted correctly in either run. Jewellery and Personal Items improves from 0 to 1 correct state of 8.

## 4. Inference comparison

Both inference notebooks load the two saved Random Forests, recreate the test split (55 states, 36 properties, identical in both) and predict one test state (PRP-000040, actual category Other Items).

|                                                    | Old                                                                                                                   | New                                                                                                                                                                                                                                                          |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Top-1 recommendation (standard Random Forest)      | Electronics (33.60%)                                                                                                  | Electronics (37.40%)                                                                                                                                                                                                                                         |
| Ranked recommendations (regularised Random Forest) | Electronics (31.42%), Other Items (21.99%), Clothings (15.24%)                                                        | Electronics (68.05%), Other Items (15.82%), Jewellery and Personal Items (5.31%)                                                                                                                                                                             |
| Notebook function and package function agree       | yes                                                                                                                   | yes                                                                                                                                                                                                                                                          |
| Package module loaded from                         | `C:\Users\cmcjj\OneDrive - UTS\Documents\2026-2\capstone\my_val_capstone_01\my_val_capstone_01\modeling\predict.py` | `my_val_capstone_01/modeling/predict.py` in the project folder of the working copy |

In both runs the notebook function and the package function return identical predictions. The actual category (Other Items) is second in both rankings. The two models are different (a standard and a regularised Random Forest), so the Top-1 probability (37.40%) and the first-ranked probability (68.05%) of the same category are not comparable. The probabilities changed because the saved models are new.

The module printed in the old output contained objects that the module in this working copy does not have (`PROCESSED_DATA_DIR`, `app`, `logger`, `main`, `tqdm`, `typer`), so the old output was produced by a different version of `predict.py` than the file in the repository copy. The reason was not investigated. The inference logic (drop the three identifier columns, predict, rank) is the same.

## 5. Differences in short

- **Unchanged:** the data, the split, the methodology, the Top-1 holdout accuracy (0.5091, 28 of 55 states), the categories that are easy and hard, the overfitting pattern (the default Random Forest fits the training data completely, Macro F1 1.0000).
- **Higher in the new run:** validation Macro F1 of the three tuned or regularised models (CatBoost Tuned 0.3261 to 0.4288, CatBoost Regularised 0.3384 to 0.4190, Random Forest Regularised 0.3676 to 0.4245) and of validation accuracy (0.4328 to 0.4776 for the regularised models). Holdout Top-2 (0.7091 to 0.7818) and Top-3 (0.8364 to 0.8545).
- **Slightly lower:** the default Random Forest validation Macro F1 (0.4205 to 0.4132) and the Top-1 per-category recall of Appliances (0.3636 to 0.2727) and Other Items (0.4286 to 0.2857).
- **Different selected settings:** all three searches picked different hyperparameters (section 2.2).
- **Same answer for the sample prediction:** the first recommendation is Electronics in both runs, with different probabilities.

## 6. Issues and limitations

1. **The comparison is not controlled.** The old outputs come from another machine and other library versions. The same code and data can give different numbers on different setups. Example: the default Random Forest on the 55 original predictors gave validation accuracy 0.4776 and Macro F1 0.4205 in the old output, but 0.4030 and 0.3215 when re-run in the working copy's environment. A fair test of the regression features therefore needs the same notebook run on the same environment with and without the five columns.
2. **A controlled check shows no consistent benefit of the regression features.** Run outside the notebooks, in the working copy's environment with the same seeds and only the five columns different: default Random Forest, validation (67 states): accuracy 0.4776 with and 0.4030 without. Over 20 random seeds the mean validation accuracy was 0.4478 with and 0.4455 without, Macro F1 0.3675 with and 0.3702 without, Top-2 0.6634 with and 0.6687 without, Top-3 0.7881 with and 0.8022 without. The differences are smaller than the variation between seeds (about 0.02). Final holdout of the Random Forest parts of the pipeline (default Random Forest and the regularised Random Forest search. The CatBoost cells were not part of this check) with identical settings: with the columns Top-1 0.5091, Macro F1 0.3257, Top-2 0.7818, Top-3 0.8545. Without them 0.5273, 0.3463, 0.7455, 0.8364 (a difference of one or two states in each measure, in both directions). The improvements against the old output should therefore not be attributed to the regression features. They are more likely to come from different search results and a different environment. These comparison scripts are not part of the notebooks, so these specific figures cannot be reproduced from the notebook cells.
3. **Very small samples.** The validation set has 67 states and the test set 55 (36 properties). One state is 1.5 and 1.8 percentage points. Only 10 validation and 8 test states carry a regression estimate, and the minority categories have 2 to 8 test states (Vehicles 2, Clothings 3).
4. **Limited reach of the regression feature.** The estimate needs at least ten documented items, which only 16.5% of the states have. The other states carry -1 (the Random Forests handle the sentinel, the logistic regression baseline scales it, which can distort that baseline slightly). Many households (636 of 795 properties) have fewer than ten items, so the regression cannot support the earlier states, where a recommendation is most needed.
5. **In-sample estimates for the training states.** The regression models were fitted on the regression team's training households, whose label is the household's full total (which includes items documented after the state). The estimates of the 53 training states are therefore in-sample, probably more accurate than those of the validation and test states and able to carry some information from the future. No out-of-fold step was done on purpose (the task was only to merge the model as features). The flag `pred_is_in_sample` is in the diagnostics file `data/interim/regression/regression_state_diagnostics.csv`. Validation and test households were never used to fit the regression models (checked by property).
6. **Model selection.** The final models are the standard Random Forest (Top-1) and the regularised Random Forest (ranking) as in the original design. In the new run the tuned CatBoost has the highest validation Macro F1 (0.4288) and the two CatBoost models the highest validation Top-2 and Top-3 (0.7313 and 0.8358), so the retained models are not the best on every validation measure. With the small samples this may not be a real difference.
7. **Randomness of the searches and overfitting.** Randomised searches with 40 candidates on 309 training states select different settings from run to run. All tree-based models still fit the training data much better than the validation data (Macro F1 gaps between train and validation of 0.35 to 0.60 in the new run).
8. **The test set was scored in each run of the modelling notebook** (as the notebook is designed) and again in the controlled comparison. It should not be used to choose between model variants or to decide on the regression features.
9. **The inference check is one example, not a performance test.** The notebooks predict a single test state (PRP-000040) to confirm that the saved models load and the functions work. The performance numbers are those of section 3.
10. **Environment dependencies.** The new notebooks install `catboost` (modelling) and `loguru` (inference) if missing. The package `my_val_capstone_01` is imported from the working copy through the import path instead of an installation. The old outputs were produced with an installed copy on another machine.

## 7. Checks that were done

- The five columns were recomputed independently from the raw workbook (40 checks passed): `documented_value_nv` plus the recorded vehicle value equals the classification team's `documented_value_total` in all 431 states. The estimate uses only items documented by the prediction time. States of the same household and basket size share one estimate. Padding is -1 only where `reg_available` is 0. The 71 estimates equal a fresh recomputation with the saved models.
- The original 60 columns of `model_state_with_reg.csv` are identical to `model_state.csv`.
- The split reproduces `state_split_index.csv` exactly (section 1.2).
- Code of the two notebooks compared with the original notebooks (section 1.3). The saved models of the new run take 60 input columns including the five regression columns.
- Both notebooks were run end to end in the working copy (04 about 50 minutes. 05 with no errors, the notebook function and the package function agree).

## 8. Conclusion

The regression features were merged correctly and without changing the classification team's methodology, split or saved-model paths, and both notebooks run end to end. The new results are higher than the old ones on several validation measures and on holdout Top-2 and Top-3 accuracy, but the controlled checks indicate that these changes probably come from different search results and a different environment rather than from the regression features. With only 71 of 431 states carrying an estimate (10 validation and 8 test states), the current data cannot show a benefit. The evidence supports keeping the household-value regression as a standalone estimate (total value and 70% interval) and treating its use as a classifier feature as an option to re-test when more household histories are available.
