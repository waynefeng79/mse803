# Missing-Value Prediction with Regression

This activity continues the cleaning of `Sample_dataset.csv` from act1 and attempts to recover or predict missing values. It compares **linear regression** with **degree-2 polynomial regression** and retains a clear record of every estimated value.

## Run the program

From the repository root:

```powershell
python -m pip install -r week3\act2\requirements.txt
python week3\act2\predict_missing_values.py
```

The script uses `Sample_dataset.csv` in this folder by default. Alternative paths can be supplied with `--input` and `--output-dir`.

## Cleaning and retrieval before regression

The program reuses the act1 cleaning rules so both activities interpret the source consistently:

- Numeric text is converted to numbers, including `"30,000"`, `thirty-eight`, and `sixty five thousand`.
- Countries are standardised to `NZL` and `AUS` where known.
- Dates are stored as `YYYY-MM-DD`; `2019-13-01` is interpreted as 13 January 2019 and becomes `2019-01-13`.
- Bob's duplicate ID 2 rows are merged. His age and net worth are retrieved from one row and his salary from the other, so regression is unnecessary for those fields.
- Eve's missing ID is inferred as 6 because it appears between IDs 5 and 7.
- Missing names and countries remain `Unknown`. Linear and polynomial regression predict numeric quantities, not identity or categorical labels, so inventing these values would be misleading.

Ten source rows become nine cleaned records after the duplicate is merged.

## Regression approaches

### Linear regression

Linear regression fits a straight-line relationship:

`predicted y = b0 + b1 × x`

It assumes that a one-unit change in the predictor has a constant effect on the target. With very little data, a linear model is relatively stable because it estimates only an intercept and one slope.

### Polynomial regression

The nonlinear approach follows the supplied `nonlinear regression -sample code.py`. It creates a squared feature and fits:

`predicted y = b0 + b1 × x + b2 × x²`

This permits a curved relationship. The implementation uses NumPy least squares rather than scikit-learn, but the fitted degree-2 model is mathematically equivalent to applying `PolynomialFeatures(degree=2)` and then fitting `LinearRegression` as shown in the sample.

The additional squared term can model genuine curvature, but it also increases the risk of fitting noise when there are only seven or eight complete observations.

## What is predicted

Simple one-predictor models are used because the dataset is too small to support reliable models with many features.

| Missing target | Predictor | Complete training observations | Reason |
|---|---|---:|---|
| Heidi's age | Join date converted to a day number | 7 | Heidi has a valid join date but no other numeric values. |
| Heidi's salary | Age | 8 | Salary and age have eight observed pairs. The model-specific predicted age is used for Heidi. |
| David's and Heidi's net worth | Salary | 7 | Salary is observed for David and is first predicted for Heidi, allowing the requested Net worth-versus-Salary comparison. |
| Charlie's join date | Age | 7 | Dates are modelled as ordinal day numbers and converted back to ISO dates. |

The models are trained only on originally observed values. Predicted values can be used as inputs in Heidi's sequence of predictions, but they are never added to the training data.

## Model evaluation

Because the dataset is very small, a normal train/test split would leave too few records for training. The program therefore uses **leave-one-out cross-validation (LOOCV)**:

1. Hold out one complete observation.
2. Fit the model to all remaining observations.
3. Predict the held-out value.
4. Repeat until every observation has been held out once.

The main comparison measure is **root mean squared error (RMSE)**. Lower RMSE means predictions were closer to the known values during validation. Salary and net-worth RMSE values use their monetary units; age uses years; join date uses days.

| Target | Linear RMSE | Polynomial RMSE | Better model |
|---|---:|---:|---|
| Age | 5.94 years | 16.11 years | Linear |
| Salary | 5,788.11 | 9,077.55 | Linear |
| Net worth | 16,632.78 | 20,169.97 | Linear |
| Join date | 379.45 days | 517.91 days | Linear |

Linear regression has the lower validation error for every target. The polynomial model is worse because its extra squared term overfits this small and irregular dataset. Linear regression therefore provides the better predictions **relative to the polynomial alternative**.

This does not mean the linear predictions are highly reliable. The cross-validated R² is only 0.06 for age and is negative for salary, net worth, and join date. A negative validation R² means the held-out predictions are worse than simply predicting the observed target mean. The estimates are useful for demonstrating the two techniques, but they should be treated as low-confidence imputations rather than known facts.

## Predicted values from both approaches

| Record and missing field | Linear prediction | Polynomial prediction | Selected value |
|---|---:|---:|---:|
| Heidi — Age | 22 | 23 | 22 |
| Heidi — Salary | 57,741 | 59,459 | 57,741 |
| David — Net worth | 40,939 | 41,402 | 40,939 |
| Heidi — Net worth | 36,985 | 38,183 | 36,985 |
| Charlie — Join date | 2019-04-05 | 2018-12-16 | 2019-04-05 |

The selected values come from linear regression because it performs better in LOOCV for all four targets. Values are rounded to sensible dataset units: whole years, whole monetary units, and whole dates.

## Net worth versus Salary visualization

The program generates `net_worth_vs_salary_prediction.png` automatically. The chart contains:

- Blue circles for records with observed Salary and Net worth.
- A solid red line for the fitted linear relationship.
- A dashed green curve for the degree-2 polynomial relationship.
- Red crosses and green diamonds for David's and Heidi's model-specific Net worth estimates.

The chart shows that observed Net worth values are widely dispersed around both fitted relationships. This visual pattern agrees with the negative cross-validated R² values and reinforces that the imputed values are low-confidence estimates.

## Generated files

- `cleaned_before_regression.csv` shows the standardised and retrieved data before any regression estimates are inserted.
- `linear_predictions.csv` contains a completed numeric/date dataset using only linear regression predictions.
- `polynomial_predictions.csv` contains the comparable degree-2 polynomial predictions.
- `completed_dataset.csv` uses the model with the lower validation RMSE for each target. Its `Predicted Fields` column distinguishes estimates from observed data.
- `model_comparison.json` records cleaning counts, retrieved values, validation metrics, raw predictions, stored predictions, model choices, and unresolved categorical values.
- `net_worth_vs_salary_prediction.png` visualizes the observed data, regression fits, and predicted Net worth values.

The two unresolved categorical values are the name for ID 7 and Grace's country. They are deliberately retained as `Unknown` rather than guessed.
