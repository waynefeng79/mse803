# Basic Data Analytics Analysis

This activity cleans `Sample_dataset.csv` and analyses the genuinely numeric variables: `Age`, `Net worth`, and `Salary`. The `ID` column is not analysed because it is an identifier, not a measurement.

## Run the program

Python 3.10 or newer is recommended. The program uses only the Python standard library, so no packages need to be installed.

```powershell
cd week3\act1
python analyze_data.py
```

Optional paths can be supplied:

```powershell
python analyze_data.py --input Sample_dataset.csv --output-dir output
```

The program prints the results and creates:

- `cleaned_dataset.csv` — the cleaned, analysis-ready records.
- `analysis_results.json` — the cleaning audit, descriptive statistics, covariance results, and observation counts.

## Data-cleaning process

The program performs the following steps before calculating statistics:

1. Trims text and standardises names to title case.
2. Converts numeric strings such as `"30,000"` and number words such as `thirty-eight` and `sixty five thousand` into numbers.
3. Rejects impossible ages outside 0–120 and negative monetary values by treating them as missing.
4. Standardises country values to ISO-style codes: `NZ` becomes `NZL`, while `AU` and `AUS` become `AUS`. A missing country becomes `Unknown`.
5. Parses `DD/MM/YYYY`, ISO `YYYY-MM-DD`, and the dataset's `YYYY-DD-MM` variant. For example, `2019-13-01` is interpreted as 13 January 2019. All valid dates are written as ISO `YYYY-MM-DD`; dates that match none of these formats become missing.
6. Replaces a missing name with `Unknown`. Missing analytical numbers remain blank rather than being guessed.
7. Merges duplicate non-missing IDs. A later duplicate fills fields missing in the first record; if both records contain different non-missing values, the first value is retained. Records without an ID are kept as separate observations.

Missing numerical values are handled with **available-case analysis**. Each mean, variance, and standard deviation uses all available values for that variable. Each covariance uses only records containing both variables. The output reports `n` (or paired `n`) so the amount of data behind every result is visible. No mean or median imputation is used because invented values would change the variation and relationships being measured.

## Statistical metrics

### Mean

The **mean** is the arithmetic average:

`mean = sum of the observed values / number of observed values`

It measures the centre or typical level of a numeric variable. For example, a mean salary of 62,625 means the average observed salary is 62,625. The mean is sensitive to unusually large or small values, so it should be interpreted together with the standard deviation and the observation count.

### Sample variance

The **sample variance** measures the average squared distance of observations from their mean. This program divides by `n - 1`, which is appropriate when the dataset is treated as a sample from a wider population.

A small variance means values cluster near the mean; a large variance means they are more spread out. Variance is expressed in squared units—for example, salary variance is in currency units squared—so it is mainly useful for calculation and comparison rather than direct real-world interpretation.

### Sample standard deviation

The **sample standard deviation** is the square root of the sample variance. It measures typical spread around the mean and is expressed in the original unit.

For example, a salary standard deviation of about 5,829 means observed salaries typically vary from the mean by roughly 5,829. A larger standard deviation indicates greater variability. It is not a guaranteed range and does not, by itself, show whether the data are skewed or contain outliers.

### Sample covariance

The **sample covariance** measures whether two variables tend to move together. This program also uses the `n - 1` sample denominator.

- Positive covariance: higher values of one variable tend to occur with higher values of the other.
- Negative covariance: higher values of one variable tend to occur with lower values of the other.
- Covariance near zero: there is little linear co-movement in the available observations.

The magnitude depends on both variables' units, so covariance is best used for its direction and for comparisons made on the same scales. It does not prove causation. A standardised measure such as correlation would be needed to compare relationship strength across variables with different units.

## Key results for `Sample_dataset.csv`

After cleaning, 10 input rows become 9 records because one duplicate ID is merged.

| Variable | Available n | Missing | Mean | Sample variance | Sample standard deviation |
|---|---:|---:|---:|---:|---:|
| Age | 8 | 1 | 30.75 | 40.50 | 6.36 |
| Net worth | 7 | 2 | 38,571.43 | 200,619,047.62 | 14,164.01 |
| Salary | 8 | 1 | 62,625.00 | 31,982,142.86 | 5,655.28 |

| Variable pair | Paired n | Sample covariance | Interpretation |
|---|---:|---:|---|
| Age and Net worth | 7 | 41,857.14 | Positive: older observations tend to have higher net worth in this small sample. |
| Age and Salary | 8 | 22,607.14 | Positive: older observations tend to have higher salaries in this sample. |
| Net worth and Salary | 7 | 12,261,904.76 | Positive: higher net worth tends to occur with higher salary in the paired records, but covariance magnitude is scale-dependent. |

These results describe only this small cleaned dataset. They should not be generalised to a population without more observations and further analysis.
