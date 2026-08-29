"""Clean the world happiness dataset and make auditable outlier decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
NUMERIC_COLUMNS = [
    "Happiness_Score",
    "GDP_per_Capita",
    "Social_Support",
    "Healthy_Life_Expectancy",
    "Freedom_to_Make_Choices",
    "Generosity",
    "Perceptions_of_Corruption",
]
REQUIRED_COLUMNS = ["Country", *NUMERIC_COLUMNS]
DISPLAY_NAMES = {
    "Happiness_Score": "Happiness score",
    "GDP_per_Capita": "GDP per capita",
    "Social_Support": "Social support",
    "Healthy_Life_Expectancy": "Healthy life expectancy",
    "Freedom_to_Make_Choices": "Freedom",
    "Generosity": "Generosity",
    "Perceptions_of_Corruption": "Perceptions of corruption",
}

# These ranges are validation rules, not statistical outlier thresholds.
# A value outside a range is treated as invalid and is eligible for removal.
VALID_RANGES: dict[str, tuple[float | None, float | None]] = {
    "Happiness_Score": (0.0, 10.0),
    "GDP_per_Capita": (0.0, None),
    "Social_Support": (0.0, 1.0),
    "Healthy_Life_Expectancy": (0.0, 120.0),
    "Freedom_to_Make_Choices": (0.0, 1.0),
    "Generosity": (0.0, 1.0),
    "Perceptions_of_Corruption": (0.0, 1.0),
}
IQR_MULTIPLIER = 1.5
MODIFIED_Z_THRESHOLD = 3.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean world happiness data and detect numeric outliers."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "world_happiness_dataset.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for generated CSV, JSON, and chart files.",
    )
    return parser.parse_args()


def clean_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardise fields, remove unusable rows, and fill numeric gaps."""
    source = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in source]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data = source[REQUIRED_COLUMNS].copy()
    input_rows = len(data)
    data["Country"] = (
        data["Country"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )

    original_numeric_missing = 0
    for column in NUMERIC_COLUMNS:
        original_numeric_missing += int(data[column].isna().sum())
        data[column] = pd.to_numeric(data[column], errors="coerce")

    numeric_missing_after_conversion = int(data[NUMERIC_COLUMNS].isna().sum().sum())
    invalid_numeric_values = numeric_missing_after_conversion - original_numeric_missing

    exact_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    missing_country_rows = int(data["Country"].isna().sum())
    data = data.dropna(subset=["Country"]).copy()

    imputed_by_column: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        missing_count = int(data[column].isna().sum())
        imputed_by_column[column] = missing_count
        if missing_count:
            median = data[column].median()
            if pd.isna(median):
                raise ValueError(f"Cannot impute {column}: the column has no numeric values.")
            data[column] = data[column].fillna(median)

    duplicate_countries = int(data.duplicated(subset="Country", keep=False).sum())
    if duplicate_countries:
        data = data.groupby("Country", as_index=False)[NUMERIC_COLUMNS].mean()

    data = data.sort_values("Country").reset_index(drop=True)
    audit = {
        "input_rows": input_rows,
        "output_rows_before_outlier_decisions": len(data),
        "exact_duplicate_rows_removed": exact_duplicates,
        "rows_without_country_removed": missing_country_rows,
        "invalid_numeric_values_converted_to_missing": invalid_numeric_values,
        "numeric_values_imputed_with_column_median": int(sum(imputed_by_column.values())),
        "imputed_values_by_column": imputed_by_column,
        "duplicate_country_rows_aggregated": duplicate_countries,
    }
    return data, audit


def detect_outliers(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply IQR, modified z-score, and domain validation to each numeric field."""
    enriched = data.copy()
    iqr_count = pd.Series(0, index=data.index, dtype="int64")
    robust_count = pd.Series(0, index=data.index, dtype="int64")
    domain_count = pd.Series(0, index=data.index, dtype="int64")
    flagged_features: list[list[str]] = [[] for _ in range(len(data))]
    report_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for column in NUMERIC_COLUMNS:
        values = data[column]
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        iqr_mask = (values < lower_bound) | (values > upper_bound)

        median = float(values.median())
        mad = float(np.median(np.abs(values - median)))
        if mad == 0:
            modified_z = pd.Series(0.0, index=data.index)
        else:
            modified_z = 0.6745 * (values - median) / mad
        robust_mask = modified_z.abs() > MODIFIED_Z_THRESHOLD

        valid_minimum, valid_maximum = VALID_RANGES[column]
        domain_mask = pd.Series(False, index=data.index)
        if valid_minimum is not None:
            domain_mask |= values < valid_minimum
        if valid_maximum is not None:
            domain_mask |= values > valid_maximum

        iqr_count += iqr_mask.astype(int)
        robust_count += robust_mask.astype(int)
        domain_count += domain_mask.astype(int)
        combined_mask = iqr_mask | robust_mask | domain_mask

        threshold_rows.append(
            {
                "Feature": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "IQR_Lower_Bound": lower_bound,
                "IQR_Upper_Bound": upper_bound,
                "Median": median,
                "MAD": mad,
                "Modified_Z_Threshold": MODIFIED_Z_THRESHOLD,
                "Valid_Minimum": valid_minimum,
                "Valid_Maximum": valid_maximum,
                "IQR_Flag_Count": int(iqr_mask.sum()),
                "Modified_Z_Flag_Count": int(robust_mask.sum()),
                "Domain_Error_Count": int(domain_mask.sum()),
            }
        )

        for index in data.index[combined_mask]:
            flagged_features[index].append(column)
            methods = []
            if bool(iqr_mask.loc[index]):
                methods.append("IQR")
            if bool(robust_mask.loc[index]):
                methods.append("Modified z-score")
            if bool(domain_mask.loc[index]):
                methods.append("Domain validation")

            if bool(domain_mask.loc[index]):
                decision = "DROP"
                rationale = (
                    "The value is outside the defined valid range, so it is treated "
                    "as an invalid record rather than a plausible country difference."
                )
            else:
                decision = "KEEP"
                rationale = (
                    "The value is statistically unusual but remains within its valid "
                    "range. With only 20 diverse countries and no evidence of a data "
                    "entry error, removing it could bias the analysis."
                )

            report_rows.append(
                {
                    "Country": data.at[index, "Country"],
                    "Feature": column,
                    "Value": float(values.loc[index]),
                    "Detection_Methods": "; ".join(methods),
                    "IQR_Lower_Bound": lower_bound,
                    "IQR_Upper_Bound": upper_bound,
                    "Modified_Z_Score": float(modified_z.loc[index]),
                    "Valid_Minimum": valid_minimum,
                    "Valid_Maximum": valid_maximum,
                    "Decision": decision,
                    "Rationale": rationale,
                }
            )

    enriched["IQR_Outlier_Count"] = iqr_count
    enriched["Modified_Z_Outlier_Count"] = robust_count
    enriched["Domain_Error_Count"] = domain_count
    enriched["Flagged_Features"] = ["; ".join(items) for items in flagged_features]
    enriched["Record_Decision"] = np.where(domain_count > 0, "DROP", "KEEP")

    report_columns = [
        "Country",
        "Feature",
        "Value",
        "Detection_Methods",
        "IQR_Lower_Bound",
        "IQR_Upper_Bound",
        "Modified_Z_Score",
        "Valid_Minimum",
        "Valid_Maximum",
        "Decision",
        "Rationale",
    ]
    report = pd.DataFrame(report_rows, columns=report_columns)
    thresholds = pd.DataFrame(threshold_rows)
    return enriched, report, thresholds


def create_boxplots(data: pd.DataFrame, report: pd.DataFrame, path: Path) -> None:
    """Create compact boxplots and highlight statistically flagged observations."""
    figure, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    axes_flat = axes.flatten()

    for axis, column in zip(axes_flat, NUMERIC_COLUMNS):
        axis.boxplot(
            data[column],
            vert=True,
            patch_artist=True,
            boxprops={"facecolor": "#9dc3e6", "edgecolor": "#2f5597"},
            medianprops={"color": "#c00000", "linewidth": 2},
            whiskerprops={"color": "#2f5597"},
            capprops={"color": "#2f5597"},
            flierprops={"marker": "o", "markerfacecolor": "#ed7d31"},
        )
        axis.scatter(np.ones(len(data)), data[column], color="#4472c4", s=18, alpha=0.6)
        flagged = report.loc[report["Feature"] == column]
        for _, row in flagged.iterrows():
            axis.scatter(1, row["Value"], color="#c00000", s=70, zorder=4)
            axis.annotate(
                str(row["Country"]),
                (1, row["Value"]),
                xytext=(8, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(DISPLAY_NAMES[column])
        axis.set_xticks([])
        axis.grid(axis="y", alpha=0.25)

    for axis in axes_flat[len(NUMERIC_COLUMNS) :]:
        axis.axis("off")

    figure.suptitle(
        "World Happiness Outlier Diagnostics\n"
        "Red labels show records flagged by at least one method",
        fontsize=16,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_data, cleaning_audit = clean_dataset(input_path)
    reviewed_data, report, thresholds = detect_outliers(clean_data)
    analysis_ready = reviewed_data.loc[
        reviewed_data["Record_Decision"] == "KEEP", REQUIRED_COLUMNS
    ].copy()

    reviewed_data.to_csv(
        output_dir / "cleaned_with_outlier_flags.csv", index=False, float_format="%.6f"
    )
    analysis_ready.to_csv(
        output_dir / "analysis_ready_dataset.csv", index=False, float_format="%.6f"
    )
    report.to_csv(output_dir / "outlier_report.csv", index=False, float_format="%.6f")
    thresholds.to_csv(
        output_dir / "outlier_thresholds.csv", index=False, float_format="%.6f"
    )
    create_boxplots(clean_data, report, output_dir / "outlier_boxplots.png")

    summary = {
        "source_file": input_path.name,
        "cleaning_audit": cleaning_audit,
        "methods": {
            "iqr": f"Values below Q1 - {IQR_MULTIPLIER}*IQR or above Q3 + {IQR_MULTIPLIER}*IQR.",
            "modified_z_score": (
                f"Median/MAD-based score with absolute threshold {MODIFIED_Z_THRESHOLD}."
            ),
            "domain_validation": "Checks values against documented valid ranges.",
        },
        "rows_flagged_by_any_method": int(
            (reviewed_data[["IQR_Outlier_Count", "Modified_Z_Outlier_Count", "Domain_Error_Count"]].sum(axis=1) > 0).sum()
        ),
        "rows_kept": int((reviewed_data["Record_Decision"] == "KEEP").sum()),
        "rows_dropped": int((reviewed_data["Record_Decision"] == "DROP").sum()),
        "decisions": report[["Country", "Feature", "Value", "Detection_Methods", "Decision"]].to_dict("records"),
    }
    (output_dir / "outlier_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )

    print(f"Input rows: {len(clean_data)}")
    print(f"Rows flagged for review: {summary['rows_flagged_by_any_method']}")
    print(f"Rows kept: {summary['rows_kept']}")
    print(f"Rows dropped: {summary['rows_dropped']}")
    if report.empty:
        print("No outlier candidates were detected.")
    else:
        print("\nOutlier decisions:")
        for row in report.itertuples(index=False):
            print(
                f"- {row.Country}, {row.Feature}={row.Value:.3f}: "
                f"{row.Decision} ({row.Detection_Methods})"
            )
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
