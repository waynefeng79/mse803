"""Clean Sample_dataset.csv and compare regression-based missing-value predictions.

Linear (degree 1) and polynomial (degree 2) least-squares regression are
implemented with NumPy. Models are compared using leave-one-out cross-validation
because the cleaned dataset is very small.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
ACT1_DIR = SCRIPT_DIR.parent / "act1"
if str(ACT1_DIR) not in sys.path:
    sys.path.insert(0, str(ACT1_DIR))

# Reuse the cleaning rules developed in act1 instead of maintaining a different
# interpretation of the same source data.
from analyze_data import EXPECTED_COLUMNS, format_csv_value, load_and_clean  # noqa: E402


@dataclass(frozen=True)
class RegressionTask:
    target: str
    predictor: str
    target_to_number: Callable[[Any], float]
    predictor_to_number: Callable[[Any], float]


@dataclass
class PolynomialModel:
    degree: int
    centre: float
    scale: float
    coefficients: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        z = (np.asarray(values, dtype=float) - self.centre) / self.scale
        columns = [np.ones_like(z)] + [z**power for power in range(1, self.degree + 1)]
        return np.column_stack(columns) @ self.coefficients


def date_to_number(value: str) -> float:
    return float(date.fromisoformat(value).toordinal())


def number_to_date(value: float) -> str:
    ordinal = max(1, int(round(value)))
    return date.fromordinal(ordinal).isoformat()


def identity_number(value: Any) -> float:
    return float(value)


TASKS = (
    RegressionTask("Age", "Join Date", identity_number, date_to_number),
    RegressionTask("Salary", "Age", identity_number, identity_number),
    RegressionTask("Net worth", "Salary", identity_number, identity_number),
    RegressionTask("Join Date", "Age", date_to_number, identity_number),
)


def present(value: Any) -> bool:
    return value is not None and value != "" and value != "Unknown"


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> PolynomialModel:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centre = float(x.mean())
    scale = float(x.std())
    if scale == 0:
        scale = 1.0
    z = (x - centre) / scale
    columns = [np.ones_like(z)] + [z**power for power in range(1, degree + 1)]
    coefficients, *_ = np.linalg.lstsq(np.column_stack(columns), y, rcond=None)
    return PolynomialModel(degree, centre, scale, coefficients)


def training_data(rows: list[dict[str, Any]], task: RegressionTask) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (
            task.predictor_to_number(row[task.predictor]),
            task.target_to_number(row[task.target]),
        )
        for row in rows
        if present(row[task.predictor]) and present(row[task.target])
    ]
    return (
        np.asarray([pair[0] for pair in pairs], dtype=float),
        np.asarray([pair[1] for pair in pairs], dtype=float),
    )


def leave_one_out_metrics(x: np.ndarray, y: np.ndarray, degree: int) -> dict[str, float]:
    predictions: list[float] = []
    actual: list[float] = []
    for held_out in range(len(x)):
        mask = np.arange(len(x)) != held_out
        model = fit_polynomial(x[mask], y[mask], degree)
        prediction = float(model.predict(np.asarray([x[held_out]]))[0])
        predictions.append(prediction)
        actual.append(float(y[held_out]))

    errors = np.asarray(predictions) - np.asarray(actual)
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    denominator = float(np.sum((np.asarray(actual) - np.mean(actual)) ** 2))
    r_squared = 1.0 - float(np.sum(errors**2)) / denominator if denominator else 0.0
    return {
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r_squared": r_squared,
    }


def evaluate_models(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], PolynomialModel]]:
    comparisons: dict[str, dict[str, Any]] = {}
    fitted: dict[tuple[str, str], PolynomialModel] = {}

    for task in TASKS:
        x, y = training_data(rows, task)
        if len(x) < 4:
            raise ValueError(f"Not enough complete observations to model {task.target}.")

        linear_metrics = leave_one_out_metrics(x, y, degree=1)
        polynomial_metrics = leave_one_out_metrics(x, y, degree=2)
        selected = (
            "linear"
            if linear_metrics["rmse"] <= polynomial_metrics["rmse"]
            else "polynomial_degree_2"
        )
        comparisons[task.target] = {
            "predictor": task.predictor,
            "training_observations": len(x),
            "validation": "leave-one-out cross-validation",
            "linear": linear_metrics,
            "polynomial_degree_2": polynomial_metrics,
            "selected_model": selected,
        }
        fitted[(task.target, "linear")] = fit_polynomial(x, y, degree=1)
        fitted[(task.target, "polynomial_degree_2")] = fit_polynomial(x, y, degree=2)

    return comparisons, fitted


def infer_sequential_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retrieved: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row["ID"] is not None or index == 0 or index == len(rows) - 1:
            continue
        previous_id = rows[index - 1]["ID"]
        next_id = rows[index + 1]["ID"]
        if isinstance(previous_id, int) and isinstance(next_id, int) and next_id - previous_id == 2:
            row["ID"] = previous_id + 1
            retrieved.append(
                {
                    "name": row["Name"],
                    "field": "ID",
                    "value": row["ID"],
                    "method": "inferred from consecutive IDs on the neighbouring records",
                }
            )
    return retrieved


def format_prediction(target: str, raw_value: float) -> int | str:
    if target == "Join Date":
        return number_to_date(raw_value)
    if target == "Age":
        return max(0, min(120, int(round(raw_value))))
    return max(0, int(round(raw_value)))


def apply_model_set(
    source_rows: list[dict[str, Any]],
    fitted: dict[tuple[str, str], PolynomialModel],
    method_for_target: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = copy.deepcopy(source_rows)
    predictions: list[dict[str, Any]] = []

    # The order is intentional: Heidi's predicted age becomes an input for her
    # salary and net-worth predictions, but predicted values never train a model.
    for task in TASKS:
        method = method_for_target[task.target]
        model = fitted[(task.target, method)]
        for row in rows:
            if present(row[task.target]) or not present(row[task.predictor]):
                continue
            predictor_value = task.predictor_to_number(row[task.predictor])
            raw_prediction = float(model.predict(np.asarray([predictor_value]))[0])
            prediction = format_prediction(task.target, raw_prediction)
            row[task.target] = prediction
            row.setdefault("Predicted Fields", []).append(f"{task.target} ({method})")
            predictions.append(
                {
                    "id": row["ID"],
                    "name": row["Name"],
                    "field": task.target,
                    "predictor": task.predictor,
                    "predictor_value": row[task.predictor],
                    "method": method,
                    "raw_prediction": raw_prediction,
                    "stored_prediction": prediction,
                }
            )
    return rows, predictions


def write_dataset(rows: list[dict[str, Any]], path: Path) -> None:
    fields = (*EXPECTED_COLUMNS, "Predicted Fields")
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: format_csv_value(row.get(field)) for field in EXPECTED_COLUMNS}
            notes = row.get("Predicted Fields", [])
            output["Predicted Fields"] = "; ".join(notes)
            writer.writerow(output)


def create_net_worth_salary_plot(
    source_rows: list[dict[str, Any]],
    fitted: dict[tuple[str, str], PolynomialModel],
    linear_predictions: list[dict[str, Any]],
    polynomial_predictions: list[dict[str, Any]],
    path: Path,
) -> None:
    """Plot observed Net worth/Salary pairs, both fits, and imputed points."""
    observed = [
        row for row in source_rows if present(row["Salary"]) and present(row["Net worth"])
    ]
    linear_points = [item for item in linear_predictions if item["field"] == "Net worth"]
    polynomial_points = [
        item for item in polynomial_predictions if item["field"] == "Net worth"
    ]

    all_salary_values = [float(row["Salary"]) for row in observed]
    all_salary_values.extend(float(item["predictor_value"]) for item in linear_points)
    all_salary_values.extend(float(item["predictor_value"]) for item in polynomial_points)
    lower, upper = min(all_salary_values), max(all_salary_values)
    padding = max((upper - lower) * 0.08, 1_000)
    salary_grid = np.linspace(lower - padding, upper + padding, 250)

    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.scatter(
        [float(row["Salary"]) for row in observed],
        [float(row["Net worth"]) for row in observed],
        color="#2563eb",
        s=58,
        label="Observed records",
        zorder=3,
    )
    for row in observed:
        axis.annotate(
            row["Name"],
            (float(row["Salary"]), float(row["Net worth"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    linear_curve = fitted[("Net worth", "linear")].predict(salary_grid)
    polynomial_curve = fitted[("Net worth", "polynomial_degree_2")].predict(salary_grid)
    axis.plot(salary_grid, linear_curve, color="#dc2626", linewidth=2, label="Linear fit")
    axis.plot(
        salary_grid,
        polynomial_curve,
        color="#16a34a",
        linewidth=2,
        linestyle="--",
        label="Polynomial fit (degree 2)",
    )

    axis.scatter(
        [float(item["predictor_value"]) for item in linear_points],
        [float(item["stored_prediction"]) for item in linear_points],
        color="#dc2626",
        marker="X",
        s=105,
        label="Linear predictions",
        zorder=4,
    )
    axis.scatter(
        [float(item["predictor_value"]) for item in polynomial_points],
        [float(item["stored_prediction"]) for item in polynomial_points],
        color="#16a34a",
        marker="D",
        s=70,
        label="Polynomial predictions",
        zorder=4,
    )
    for item in linear_points:
        axis.annotate(
            f"{item['name']} (linear)",
            (float(item["predictor_value"]), float(item["stored_prediction"])),
            xytext=(7, -14),
            textcoords="offset points",
            fontsize=8,
            color="#991b1b",
        )
    for item in polynomial_points:
        axis.annotate(
            f"{item['name']} (poly)",
            (float(item["predictor_value"]), float(item["stored_prediction"])),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=8,
            color="#166534",
        )

    axis.set_title("Net worth predicted from Salary")
    axis.set_xlabel("Salary")
    axis.set_ylabel("Net worth")
    axis.ticklabel_format(style="plain", axis="both")
    axis.grid(True, alpha=0.22)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def rounded_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: rounded_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded_for_json(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Sample_dataset.csv and compare linear with degree-2 polynomial missing-value predictions."
    )
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "Sample_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, cleaning_summary = load_and_clean(args.input.resolve())
    retrieved = infer_sequential_ids(rows)
    write_dataset(rows, output_dir / "cleaned_before_regression.csv")
    comparisons, fitted = evaluate_models(rows)

    linear_methods = {task.target: "linear" for task in TASKS}
    polynomial_methods = {task.target: "polynomial_degree_2" for task in TASKS}
    selected_methods = {
        target: details["selected_model"] for target, details in comparisons.items()
    }

    linear_rows, linear_predictions = apply_model_set(rows, fitted, linear_methods)
    polynomial_rows, polynomial_predictions = apply_model_set(rows, fitted, polynomial_methods)
    selected_rows, selected_predictions = apply_model_set(rows, fitted, selected_methods)

    write_dataset(linear_rows, output_dir / "linear_predictions.csv")
    write_dataset(polynomial_rows, output_dir / "polynomial_predictions.csv")
    write_dataset(selected_rows, output_dir / "completed_dataset.csv")
    create_net_worth_salary_plot(
        rows,
        fitted,
        linear_predictions,
        polynomial_predictions,
        output_dir / "net_worth_vs_salary_prediction.png",
    )

    results = rounded_for_json(
        {
            "source_file": args.input.name,
            "cleaning_summary": cleaning_summary,
            "retrieved_without_regression": retrieved,
            "model_comparison": comparisons,
            "linear_predictions": linear_predictions,
            "polynomial_predictions": polynomial_predictions,
            "selected_predictions": selected_predictions,
            "unresolved_values": [
                {
                    "id": row["ID"],
                    "name": row["Name"],
                    "fields": [
                        field
                        for field in ("Name", "Country")
                        if row[field] == "Unknown"
                    ],
                    "reason": "Categorical identity values are not suitable targets for these regression models.",
                }
                for row in selected_rows
                if row["Name"] == "Unknown" or row["Country"] == "Unknown"
            ],
        }
    )
    with (output_dir / "model_comparison.json").open("w", encoding="utf-8") as destination:
        json.dump(results, destination, indent=2, ensure_ascii=False)
        destination.write("\n")

    print("Model comparison (leave-one-out RMSE; lower is better):")
    for target, details in comparisons.items():
        print(
            f"- {target}: linear={details['linear']['rmse']:.2f}, "
            f"polynomial={details['polynomial_degree_2']['rmse']:.2f}; "
            f"selected={details['selected_model']}"
        )
    print("\nSelected predictions:")
    for item in selected_predictions:
        print(
            f"- {item['name']} {item['field']}: {item['stored_prediction']} "
            f"using {item['method']}"
        )
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
