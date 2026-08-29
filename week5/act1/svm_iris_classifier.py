"""Train and evaluate a three-class SVM classifier on the Iris dataset."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP = SCRIPT_DIR.parent / "iris.zip"
SOURCE_MEMBER = "bezdekIris.data"
FEATURE_COLUMNS = [
    "Sepal_Length_cm",
    "Sepal_Width_cm",
    "Petal_Length_cm",
    "Petal_Width_cm",
]
TARGET_COLUMN = "Species"
EXPECTED_CLASSES = ["setosa", "versicolor", "virginica"]
CLASS_COLORS = ["#4472c4", "#ed7d31", "#70ad47"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a three-class SVC using the Iris dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ZIP,
        help="Path to iris.zip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for model, metrics, predictions, and charts.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0, help="SVC penalty C.")
    return parser.parse_args()


def load_and_clean_iris(zip_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the corrected Iris data member directly from the ZIP and validate it."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        if SOURCE_MEMBER not in archive.namelist():
            raise ValueError(
                f"{SOURCE_MEMBER!r} was not found in {zip_path.name}. "
                f"Available members: {', '.join(archive.namelist())}"
            )
        source_bytes = archive.read(SOURCE_MEMBER)

    data = pd.read_csv(
        io.BytesIO(source_bytes),
        header=None,
        names=[*FEATURE_COLUMNS, TARGET_COLUMN],
        skip_blank_lines=True,
    )
    input_rows = len(data)

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.replace(r"^Iris-", "", regex=True)
        .str.lower()
        .replace("", pd.NA)
    )

    missing_by_column = data.isna().sum().astype(int).to_dict()
    rows_with_missing_values = int(data.isna().any(axis=1).sum())
    data = data.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).copy()

    nonpositive_measurements = int((data[FEATURE_COLUMNS] <= 0).any(axis=1).sum())
    data = data.loc[(data[FEATURE_COLUMNS] > 0).all(axis=1)].copy()

    unexpected_classes = sorted(set(data[TARGET_COLUMN]) - set(EXPECTED_CLASSES))
    if unexpected_classes:
        raise ValueError(f"Unexpected species labels: {', '.join(unexpected_classes)}")
    missing_classes = sorted(set(EXPECTED_CLASSES) - set(data[TARGET_COLUMN]))
    if missing_classes:
        raise ValueError(f"Required species classes are absent: {', '.join(missing_classes)}")

    exact_duplicate_rows = int(data.duplicated().sum())
    data = data.reset_index(drop=True)
    data.insert(0, "Sample_ID", np.arange(1, len(data) + 1))

    audit = {
        "zip_file": zip_path.name,
        "archive_member": SOURCE_MEMBER,
        "input_rows": input_rows,
        "clean_rows": len(data),
        "rows_removed_for_missing_values": rows_with_missing_values,
        "missing_values_by_column": missing_by_column,
        "rows_removed_for_nonpositive_measurements": nonpositive_measurements,
        "exact_duplicate_measurement_rows_retained": exact_duplicate_rows,
        "class_counts": {
            key: int(value)
            for key, value in data[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }
    return data, audit


def make_svm_pipeline(c_value: float = 1.0) -> Pipeline:
    """Scale the four measurements, then fit an RBF-kernel support vector classifier."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma="scale",
                    decision_function_shape="ovr",
                ),
            ),
        ]
    )


def save_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, path: Path
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=EXPECTED_CLASSES)
    figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[name.title() for name in EXPECTED_CLASSES],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Three-Class Iris SVM — Test Confusion Matrix")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_decision_regions(data: pd.DataFrame, c_value: float, path: Path) -> None:
    """Fit a separate two-feature SVM so its decision regions can be plotted."""
    plot_features = ["Petal_Length_cm", "Petal_Width_cm"]
    plot_model = make_svm_pipeline(c_value)
    plot_model.fit(data[plot_features], data[TARGET_COLUMN])

    x_values = data[plot_features[0]]
    y_values = data[plot_features[1]]
    x_margin = (x_values.max() - x_values.min()) * 0.08
    y_margin = (y_values.max() - y_values.min()) * 0.10
    xx, yy = np.meshgrid(
        np.linspace(x_values.min() - x_margin, x_values.max() + x_margin, 500),
        np.linspace(y_values.min() - y_margin, y_values.max() + y_margin, 400),
    )
    grid = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()], columns=plot_features
    )
    grid_labels = plot_model.predict(grid)
    class_to_number = {name: index for index, name in enumerate(EXPECTED_CLASSES)}
    zz = np.array([class_to_number[label] for label in grid_labels]).reshape(xx.shape)

    background = ListedColormap(["#d9e5f6", "#fce4d6", "#e2f0d9"])
    points = ListedColormap(CLASS_COLORS)
    figure, axis = plt.subplots(figsize=(9, 6.8), constrained_layout=True)
    axis.contourf(xx, yy, zz, alpha=0.75, cmap=background)

    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        subset = data.loc[data[TARGET_COLUMN] == class_name]
        axis.scatter(
            subset[plot_features[0]],
            subset[plot_features[1]],
            color=points(class_index),
            edgecolor="white",
            linewidth=0.5,
            s=45,
            label=class_name.title(),
        )

    scaler = plot_model.named_steps["scaler"]
    svc = plot_model.named_steps["svc"]
    support_vectors = scaler.inverse_transform(svc.support_vectors_)
    axis.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        s=100,
        label="Support vector",
    )
    axis.set_xlabel("Petal length (cm)")
    axis.set_ylabel("Petal width (cm)")
    axis.set_title("RBF SVM Decision Regions Using Two Petal Features")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.18)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.c <= 0:
        raise ValueError("--c must be greater than zero.")

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, cleaning_audit = load_and_clean_iris(input_path)
    x_data = data[FEATURE_COLUMNS]
    y_data = data[TARGET_COLUMN]
    sample_ids = data["Sample_ID"]

    x_train, x_test, y_train, y_test, _, id_test = train_test_split(
        x_data,
        y_data,
        sample_ids,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_data,
    )

    model = make_svm_pipeline(args.c)
    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=args.random_state
    )
    cv_scores = cross_val_score(
        model, x_data, y_data, cv=cross_validation, scoring="accuracy"
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    decision_scores = model.decision_function(x_test)
    test_accuracy = float(accuracy_score(y_test, predictions))
    report_dict = classification_report(
        y_test,
        predictions,
        labels=EXPECTED_CLASSES,
        target_names=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    prediction_output = x_test.copy()
    prediction_output.insert(0, "Sample_ID", id_test.astype(int))
    prediction_output["Actual_Species"] = y_test
    prediction_output["Predicted_Species"] = predictions
    prediction_output["Correct"] = prediction_output["Actual_Species"] == predictions
    for index, class_name in enumerate(model.named_steps["svc"].classes_):
        prediction_output[f"Decision_Score_{class_name}"] = decision_scores[:, index]
    prediction_output = prediction_output.sort_values("Sample_ID")

    clean_output = data.copy()
    clean_output.to_csv(output_dir / "cleaned_iris.csv", index=False)
    prediction_output.to_csv(
        output_dir / "test_predictions.csv", index=False, float_format="%.6f"
    )

    report_frame = pd.DataFrame(report_dict).transpose()
    report_frame.loc["accuracy", "precision"] = np.nan
    report_frame.loc["accuracy", "recall"] = np.nan
    report_frame.loc["accuracy", "f1-score"] = test_accuracy
    report_frame.loc["accuracy", "support"] = len(y_test)
    report_frame.index.name = "Class_or_Average"
    report_frame.to_csv(
        output_dir / "classification_report.csv", float_format="%.6f"
    )

    save_confusion_matrix(
        y_test, predictions, output_dir / "confusion_matrix.png"
    )
    save_decision_regions(
        data, args.c, output_dir / "svm_decision_regions.png"
    )
    joblib.dump(model, output_dir / "svm_iris_model.joblib")

    svc = model.named_steps["svc"]
    support_vectors_by_class = {
        class_name: int(count)
        for class_name, count in zip(svc.classes_, svc.n_support_)
    }
    summary = {
        "dataset": cleaning_audit,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "classes": EXPECTED_CLASSES,
        "split": {
            "training_rows": len(x_train),
            "test_rows": len(x_test),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratified": True,
            "training_class_counts": {
                key: int(value)
                for key, value in y_train.value_counts().sort_index().items()
            },
            "test_class_counts": {
                key: int(value)
                for key, value in y_test.value_counts().sort_index().items()
            },
        },
        "model": {
            "estimator": "sklearn.svm.SVC",
            "pipeline": ["StandardScaler", "SVC"],
            "kernel": "rbf",
            "C": args.c,
            "gamma": "scale",
            "multiclass_training_strategy": "one-versus-one",
            "decision_function_shape": "one-versus-rest",
            "support_vectors_by_class": support_vectors_by_class,
            "total_support_vectors": int(sum(svc.n_support_)),
        },
        "evaluation": {
            "test_accuracy": test_accuracy,
            "correct_test_predictions": int((predictions == y_test).sum()),
            "incorrect_test_predictions": int((predictions != y_test).sum()),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=EXPECTED_CLASSES
            ),
            "five_fold_cv_scores": cv_scores,
            "five_fold_cv_mean_accuracy": float(cv_scores.mean()),
            "five_fold_cv_standard_deviation": float(cv_scores.std(ddof=0)),
            "classification_report": report_dict,
        },
        "notes": {
            "visualisation_model": (
                "The decision-region chart uses a separate SVC trained on petal length "
                "and petal width only; the evaluated model uses all four features."
            )
        },
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )

    print("Three-class Iris SVM completed")
    print(f"Dataset: {len(data)} rows; classes={cleaning_audit['class_counts']}")
    print(f"Split: {len(x_train)} training rows, {len(x_test)} test rows")
    print(
        f"Test accuracy: {test_accuracy:.4f} "
        f"({int((predictions == y_test).sum())}/{len(y_test)} correct)"
    )
    print(
        f"5-fold CV accuracy: {cv_scores.mean():.4f} "
        f"+/- {cv_scores.std(ddof=0):.4f}"
    )
    print(f"Support vectors by class: {support_vectors_by_class}")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
