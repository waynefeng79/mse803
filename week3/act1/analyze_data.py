"""Clean Sample_dataset.csv and calculate basic descriptive statistics.

The program uses only Python's standard library. It writes a cleaned CSV and a
machine-readable JSON results file, then prints the key results to the console.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


NUMERIC_COLUMNS = ("Age", "Net worth", "Salary")
EXPECTED_COLUMNS = ("ID", "Name", "Age", "Net worth", "Country", "Salary", "Join Date")

SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def parse_number(value: str) -> float | None:
    """Convert digits or simple English number words to a float."""
    text = value.strip().lower()
    if not text:
        return None

    compact = re.sub(r"[$,\s]", "", text)
    try:
        number = float(compact)
        return number if math.isfinite(number) else None
    except ValueError:
        pass

    words = re.findall(r"[a-z]+", text.replace("-", " "))
    if not words:
        return None

    current = 0
    total = 0
    for word in words:
        if word == "and":
            continue
        if word in SMALL_NUMBERS:
            current += SMALL_NUMBERS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word in {"thousand", "million"}:
            scale = 1_000 if word == "thousand" else 1_000_000
            total += max(current, 1) * scale
            current = 0
        else:
            return None
    return float(total + current)


def clean_integer(value: str, minimum: int, maximum: int | None = None) -> int | None:
    number = parse_number(value)
    if number is None or not number.is_integer():
        return None
    integer = int(number)
    if integer < minimum or (maximum is not None and integer > maximum):
        return None
    return integer


def clean_money(value: str) -> float | None:
    number = parse_number(value)
    return number if number is not None and number >= 0 else None


def clean_country(value: str) -> str:
    aliases = {
        "NZ": "NZL",
        "NZL": "NZL",
        "NEW ZEALAND": "NZL",
        "AU": "AUS",
        "AUS": "AUS",
        "AUSTRALIA": "AUS",
    }
    text = value.strip().upper()
    return aliases.get(text, text or "Unknown")


def clean_date(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    # Try conventional formats first, then accept the dataset's YYYY-DD-MM
    # variant (for example, 2019-13-01 means 13 January 2019).
    for date_format in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%d-%m"):
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    return ""


def clean_row(raw: dict[str, str]) -> dict[str, Any]:
    return {
        "ID": clean_integer(raw.get("ID", ""), minimum=1),
        "Name": raw.get("Name", "").strip().title() or "Unknown",
        "Age": clean_integer(raw.get("Age", ""), minimum=0, maximum=120),
        "Net worth": clean_money(raw.get("Net worth", "")),
        "Country": clean_country(raw.get("Country", "")),
        "Salary": clean_money(raw.get("Salary", "")),
        "Join Date": clean_date(raw.get("Join Date", "")),
    }


def is_missing(value: Any) -> bool:
    return value is None or value == "" or value == "Unknown"


def merge_duplicate_ids(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge duplicate IDs, filling blanks from later records; first conflicts win."""
    merged: list[dict[str, Any]] = []
    positions: dict[int, int] = {}
    duplicate_count = 0

    for row in rows:
        row_id = row["ID"]
        if row_id is None or row_id not in positions:
            if row_id is not None:
                positions[row_id] = len(merged)
            merged.append(row)
            continue

        duplicate_count += 1
        existing = merged[positions[row_id]]
        for column in EXPECTED_COLUMNS:
            if is_missing(existing[column]) and not is_missing(row[column]):
                existing[column] = row[column]

    return merged, duplicate_count


def load_and_clean(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("The CSV file does not contain a header row.")
        missing_columns = [column for column in EXPECTED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        raw_rows = list(reader)

    cleaned_rows = [clean_row(row) for row in raw_rows]
    cleaned_rows, duplicates_merged = merge_duplicate_ids(cleaned_rows)
    audit = {
        "input_rows": len(raw_rows),
        "cleaned_rows": len(cleaned_rows),
        "duplicate_rows_merged": duplicates_merged,
    }
    return cleaned_rows, audit


def available_values(rows: Iterable[dict[str, Any]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row[column] is not None]


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    descriptive: dict[str, dict[str, float | int | None]] = {}
    for column in NUMERIC_COLUMNS:
        values = available_values(rows, column)
        descriptive[column] = {
            "count": len(values),
            "missing_count": len(rows) - len(values),
            "mean": statistics.mean(values) if values else None,
            "sample_variance": statistics.variance(values) if len(values) >= 2 else None,
            "sample_standard_deviation": statistics.stdev(values) if len(values) >= 2 else None,
        }

    covariances: list[dict[str, Any]] = []
    for left_index, left in enumerate(NUMERIC_COLUMNS):
        for right in NUMERIC_COLUMNS[left_index + 1 :]:
            pairs = [
                (float(row[left]), float(row[right]))
                for row in rows
                if row[left] is not None and row[right] is not None
            ]
            covariances.append(
                {
                    "variables": [left, right],
                    "pair_count": len(pairs),
                    "sample_covariance": (
                        statistics.covariance(
                            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                        )
                        if len(pairs) >= 2
                        else None
                    ),
                }
            )

    return {
        "method": "Available-case analysis using sample variance, sample standard deviation, and sample covariance (n - 1 denominator).",
        "descriptive_statistics": descriptive,
        "covariances": covariances,
    }


def format_csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def write_cleaned_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row[key]) for key in EXPECTED_COLUMNS})


def write_results(results: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as destination:
        json.dump(results, destination, indent=2, ensure_ascii=False)
        destination.write("\n")


def print_results(results: dict[str, Any]) -> None:
    print("\nDESCRIPTIVE STATISTICS (sample statistics)\n")
    print(f"{'Variable':<12} {'n':>3} {'Missing':>7} {'Mean':>14} {'Variance':>16} {'Std. dev.':>14}")
    for column, metrics in results["descriptive_statistics"].items():
        print(
            f"{column:<12} {metrics['count']:>3} {metrics['missing_count']:>7} "
            f"{metrics['mean']:>14,.2f} {metrics['sample_variance']:>16,.2f} "
            f"{metrics['sample_standard_deviation']:>14,.2f}"
        )

    print("\nSAMPLE COVARIANCE (pairwise complete observations)\n")
    for item in results["covariances"]:
        left, right = item["variables"]
        print(
            f"{left} vs {right}: {item['sample_covariance']:,.2f} "
            f"(paired n = {item['pair_count']})"
        )


def parse_args() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Clean a CSV dataset and calculate mean, variance, standard deviation, and covariance."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_directory / "Sample_dataset.csv",
        help="Input CSV path (default: Sample_dataset.csv beside this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_directory,
        help="Directory for cleaned_dataset.csv and analysis_results.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    rows, audit = load_and_clean(input_path)
    analysis = analyse(rows)
    results = {
        "source_file": input_path.name,
        "cleaning_summary": audit,
        **analysis,
    }

    cleaned_path = output_directory / "cleaned_dataset.csv"
    results_path = output_directory / "analysis_results.json"
    write_cleaned_csv(rows, cleaned_path)
    write_results(results, results_path)

    print(f"Cleaned data written to: {cleaned_path}")
    print(f"Analysis results written to: {results_path}")
    print_results(results)


if __name__ == "__main__":
    main()
