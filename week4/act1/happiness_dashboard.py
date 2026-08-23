"""Create Matplotlib and Plotly dashboards from the cleaned happiness dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
COMPOSITE_FIELDS = [
    "Happiness_Score",
    "Healthy_Life_Expectancy",
    "Freedom_to_Make_Choices",
]
NORMALIZED_FIELDS = {
    "Happiness_Score": "Happiness_Normalized",
    "Healthy_Life_Expectancy": "Health_Normalized",
    "Freedom_to_Make_Choices": "Freedom_Normalized",
}
DISPLAY_NAMES = {
    "Happiness_Normalized": "Happiness",
    "Health_Normalized": "Healthy life expectancy",
    "Freedom_Normalized": "Freedom",
}
CORRELATION_LABELS = [
    "Happiness",
    "GDP",
    "Support",
    "Health",
    "Freedom",
    "Generosity",
    "Corruption",
]
COLORS = ["#4472c4", "#ed7d31", "#70ad47"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create static and interactive world-happiness dashboards."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "world_happiness_dataset.csv",
        help="Cleaned input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for dashboard outputs.",
    )
    return parser.parse_args()


def load_clean_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate, standardise, fill gaps, and aggregate to one row per country."""
    source = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data = source[REQUIRED_COLUMNS].copy()
    data["Country"] = (
        data["Country"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    exact_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    missing_country_rows = int(data["Country"].isna().sum())
    data = data.dropna(subset=["Country"]).copy()

    missing_numeric_cells = int(data[NUMERIC_COLUMNS].isna().sum().sum())
    if missing_numeric_cells:
        data[NUMERIC_COLUMNS] = data[NUMERIC_COLUMNS].fillna(
            data[NUMERIC_COLUMNS].mean()
        )

    rows_before_aggregation = len(data)
    data = (
        data.groupby("Country", as_index=False, observed=True)[NUMERIC_COLUMNS]
        .mean()
        .sort_values("Country")
        .reset_index(drop=True)
    )
    audit = {
        "source_rows": len(source),
        "exact_duplicates_removed": exact_duplicates,
        "rows_without_country_removed": missing_country_rows,
        "numeric_cells_mean_filled": missing_numeric_cells,
        "rows_before_country_aggregation": rows_before_aggregation,
        "countries_after_aggregation": len(data),
    }
    return data, audit


def add_composite_index(data: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise three factors and calculate their equal-weight mean."""
    result = data.copy()
    for source_column, normalized_column in NORMALIZED_FIELDS.items():
        minimum = float(result[source_column].min())
        maximum = float(result[source_column].max())
        span = maximum - minimum
        result[normalized_column] = (
            (result[source_column] - minimum) / span if span else 0.5
        )
    result["Composite_Happiness_Index"] = result[
        list(NORMALIZED_FIELDS.values())
    ].mean(axis=1)
    return result


def select_top_three(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.nlargest(3, "Composite_Happiness_Index")
        .sort_values("Composite_Happiness_Index", ascending=False)
        .reset_index(drop=True)
    )


def create_matplotlib_dashboard(
    data: pd.DataFrame,
    top_three: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplot_mosaic(
        [
            ["composite", "factors"],
            ["relationship", "correlations"],
        ],
        figsize=(16, 11),
        constrained_layout=True,
    )
    figure.suptitle(
        "World Happiness Dashboard — Top Three Composite Countries", fontsize=17
    )

    country_colors = {
        country: COLORS[index] for index, country in enumerate(top_three["Country"])
    }
    ordered = top_three.sort_values("Composite_Happiness_Index")

    ax = axes["composite"]
    bars = ax.barh(
        ordered["Country"],
        ordered["Composite_Happiness_Index"],
        color=[country_colors[country] for country in ordered["Country"]],
    )
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_xlim(0, 1)
    ax.set_title("Composite ranking")
    ax.set_xlabel("Equal-weight normalized index (0–1)")
    ax.grid(axis="x", alpha=0.25)

    ax = axes["factors"]
    x = np.arange(len(top_three))
    width = 0.24
    for index, normalized_column in enumerate(NORMALIZED_FIELDS.values()):
        ax.bar(
            x + (index - 1) * width,
            top_three[normalized_column],
            width,
            label=DISPLAY_NAMES[normalized_column],
        )
    ax.set_xticks(x, top_three["Country"])
    ax.set_ylim(0, 1.08)
    ax.set_title("Factors used in the composite index")
    ax.set_ylabel("Min-max normalized score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    ax.grid(axis="y", alpha=0.25)

    ax = axes["relationship"]
    top_names = set(top_three["Country"])
    other_countries = data.loc[~data["Country"].isin(top_names)]
    ax.scatter(
        other_countries["Freedom_to_Make_Choices"],
        other_countries["Happiness_Score"],
        color="#a5a5a5",
        s=50,
        alpha=0.75,
        label="Other countries",
    )
    for index, row in top_three.iterrows():
        ax.scatter(
            row["Freedom_to_Make_Choices"],
            row["Happiness_Score"],
            color=COLORS[index],
            s=105,
            edgecolor="black",
            linewidth=0.7,
            label=row["Country"],
            zorder=3,
        )
        ax.annotate(
            row["Country"],
            (row["Freedom_to_Make_Choices"], row["Happiness_Score"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    x_values = data["Freedom_to_Make_Choices"].to_numpy(dtype=float)
    y_values = data["Happiness_Score"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    x_line = np.linspace(x_values.min(), x_values.max(), 150)
    ax.plot(x_line, slope * x_line + intercept, color="#c00000", linewidth=1.8, label="Linear trend")
    correlation = float(data[["Freedom_to_Make_Choices", "Happiness_Score"]].corr().iloc[0, 1])
    ax.set_title(f"Freedom versus Happiness (Pearson r = {correlation:.3f})")
    ax.set_xlabel("Freedom to make choices (0–1)")
    ax.set_ylabel("Happiness score (0–10)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2)

    ax = axes["correlations"]
    correlation_matrix = data[NUMERIC_COLUMNS].corr()
    image = ax.imshow(correlation_matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(
        range(len(CORRELATION_LABELS)),
        CORRELATION_LABELS,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(CORRELATION_LABELS)), CORRELATION_LABELS)
    ax.set_title("Feature correlation matrix")
    for row in range(len(CORRELATION_LABELS)):
        for column in range(len(CORRELATION_LABELS)):
            value = correlation_matrix.iloc[row, column]
            text_color = "white" if abs(value) > 0.65 else "black"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    figure.colorbar(image, ax=ax, shrink=0.82, label="Pearson r")

    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_plotly_dashboard(
    data: pd.DataFrame,
    top_three: pd.DataFrame,
    path: Path,
) -> None:
    """Create an interactive dashboard using Plotly's Python graph objects."""
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Composite ranking",
            "Normalized aggregation factors",
            "Freedom versus Happiness",
            "Feature correlation matrix",
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )

    figure.add_trace(
        go.Bar(
            x=top_three["Country"],
            y=top_three["Composite_Happiness_Index"],
            marker_color=COLORS,
            text=top_three["Composite_Happiness_Index"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            name="Composite index",
            hovertemplate="%{x}<br>Composite index: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    factor_colors = ["#5b9bd5", "#a5a5a5", "#ffc000"]
    for normalized_column, color in zip(NORMALIZED_FIELDS.values(), factor_colors):
        figure.add_trace(
            go.Bar(
                x=top_three["Country"],
                y=top_three[normalized_column],
                name=DISPLAY_NAMES[normalized_column],
                marker_color=color,
                hovertemplate=(
                    f"%{{x}}<br>{DISPLAY_NAMES[normalized_column]}: %{{y:.3f}}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    top_names = set(top_three["Country"])
    other_countries = data.loc[~data["Country"].isin(top_names)]
    figure.add_trace(
        go.Scatter(
            x=other_countries["Freedom_to_Make_Choices"],
            y=other_countries["Happiness_Score"],
            mode="markers",
            marker={"color": "#a5a5a5", "size": 9, "opacity": 0.75},
            customdata=np.column_stack(
                [other_countries["Country"], other_countries["Healthy_Life_Expectancy"]]
            ),
            name="Other countries",
            hovertemplate=(
                "%{customdata[0]}<br>Freedom: %{x:.2f}<br>Happiness: %{y:.2f}"
                "<br>Healthy life expectancy: %{customdata[1]:.1f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    correlation_matrix = data[NUMERIC_COLUMNS].corr()
    figure.add_trace(
        go.Heatmap(
            z=correlation_matrix.to_numpy(),
            x=CORRELATION_LABELS,
            y=CORRELATION_LABELS,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=correlation_matrix.to_numpy(),
            texttemplate="%{text:.2f}",
            colorbar={"title": "Pearson r", "len": 0.42, "y": 0.22},
            hovertemplate="%{y} vs %{x}<br>Pearson r: %{z:.3f}<extra></extra>",
            name="Feature correlations",
        ),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=top_three["Freedom_to_Make_Choices"],
            y=top_three["Happiness_Score"],
            mode="markers+text",
            marker={"color": COLORS, "size": 15, "line": {"color": "black", "width": 1}},
            text=top_three["Country"],
            textposition="top center",
            customdata=top_three["Healthy_Life_Expectancy"],
            name="Top three composite countries",
            hovertemplate=(
                "%{text}<br>Freedom: %{x:.2f}<br>Happiness: %{y:.2f}"
                "<br>Healthy life expectancy: %{customdata:.1f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    x_values = data["Freedom_to_Make_Choices"].to_numpy(dtype=float)
    y_values = data["Happiness_Score"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    x_line = np.linspace(x_values.min(), x_values.max(), 150)
    figure.add_trace(
        go.Scatter(
            x=x_line,
            y=slope * x_line + intercept,
            mode="lines",
            line={"color": "#c00000", "width": 2},
            name="Linear trend",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    figure.update_yaxes(range=[0, 1.08], title_text="Composite index", row=1, col=1)
    figure.update_yaxes(range=[0, 1.08], title_text="Normalized score", row=1, col=2)
    figure.update_xaxes(title_text="Freedom to make choices (0–1)", row=2, col=1)
    figure.update_yaxes(title_text="Happiness score (0–10)", row=2, col=1)
    figure.update_xaxes(tickangle=-45, row=2, col=2)
    figure.update_layout(
        title={
            "text": "Interactive World Happiness Dashboard — Top Three Composite Countries",
            "x": 0.5,
        },
        barmode="group",
        height=850,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.12, "x": 0},
        margin={"l": 70, "r": 40, "t": 100, "b": 110},
    )
    figure.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
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

    clean_data, audit = load_clean_dataset(input_path)
    dashboard_data = add_composite_index(clean_data)
    top_three = select_top_three(dashboard_data)
    average_freedom = float(dashboard_data["Freedom_to_Make_Choices"].mean())

    clean_data.to_csv(
        output_dir / "cleaned_world_happiness.csv", index=False, float_format="%.6f"
    )
    dashboard_data.to_csv(
        output_dir / "dashboard_dataset.csv", index=False, float_format="%.6f"
    )
    top_three.to_csv(
        output_dir / "top_three_happiest.csv", index=False, float_format="%.6f"
    )
    correlation_matrix = dashboard_data[NUMERIC_COLUMNS].corr()
    correlation_matrix.index.name = "Feature"
    correlation_matrix.to_csv(
        output_dir / "feature_correlation_matrix.csv", float_format="%.6f"
    )
    create_matplotlib_dashboard(
        dashboard_data,
        top_three,
        output_dir / "matplotlib_happiness_dashboard.png",
    )
    create_plotly_dashboard(
        dashboard_data,
        top_three,
        output_dir / "plotly_happiness_dashboard.html",
    )
    upper_correlations = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    ).stack()
    strongest_pair = upper_correlations.abs().idxmax()
    summary = {
        "source_file": input_path.name,
        "cleaning_audit": audit,
        "aggregation": {
            "method": "Equal-weight mean of min-max normalized Happiness Score, Healthy Life Expectancy, and Freedom to Make Choices.",
            "weights": {field: 1 / 3 for field in COMPOSITE_FIELDS},
        },
        "dataset_average_freedom": average_freedom,
        "freedom_happiness_pearson_r": float(
            dashboard_data[["Freedom_to_Make_Choices", "Happiness_Score"]]
            .corr()
            .iloc[0, 1]
        ),
        "strongest_absolute_feature_correlation": {
            "features": list(strongest_pair),
            "pearson_r": float(upper_correlations.loc[strongest_pair]),
        },
        "top_three_composite_countries": [
            {
                "rank": index + 1,
                "country": row["Country"],
                "composite_index": row["Composite_Happiness_Index"],
                "happiness_score": row["Happiness_Score"],
                "healthy_life_expectancy": row["Healthy_Life_Expectancy"],
                "freedom_score": row["Freedom_to_Make_Choices"],
            }
            for index, (_, row) in enumerate(top_three.iterrows())
        ],
    }
    with (output_dir / "dashboard_summary.json").open("w", encoding="utf-8") as destination:
        json.dump(json_ready(summary), destination, indent=2, ensure_ascii=False)
        destination.write("\n")

    print("Top three countries by the composite happiness index:")
    for item in summary["top_three_composite_countries"]:
        print(
            f"{item['rank']}. {item['country']}: composite={item['composite_index']:.3f}, "
            f"happiness={item['happiness_score']:.2f}, "
            f"healthy_life_expectancy={item['healthy_life_expectancy']:.1f}, "
            f"freedom={item['freedom_score']:.2f}"
        )
    print(f"Dashboard outputs written to {output_dir}")


if __name__ == "__main__":
    main()
