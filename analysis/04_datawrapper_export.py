"""Export Datawrapper-ready datasets for The Habitual Shopper story.

Run from repository root:
    py analysis/04_datawrapper_export.py

Inputs (created by analysis/03_story_analysis.py):
    outputs/tables/03_shopper_repeat_profiles.csv
    outputs/tables/03_shopper_basket_composition.csv

Outputs:
    outputs/tables/04_dw_repeat_5pct.csv            (20 bins, 3 series)
    outputs/tables/04_dw_repeat_product.csv         (Product only)
    outputs/tables/04_dw_repeat_aisle.csv           (Aisle only)
    outputs/tables/04_dw_repeat_department.csv      (Department only)
    outputs/tables/04_dw_basket_composition.csv     (1 row, 3 series)
    outputs/tables/04_dw_medians.csv                (reference labels)

The repeat percentages are shopper-level averages across Orders 5–10.
The composition percentages are the equally weighted mean across shoppers,
not a pooled item-weighted composition.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
REPEAT_INPUT = TABLE_DIR / "03_shopper_repeat_profiles.csv"
COMPOSITION_INPUT = TABLE_DIR / "03_shopper_basket_composition.csv"
LEVELS = ["Product", "Aisle", "Department"]
STATUS = [
    "Same as last order",
    "Bought before, not last time",
    "New to recorded history",
]


def load_profiles(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run analysis/03_story_analysis.py first")
    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if df["user_id"].isna().any() or df["user_id"].duplicated().any():
        raise ValueError(f"{path.name}: user_id must be unique and non-null")
    for col in required_columns:
        if col == "user_id":
            continue
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any() or not df[col].between(0, 100).all():
            raise ValueError(f"{path.name}: {col} must be between 0 and 100, with no missing values")
    return df


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    repeat = load_profiles(REPEAT_INPUT, ["user_id", *LEVELS])
    composition = load_profiles(COMPOSITION_INPUT, ["user_id", *STATUS])

    if set(repeat.user_id) != set(composition.user_id):
        raise ValueError("The two input files contain different shoppers")
    n = len(repeat)
    if n == 0:
        raise ValueError("No shoppers found")
    if not np.allclose(composition[STATUS].sum(axis=1), 100, atol=1e-6):
        raise ValueError("Each shopper's composition must sum to 100%")
    if not ((repeat.Product <= repeat.Aisle + 1e-8) &
            (repeat.Aisle <= repeat.Department + 1e-8)).all():
        raise ValueError("Expected Product <= Aisle <= Department for each shopper")

    # Datawrapper chart 1: 20 five-percentage-point bins.
    # np.histogram uses [0,5), [5,10), ... [95,100] (100 included in the last bin).
    edges = np.arange(0, 105, 5, dtype=int)
    bin_labels = [f"{left}–{right}%" for left, right in zip(edges[:-1], edges[1:])]
    histogram = pd.DataFrame({"Repeat share": bin_labels})
    for level in LEVELS:
        histogram[level] = np.histogram(repeat[level], bins=edges)[0].astype(int)
        assert int(histogram[level].sum()) == n
    histogram.to_csv(TABLE_DIR / "04_dw_repeat_5pct.csv", index=False, encoding="utf-8-sig")

    # One file per chart is easiest for three separate Datawrapper column charts.
    for level in LEVELS:
        histogram[["Repeat share", level]].to_csv(
            TABLE_DIR / f"04_dw_repeat_{level.lower()}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Datawrapper chart 2: single 100%-stacked bar; percentages already total 100.
    # Keep full precision in the CSV to avoid rounding 30.5+30.6+38.9 accidentally.
    shares = composition[STATUS].mean()
    if not np.isclose(shares.sum(), 100, atol=1e-6):
        raise AssertionError("Average basket composition does not total 100%")
    composition_wide = pd.DataFrame([{
        "Group": f"Orders 5–10 ({n} shoppers)",
        **{status: float(shares[status]) for status in STATUS},
    }])
    composition_wide.to_csv(
        TABLE_DIR / "04_dw_basket_composition.csv", index=False, encoding="utf-8-sig"
    )

    # Reference medians for manual Datawrapper annotations.
    medians = pd.DataFrame({
        "Level": LEVELS,
        "Median repeat share (%)": [repeat[level].median() for level in LEVELS],
        "Mean repeat share (%)": [repeat[level].mean() for level in LEVELS],
        "Shoppers": [n] * len(LEVELS),
    })
    medians.to_csv(TABLE_DIR / "04_dw_medians.csv", index=False, encoding="utf-8-sig")

    print(f"Shoppers: {n}")
    print("\nHistogram counts (first five bins):")
    print(histogram.head().to_string(index=False))
    print("\nHistogram totals:", {col: int(histogram[col].sum()) for col in LEVELS})
    print("\nBasket composition (%):")
    for status in STATUS:
        print(f"  {status}: {shares[status]:.1f}%")
    print("\nMedians (%):")
    print(medians[["Level", "Median repeat share (%)"]].round(1).to_string(index=False))
    print(f"\nDatawrapper CSV files saved to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
