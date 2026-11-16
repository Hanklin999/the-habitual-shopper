"""
The Habitual Shopper
Part II - Behavioral Validation and Storytelling Figures

Inputs:
    data/processed/basket_pairs.csv
    data/processed/joined_transactions.csv

Outputs:
    outputs/tables/
        fixed_window_user_profiles.csv
        segment_summary.csv
        later_window_comparison.csv
        future_segment_summary.csv
        segment_transition.csv
        basket_size_robustness.csv
        validation_selection_bias.csv
        selection_segment_composition.csv
        shopper_42_orders.csv
        story_metrics.csv
        claim_check.csv

    outputs/figures/
        sketch_01_similarity.png
        sketch_02_categories.png
        sketch_03_discovery.png
        sketch_03_basket_example.png
        sketch_04_persistence.png

Raw Instacart files are prepared by analysis/01_prepare_data.py.
"""

# %% 0. SETUP

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "basket_pairs.csv"
JOINED_FILE = PROCESSED_DIR / "joined_transactions.csv"

for input_file in [INPUT_FILE, JOINED_FILE]:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Cannot find {input_file}. "
            "Run analysis/01_prepare_data.py first."
        )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "user_id",
    "previous_order_id",
    "order_id",
    "order_number",
    "product_jaccard",
    "aisle_jaccard",
    "department_jaccard",
    "novelty_share",
    "basket_size",
    "previous_basket_size",
    "days_since_prior_order",
]

missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing basket-pair columns: {missing}")

duplicate_pairs = df.duplicated(subset=["user_id", "order_number"])

if duplicate_pairs.any():
    raise ValueError(
        f"Found {duplicate_pairs.sum()} duplicated user/order pairs."
    )

print("=" * 65)
print("THE HABITUAL SHOPPER - STORY ANALYSIS")
print("=" * 65)
print(f"Loaded {len(df):,} order pairs")
print(f"Users: {df['user_id'].nunique():,}")


# %% 1. FIXED ORDER WINDOW (ORDERS 5-10)

print("\n[1] Fixed Order Window")

window = df[df["order_number"].between(5, 10)].copy()

# Every included user must have all six order numbers: 5, 6, 7, 8, 9, 10.
valid_counts = window.groupby("user_id")["order_number"].nunique()
valid_users = valid_counts[valid_counts == 6].index

window = window[window["user_id"].isin(valid_users)].copy()

profiles = (
    window.groupby("user_id")
    .agg(
        avg_jaccard=("product_jaccard", "mean"),
        avg_novelty=("novelty_share", "mean"),
        avg_aisle_jaccard=("aisle_jaccard", "mean"),
        avg_department_jaccard=("department_jaccard", "mean"),
        avg_basket_size=("basket_size", "mean"),
        avg_interval=("days_since_prior_order", "mean"),
    )
    .reset_index()
)

if profiles.empty:
    raise ValueError("No users have complete order histories for orders 5-10.")

mean_product_overlap = profiles["avg_jaccard"].mean()
mean_novelty = profiles["avg_novelty"].mean()
mean_aisle_overlap = profiles["avg_aisle_jaccard"].mean()
mean_department_overlap = profiles["avg_department_jaccard"].mean()

correlation = profiles[["avg_jaccard", "avg_novelty"]].corr().iloc[0, 1]

print("Eligible users:", len(profiles))
print("Mean Product Jaccard:", round(mean_product_overlap, 3))
print("Mean Aisle Jaccard:", round(mean_aisle_overlap, 3))
print("Mean Department Jaccard:", round(mean_department_overlap, 3))
print("Mean Novelty:", round(mean_novelty, 3))
print("Jaccard/Novelty Correlation:", round(correlation, 3))


# %% 2. EXPLORATORY BEHAVIORAL SEGMENTATION

print("\n[2] Exploratory Segmentation")

# Thresholds are calculated only from the historical window, orders 5-10.
j_cut = profiles["avg_jaccard"].median()
n_cut = profiles["avg_novelty"].median()


def assign_segment(jaccard, novelty, jaccard_cutoff, novelty_cutoff):
    """Assign a descriptive segment using fixed historical thresholds."""
    high_jaccard = jaccard >= jaccard_cutoff
    high_novelty = novelty >= novelty_cutoff

    if not high_jaccard and not high_novelty:
        return "Rotational"
    if not high_jaccard and high_novelty:
        return "New-to-user"
    if high_jaccard and not high_novelty:
        return "Routine"
    return "Mixed"


profiles["segment"] = [
    assign_segment(jaccard, novelty, j_cut, n_cut)
    for jaccard, novelty in zip(
        profiles["avg_jaccard"],
        profiles["avg_novelty"],
    )
]

segment_order = [
    "New-to-user",
    "Rotational",
    "Routine",
    "Mixed",
]

segment_summary = (
    profiles.groupby("segment")
    .agg(
        users=("user_id", "size"),
        product_similarity=("avg_jaccard", "mean"),
        aisle_similarity=("avg_aisle_jaccard", "mean"),
        department_similarity=("avg_department_jaccard", "mean"),
        novelty=("avg_novelty", "mean"),
        basket_size=("avg_basket_size", "mean"),
        interval=("avg_interval", "mean"),
    )
    .reindex(segment_order)
)

segment_summary["share"] = segment_summary["users"] / len(profiles)

print(f"Jaccard cutoff: {j_cut:.3f}")
print(f"Novelty cutoff: {n_cut:.3f}")
print(segment_summary.round(3).to_string())

profiles.to_csv(
    TABLE_DIR / "fixed_window_user_profiles.csv",
    index=False,
)

segment_summary.to_csv(
    TABLE_DIR / "segment_summary.csv",
    index_label="segment",
)


# %% 3. LATER-WINDOW COMPARISON (ORDERS 11-15)

print("\n[3] Later-window Comparison")

future = df[df["order_number"].between(11, 15)].copy()

future_counts = future.groupby("user_id")["order_number"].nunique()
future_users = future_counts[future_counts == 5].index

future = future[future["user_id"].isin(future_users)].copy()

future_profiles = (
    future.groupby("user_id")
    .agg(
        future_jaccard=("product_jaccard", "mean"),
        future_novelty=("novelty_share", "mean"),
        future_aisle=("aisle_jaccard", "mean"),
        future_department=("department_jaccard", "mean"),
        future_basket_size=("basket_size", "mean"),
        future_interval=("days_since_prior_order", "mean"),
    )
    .reset_index()
)

# Retain historical segment assignments when joining the later window.
validation = profiles.merge(
    future_profiles,
    on="user_id",
    how="inner",
    validate="one_to_one",
)

if validation.empty:
    raise ValueError("No users have complete order histories for orders 11-15.")

print("Historical users:", len(profiles))
print("Later-window users:", len(validation))

future_summary = (
    validation.groupby("segment")
    .agg(
        users=("user_id", "size"),
        historical_jaccard=("avg_jaccard", "mean"),
        future_jaccard=("future_jaccard", "mean"),
        historical_novelty=("avg_novelty", "mean"),
        future_novelty=("future_novelty", "mean"),
    )
    .reindex(segment_order)
)

print("\nHistorical vs. Later Window:")
print(future_summary.round(3).to_string())

# Apply the original historical thresholds, not later-period medians.
validation["future_segment"] = [
    assign_segment(jaccard, novelty, j_cut, n_cut)
    for jaccard, novelty in zip(
        validation["future_jaccard"],
        validation["future_novelty"],
    )
]

transition_counts = pd.crosstab(
    validation["segment"],
    validation["future_segment"],
).reindex(
    index=segment_order,
    columns=segment_order,
    fill_value=0,
)

transition_denominator = transition_counts.sum(axis=1).replace(0, np.nan)
transition = transition_counts.div(transition_denominator, axis=0)

print("\nLater-window Segment Transition (%):")
print((transition * 100).round(1).to_string())

validation.to_csv(
    TABLE_DIR / "later_window_comparison.csv",
    index=False,
)

future_summary.to_csv(
    TABLE_DIR / "future_segment_summary.csv",
    index_label="segment",
)

transition.to_csv(
    TABLE_DIR / "segment_transition.csv",
    index_label="historical_segment",
)


# %% 4. AVERAGE BASKET-SIZE SENSITIVITY

print("\n[4] Average Basket-size Sensitivity")

robustness_rows = []

for minimum_average_size in [1, 3, 5, 8, 10]:
    subset = profiles[
        profiles["avg_basket_size"] >= minimum_average_size
    ]

    rotational_share = subset["segment"].eq("Rotational").mean()
    subset_correlation = subset[
        ["avg_jaccard", "avg_novelty"]
    ].corr().iloc[0, 1]

    robustness_rows.append(
        {
            "minimum_average_basket_size": minimum_average_size,
            "users": len(subset),
            "rotational_share": rotational_share,
            "correlation": subset_correlation,
        }
    )

robustness = pd.DataFrame(robustness_rows)

print(robustness.round(3).to_string(index=False))

robustness.to_csv(
    TABLE_DIR / "basket_size_robustness.csv",
    index=False,
)


# %% 5. LATER-WINDOW SELECTION CHECK

print("\n[5] Later-window Selection Check")

comparison = profiles.copy()
comparison["has_future_orders"] = comparison["user_id"].isin(
    validation["user_id"]
)

selection_summary = (
    comparison.groupby("has_future_orders")
    .agg(
        users=("user_id", "size"),
        avg_jaccard=("avg_jaccard", "mean"),
        avg_novelty=("avg_novelty", "mean"),
        avg_basket_size=("avg_basket_size", "mean"),
        avg_interval=("avg_interval", "mean"),
    )
)

print(selection_summary.round(3).to_string())

segment_comparison = pd.crosstab(
    comparison["segment"],
    comparison["has_future_orders"],
    normalize="columns",
).reindex(
    index=segment_order,
    columns=[False, True],
    fill_value=0,
)

print("\nSegment distribution (%):")
print((segment_comparison * 100).round(1).to_string())

selection_summary.to_csv(
    TABLE_DIR / "validation_selection_bias.csv",
    index_label="has_future_orders",
)

segment_comparison.to_csv(
    TABLE_DIR / "selection_segment_composition.csv",
    index_label="segment",
)


# %% 6. CHART STYLE AND EXPORT HELPER

print("\n[6] Creating Story Figures")

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

ORANGE = "#D98B35"
BLUE = "#477FA3"
GREEN = "#398477"
GRAY = "#B8BCC2"
DARK = "#25313D"


def save_figure(fig, filename):
    """Save a story figure and close its Matplotlib object."""
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# %% 7. FIGURE 1 - PRODUCT-LEVEL BASKET OVERLAP

fig, ax = plt.subplots(figsize=(10, 6))

similarity_values = profiles["avg_jaccard"].dropna()

ax.hist(
    similarity_values,
    bins=np.linspace(0, 1, 26),
    color=BLUE,
    edgecolor="white",
    alpha=0.85,
)

ax.axvline(
    mean_product_overlap,
    color=ORANGE,
    linestyle="--",
    linewidth=2.2,
    label=f"Mean = {mean_product_overlap:.3f}",
)

ax.set_title(
    "Consecutive Baskets Share Relatively Few Products",
    fontweight="bold",
    loc="left",
    pad=18,
)
ax.set_xlabel("Average product-level basket overlap")
ax.set_ylabel("Number of shoppers")
ax.set_xlim(0, 1)
ax.legend(frameon=False)

ax.text(
    0.98,
    0.97,
    f"Orders 5-10 | n = {len(profiles)} shoppers",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color=DARK,
)

fig.tight_layout()
save_figure(fig, "sketch_01_similarity.png")


# %% 8. FIGURE 2 - PRODUCT VS. CATEGORY OVERLAP

similarity_means = pd.Series(
    {
        "Product": mean_product_overlap,
        "Aisle": mean_aisle_overlap,
        "Department": mean_department_overlap,
    }
)

fig, ax = plt.subplots(figsize=(10, 5.5))

levels = ["Product", "Aisle", "Department"]
level_values = similarity_means[levels].values

bars = ax.barh(
    levels,
    level_values,
    color=[ORANGE, BLUE, GREEN],
    height=0.55,
)

ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Average Jaccard overlap")
ax.set_title(
    "Baskets Look More Familiar at Broader Category Levels",
    fontweight="bold",
    loc="left",
    pad=18,
)

for bar, value in zip(bars, level_values):
    ax.text(
        value + 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1%}",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

ax.text(
    0.01,
    -0.19,
    "Category-level overlap is mechanically affected by classification granularity.",
    transform=ax.transAxes,
    fontsize=9,
    color="gray",
)

fig.tight_layout()
save_figure(fig, "sketch_02_categories.png")


# %% 9. FIGURE 3 - ROTATIONAL SHOPPERS

fig, ax = plt.subplots(figsize=(10, 6.5))

background = profiles[profiles["segment"] != "Rotational"]
rotational = profiles[profiles["segment"] == "Rotational"]

ax.scatter(
    background["avg_jaccard"],
    background["avg_novelty"],
    color=GRAY,
    alpha=0.40,
    s=28,
    label="Other shoppers",
)

ax.scatter(
    rotational["avg_jaccard"],
    rotational["avg_novelty"],
    color=ORANGE,
    alpha=0.9,
    s=55,
    label=f"Rotational (n={len(rotational)})",
)

ax.axvline(j_cut, linestyle="--", color="gray", alpha=0.5)
ax.axhline(n_cut, linestyle="--", color="gray", alpha=0.5)

ax.annotate(
    "Different baskets,\nbut mostly familiar products",
    xy=(0.10, 0.29),
    xytext=(0.37, 0.72),
    arrowprops={
        "arrowstyle": "->",
        "color": ORANGE,
        "lw": 1.8,
    },
    color="#A6611E",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Basket overlap (Product Jaccard)")
ax.set_ylabel("New-to-recorded-history product share")
ax.set_title(
    "A Different Basket Can Still Contain Familiar Products",
    fontweight="bold",
    loc="left",
    pad=18,
)
ax.legend(frameon=False, loc="upper right")

ax.text(
    0.02,
    -0.15,
    f"Orders 5-10 | n = {len(profiles)} shoppers | "
    "Exploratory median-based patterns",
    transform=ax.transAxes,
    fontsize=9,
    color="gray",
)

fig.tight_layout()
save_figure(fig, "sketch_03_discovery.png")


# %% 10. FIGURE 4 - DESCRIPTIVE PATTERN PERSISTENCE

fig, ax = plt.subplots(figsize=(10, 5.5))

groups = ["Routine", "Rotational"]
same = [
    transition.loc["Routine", "Routine"],
    transition.loc["Rotational", "Rotational"],
]
different = [1 - value for value in same]

y_positions = np.arange(len(groups))

ax.barh(
    y_positions,
    same,
    color=GREEN,
    height=0.42,
    label="Same classification",
)
ax.barh(
    y_positions,
    different,
    left=same,
    color="#D9DEE2",
    height=0.42,
    label="Different classification",
)

ax.set_yticks(y_positions)
ax.set_yticklabels(groups, fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0, 1.13)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xticklabels(
    [f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)]
)

for index, group in enumerate(groups):
    percentage = same[index] * 100
    group_size = int(future_summary.loc[group, "users"])

    ax.text(
        same[index] / 2,
        index,
        f"{percentage:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        1.02,
        index,
        f"n = {group_size}",
        va="center",
        fontsize=11,
        color=DARK,
    )

ax.set_xlabel("Share of shoppers")
ax.set_title(
    "Some Shopping Patterns Continue Across Later Orders",
    fontsize=17,
    fontweight="bold",
    loc="left",
    pad=20,
)
ax.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.27),
    ncol=2,
)

included_interval = selection_summary.loc[True, "avg_interval"]
excluded_interval = selection_summary.loc[False, "avg_interval"]

fig.text(
    0.12,
    -0.04,
    "Orders 5-10 vs. 11-15 | Complete later histories only.\n"
    f"Average order interval: included {included_interval:.1f} days; "
    f"excluded {excluded_interval:.1f} days.",
    fontsize=9,
    color="gray",
)

fig.tight_layout(rect=[0, 0.10, 1, 1])
save_figure(fig, "sketch_04_persistence.png")


# %% 11. FIXED SHOPPER 42 EXAMPLE

print("\n[7] Building Shopper 42 Example")

joined = pd.read_csv(JOINED_FILE)

joined_required_columns = [
    "user_id",
    "order_id",
    "order_number",
    "product_id",
    "product_name",
    "add_to_cart_order",
    "reordered",
]

joined_missing = [
    column for column in joined_required_columns if column not in joined.columns
]

if joined_missing:
    raise ValueError(f"Missing joined-data columns: {joined_missing}")

rotational_users = profiles.loc[
    profiles["segment"] == "Rotational",
    "user_id",
]

rotational_pairs = window[
    window["user_id"].isin(rotational_users)
].copy()

candidates = rotational_pairs[
    (rotational_pairs["basket_size"] >= 5)
    & (rotational_pairs["previous_basket_size"] >= 5)
    & (rotational_pairs["product_jaccard"] <= 0.15)
    & (rotational_pairs["novelty_share"] <= 0.20)
].copy()

candidates = candidates.sort_values(
    by=["novelty_share", "product_jaccard", "basket_size"],
    ascending=[True, True, False],
)

candidate_columns = [
    "user_id",
    "previous_order_id",
    "order_id",
    "order_number",
    "product_jaccard",
    "novelty_share",
    "basket_size",
    "previous_basket_size",
]

print("Suitable candidates:", len(candidates))
print(candidates[candidate_columns].head(20).to_string(index=False))

shopper_42_candidate = candidates[
    (candidates["user_id"] == 42)
    & (candidates["order_number"] == 10)
]

if len(shopper_42_candidate) != 1:
    raise ValueError(
        "Expected exactly one Shopper 42 order-10 example, "
        f"but found {len(shopper_42_candidate)}."
    )

selected = shopper_42_candidate.iloc[0]

example_user = int(selected["user_id"])
example_previous_order = int(selected["previous_order_id"])
example_current_order = int(selected["order_id"])
example_current_order_number = int(selected["order_number"])
example_previous_order_number = example_current_order_number - 1

print("Selected user:", example_user)
print("Previous order:", example_previous_order)
print("Current order:", example_current_order)
print("Product Jaccard:", selected["product_jaccard"])
print("New-to-history share:", selected["novelty_share"])

example_rows = joined[
    (joined["user_id"] == example_user)
    & joined["order_id"].isin(
        [example_previous_order, example_current_order]
    )
].copy()

example_rows = example_rows.sort_values(
    ["order_number", "add_to_cart_order"]
)

if example_rows.empty:
    raise ValueError("No joined transaction rows found for Shopper 42.")

example_rows.to_csv(
    TABLE_DIR / "shopper_42_orders.csv",
    index=False,
)

previous_products = example_rows[
    example_rows["order_id"] == example_previous_order
].copy()

current_products = example_rows[
    example_rows["order_id"] == example_current_order
].copy()

if previous_products.empty or current_products.empty:
    raise ValueError("One of the Shopper 42 example orders is empty.")

common_product_ids = set(previous_products["product_id"]) & set(
    current_products["product_id"]
)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

example_panels = [
    (
        axes[0],
        previous_products,
        f"Order {example_previous_order_number} | "
        f"{len(previous_products)} products",
    ),
    (
        axes[1],
        current_products,
        f"Order {example_current_order_number} | "
        f"{len(current_products)} products",
    ),
]

for axis, products, title in example_panels:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, len(products) + 1)
    axis.axis("off")
    axis.set_title(title, fontsize=15, fontweight="bold")

    for index, row in enumerate(products.itertuples(), start=1):
        highlight = row.product_id in common_product_ids

        axis.text(
            0.04,
            len(products) + 0.4 - index,
            row.product_name,
            fontsize=10,
            color=GREEN if highlight else DARK,
            fontweight="bold" if highlight else "normal",
        )

fig.suptitle(
    "Different Baskets, Familiar Products",
    fontsize=19,
    fontweight="bold",
)

fig.text(
    0.5,
    0.025,
    f"Shopper {example_user} | "
    f"Jaccard = {selected['product_jaccard']:.1%} | "
    f"New-to-history share = {selected['novelty_share']:.1%}",
    ha="center",
    fontsize=11,
)

fig.tight_layout(rect=[0, 0.06, 1, 0.93])
save_figure(fig, "sketch_03_basket_example.png")


# %% 12. STORY METRICS AND CLAIM CHECKS

routine_retention = transition.loc["Routine", "Routine"]
rotational_retention = transition.loc["Rotational", "Rotational"]
rotational_share = len(rotational) / len(profiles)

story_metrics = pd.Series(
    {
        "main_users": len(profiles),
        "mean_product_overlap": mean_product_overlap,
        "mean_aisle_overlap": mean_aisle_overlap,
        "mean_department_overlap": mean_department_overlap,
        "mean_novelty": mean_novelty,
        "jaccard_cutoff": j_cut,
        "novelty_cutoff": n_cut,
        "jaccard_novelty_correlation": correlation,
        "rotational_users": len(rotational),
        "rotational_share": rotational_share,
        "later_window_users": len(validation),
        "routine_retention": routine_retention,
        "rotational_retention": rotational_retention,
        "shopper_42_jaccard": selected["product_jaccard"],
        "shopper_42_novelty": selected["novelty_share"],
    },
    name="value",
)

story_metrics.to_csv(
    TABLE_DIR / "story_metrics.csv",
    header=True,
)

expected_claims = {
    "main_users": (484, 0),
    "mean_product_overlap": (0.194, 0.001),
    "mean_aisle_overlap": (0.296, 0.001),
    "mean_department_overlap": (0.461, 0.001),
    "rotational_users": (50, 0),
    "later_window_users": (323, 0),
    "routine_retention": (0.803, 0.001),
    "rotational_retention": (0.537, 0.001),
    "shopper_42_jaccard": (1 / 12, 0.000001),
    "shopper_42_novelty": (0.0, 0.000001),
}

claim_rows = []

for metric, (expected, tolerance) in expected_claims.items():
    actual = float(story_metrics.loc[metric])
    passed = abs(actual - expected) <= tolerance

    claim_rows.append(
        {
            "metric": metric,
            "actual": actual,
            "expected": expected,
            "tolerance": tolerance,
            "passed": passed,
        }
    )

claim_check = pd.DataFrame(claim_rows)
claim_check.to_csv(
    TABLE_DIR / "claim_check.csv",
    index=False,
)

failed_claims = claim_check[~claim_check["passed"]]

if not failed_claims.empty:
    print("\nWARNING: Some published claims no longer match:")
    print(failed_claims.to_string(index=False))
else:
    print("\nAll published claim checks passed.")


# %% 13. FINAL SUMMARY

print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)
print(f"Historical fixed-window users: {len(profiles)}")
print(f"Later-window users: {len(validation)}")
print(f"Historical Jaccard/Novelty r: {correlation:.3f}")
print(f"Rotational share: {rotational_share:.1%}")
print(f"\nCSV outputs: {TABLE_DIR}")
print(f"Story figures: {FIGURE_DIR}")

print("\nIMPORTANT LIMITATIONS:")
print("- First 1,000 prior-order users by user ID, not a random sample.")
print("- Segment definitions use historical-window sample medians.")
print("- Novelty depends on the available recorded purchase history.")
print("- Later-window results require complete orders 11-15.")
print("- Included later-window shoppers purchase more frequently.")
print("- Category overlap depends on classification granularity.")
print("- No causal evidence about recommendation effectiveness.")
