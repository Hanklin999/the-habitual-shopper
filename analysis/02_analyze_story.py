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
        conditional_no_new_summary.csv
        conditional_threshold_sensitivity.csv
        conditional_bootstrap_summary.csv
        future_conditional_no_new_summary.csv

    outputs/figures/
        sketch_01_similarity.png
        sketch_02_conditional_no_new.png
        sketch_02_categories.png
        sketch_03_discovery.png
        sketch_03_basket_example.png
        sketch_04_persistence.png

Raw Instacart files are prepared by analysis/01_prepare_data.py.
"""

# %% 0. SETUP

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Cannot find: {INPUT_FILE}")

# Read prepared order-pair data
df = pd.read_csv(INPUT_FILE)

required_columns = [
    "user_id",
    "order_id",
    "order_number",
    "product_jaccard",
    "aisle_jaccard",
    "department_jaccard",
    "novelty_share",
    "basket_size",
    "previous_basket_size",
    "days_since_prior_order",
    "previous_order_id",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

duplicate_pairs = df.duplicated(
    subset=["user_id", "order_number"]
)

if duplicate_pairs.any():
    raise ValueError(
        f"Found {duplicate_pairs.sum()} duplicated user/order pairs."
    )

for column in [
    "product_jaccard",
    "aisle_jaccard",
    "department_jaccard",
    "novelty_share",
]:
    if not df[column].between(0, 1).all():
        raise ValueError(
            f"{column} contains values outside 0-1."
        )
        
print("=" * 65)
print("THE HABITUAL SHOPPER - STORY ANALYSIS")
print("=" * 65)
print(f"Loaded {len(df):,} order pairs")
print(f"Users: {df['user_id'].nunique():,}")


# %% 1. FIXED ORDER WINDOW (ORDERS 5-10)

print("\n[1] Fixed Order Window")

window = df[
    df["order_number"].between(5, 10)
].copy()

# Each user must have all six order numbers.
valid_counts = (
    window.groupby("user_id")["order_number"]
    .nunique()
)

valid_users = valid_counts[valid_counts == 6].index

window = window[
    window["user_id"].isin(valid_users)
].copy()

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

print("Eligible users:", len(profiles))
print("Mean Jaccard:", round(profiles["avg_jaccard"].mean(), 3))
print("Mean Novelty:", round(profiles["avg_novelty"].mean(), 3))

correlation = profiles[
    ["avg_jaccard", "avg_novelty"]
].corr().iloc[0, 1]

print("Correlation:", round(correlation, 3))


# %% 2. BEHAVIORAL SEGMENTATION

print("\n[2] Exploratory Segmentation")

# Median thresholds based only on historical orders 5-10
j_cut = profiles["avg_jaccard"].median()
n_cut = profiles["avg_novelty"].median()

# %% 2A. ORDER-LEVEL CONDITIONAL NO-NEW-PRODUCT RATE

print("\n[2A] Conditional No-New-Product Rate")

# j_cut 原本來自 shopper-level average Jaccard 的 median。
# 這裡只是為了可比較性，探索性地套用到 order level。
order_level_cutoff = j_cut

order_level = window.copy()

order_level["no_new_products"] = np.isclose(
    order_level["novelty_share"],
    0.0,
    atol=1e-12,
)

order_level["similarity_group"] = np.where(
    order_level["product_jaccard"] < order_level_cutoff,
    "Low similarity",
    "High similarity",
)

conditional_summary = (
    order_level.groupby("similarity_group")
    .agg(
        order_pairs=("order_id", "size"),
        shoppers=("user_id", "nunique"),
        no_new_pairs=("no_new_products", "sum"),
        no_new_product_rate=("no_new_products", "mean"),
        average_novelty=("novelty_share", "mean"),
        median_novelty=("novelty_share", "median"),
        average_jaccard=("product_jaccard", "mean"),
    )
    .reset_index()
)

no_new_shopper_counts = (
    order_level.loc[order_level["no_new_products"]]
    .groupby("similarity_group")["user_id"]
    .nunique()
)

conditional_summary.insert(
    4,
    "no_new_shoppers",
    conditional_summary["similarity_group"]
    .map(no_new_shopper_counts)
    .fillna(0)
    .astype(int),
)

print(
    conditional_summary.assign(
        no_new_product_rate=lambda x: x["no_new_product_rate"] * 100
    ).round(3).to_string(index=False)
)

conditional_summary.to_csv(
    TABLE_DIR / "conditional_no_new_summary.csv",
    index=False,
)

# %% 2B. ORDER-LEVEL THRESHOLD SENSITIVITY

thresholds = sorted(set([
    0.05,
    0.10,
    0.125,
    0.15,
    float(j_cut),
    0.175,
    0.20,
    0.25,
    0.30,
]))

sensitivity_rows = []

for cutoff in thresholds:
    low = order_level[
        order_level["product_jaccard"] < cutoff
    ]
    high = order_level[
        order_level["product_jaccard"] >= cutoff
    ]

    if low.empty or high.empty:
        continue

    sensitivity_rows.append({
        "jaccard_cutoff": cutoff,
        "low_order_pairs": len(low),
        "high_order_pairs": len(high),
        "low_shoppers": low["user_id"].nunique(),
        "high_shoppers": high["user_id"].nunique(),
        "low_no_new_rate": low["no_new_products"].mean(),
        "high_no_new_rate": high["no_new_products"].mean(),
        "rate_difference": (
            low["no_new_products"].mean()
            - high["no_new_products"].mean()
        ),
        "low_average_novelty": low["novelty_share"].mean(),
        "high_average_novelty": high["novelty_share"].mean(),
    })

threshold_sensitivity = pd.DataFrame(sensitivity_rows)

print(
    threshold_sensitivity.assign(
        low_no_new_rate=lambda x: x["low_no_new_rate"] * 100,
        high_no_new_rate=lambda x: x["high_no_new_rate"] * 100,
        rate_difference=lambda x: x["rate_difference"] * 100,
    ).round(3).to_string(index=False)
)

threshold_sensitivity.to_csv(
    TABLE_DIR / "conditional_threshold_sensitivity.csv",
    index=False,
)

# %% 2C. SHOPPER-CLUSTER BOOTSTRAP

def bootstrap_conditional_rates(
    data,
    cutoff,
    iterations=1000,  # 第一次先跑 1000
    seed=42,
):
    rng = np.random.default_rng(seed)

    clusters = {
        user_id: group
        for user_id, group in data.groupby("user_id")
    }

    users = np.array(list(clusters))
    results = []

    for _ in range(iterations):
        sampled_users = rng.choice(
            users,
            size=len(users),
            replace=True,
        )

        sample = pd.concat(
            [clusters[user_id] for user_id in sampled_users],
            ignore_index=True,
        )

        low = sample[
            sample["product_jaccard"] < cutoff
        ]

        high = sample[
            sample["product_jaccard"] >= cutoff
        ]

        if low.empty or high.empty:
            continue

        low_rate = low["no_new_products"].mean()
        high_rate = high["no_new_products"].mean()

        results.append({
            "low_rate": low_rate,
            "high_rate": high_rate,
            "rate_difference": low_rate - high_rate,
        })

    return pd.DataFrame(results)


bootstrap_results = bootstrap_conditional_rates(
    order_level,
    order_level_cutoff,
)

bootstrap_summary = (
    bootstrap_results.quantile([0.025, 0.5, 0.975])
    .rename_axis("quantile")
    .reset_index()
)

bootstrap_display = bootstrap_summary.copy()

for column in [
    "low_rate",
    "high_rate",
    "rate_difference",
]:
    bootstrap_display[column] *= 100

print(bootstrap_display.round(2).to_string(index=False))

bootstrap_summary.to_csv(
    TABLE_DIR / "conditional_bootstrap_summary.csv",
    index=False,
)


def assign_segment(jaccard, novelty):
    high_j = jaccard >= j_cut
    high_n = novelty >= n_cut

    if not high_j and not high_n:
        return "Rotational"
    elif not high_j and high_n:
        return "New-to-user"
    elif high_j and not high_n:
        return "Routine"
    else:
        return "Mixed"


profiles["segment"] = [
    assign_segment(j, n)
    for j, n in zip(
        profiles["avg_jaccard"],
        profiles["avg_novelty"]
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

segment_summary["share"] = (
    segment_summary["users"] / len(profiles)
)

print(f"Jaccard cutoff: {j_cut:.3f}")
print(f"Novelty cutoff: {n_cut:.3f}")
print(segment_summary.round(3).to_string())

profiles.to_csv(
    TABLE_DIR / "fixed_window_user_profiles.csv",
    index=False
)

segment_summary.to_csv(
    TABLE_DIR / "segment_summary.csv"
)


# %% 3. OUT-OF-SAMPLE VALIDATION (ORDERS 11-15)

print("\n[3] Out-of-sample Validation")

future = df[
    df["order_number"].between(11, 15)
].copy()

future_counts = (
    future.groupby("user_id")["order_number"]
    .nunique()
)

future_users = future_counts[future_counts == 5].index

future = future[
    future["user_id"].isin(future_users)
].copy()

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

# Keep historical segment assignments
validation = profiles.merge(
    future_profiles,
    on="user_id",
    how="inner",
    validate="one_to_one"
)

print("Historical users:", len(profiles))
print("Validation users:", len(validation))

# %% 3A. LATER-WINDOW CONDITIONAL CHECK

future_order_level = future[
    future["user_id"].isin(validation["user_id"])
].copy()

future_order_level["no_new_products"] = np.isclose(
    future_order_level["novelty_share"],
    0.0,
    atol=1e-12,
)

future_order_level["similarity_group"] = np.where(
    future_order_level["product_jaccard"] < order_level_cutoff,
    "Low similarity",
    "High similarity",
)

future_conditional_summary = (
    future_order_level.groupby("similarity_group")
    .agg(
        order_pairs=("order_id", "size"),
        shoppers=("user_id", "nunique"),
        no_new_pairs=("no_new_products", "sum"),
        no_new_product_rate=("no_new_products", "mean"),
        average_novelty=("novelty_share", "mean"),
        average_jaccard=("product_jaccard", "mean"),
    )
    .reset_index()
)

future_no_new_shopper_counts = (
    future_order_level.loc[future_order_level["no_new_products"]]
    .groupby("similarity_group")["user_id"]
    .nunique()
)

future_conditional_summary.insert(
    4,
    "no_new_shoppers",
    future_conditional_summary["similarity_group"]
    .map(future_no_new_shopper_counts)
    .fillna(0)
    .astype(int),
)

print("\nLater-window conditional comparison:")
print(
    future_conditional_summary.assign(
        no_new_product_rate=lambda x: x["no_new_product_rate"] * 100
    ).round(3).to_string(index=False)
)

future_conditional_summary.to_csv(
    TABLE_DIR / "future_conditional_no_new_summary.csv",
    index=False,
)

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

print("\nHistorical vs Future:")
print(future_summary.round(3).to_string())

# Apply original cutoffs, not new future-period medians.
validation["future_segment"] = [
    assign_segment(j, n)
    for j, n in zip(
        validation["future_jaccard"],
        validation["future_novelty"]
    )
]

transition_counts = pd.crosstab(
    validation["segment"],
    validation["future_segment"]
).reindex(
    index=segment_order,
    columns=segment_order,
    fill_value=0
)

transition = transition_counts.div(
    transition_counts.sum(axis=1),
    axis=0
)

print("\nFuture Segment Transition (%):")
print((transition * 100).round(1).to_string())

validation.to_csv(
    TABLE_DIR / "out_of_sample_validation.csv",
    index=False
)

future_summary.to_csv(
    TABLE_DIR / "future_segment_summary.csv"
)

transition.to_csv(
    TABLE_DIR / "segment_transition.csv"
)


# %% 4. BASKET SIZE ROBUSTNESS

print("\n[4] Basket Size Robustness")

robustness_rows = []

for minimum_size in [1, 3, 5, 8, 10]:

    sub = profiles[
        profiles["avg_basket_size"] >= minimum_size
    ]

    rotational_share = sub["segment"].eq(
        "Rotational"
    ).mean()

    corr = sub[
        ["avg_jaccard", "avg_novelty"]
    ].corr().iloc[0, 1]

    robustness_rows.append({
        "minimum_basket_size": minimum_size,
        "users": len(sub),
        "rotational_share": rotational_share,
        "correlation": corr,
    })

robustness = pd.DataFrame(robustness_rows)

print(robustness.round(3).to_string(index=False))

robustness.to_csv(
    TABLE_DIR / "basket_size_robustness.csv",
    index=False
)


# %% 5. VALIDATION SELECTION BIAS

print("\n[5] Selection Bias Check")

comparison = profiles.copy()

comparison["has_future_orders"] = (
    comparison["user_id"].isin(validation["user_id"])
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
    normalize="columns"
)

print("\nSegment distribution (%):")
print((segment_comparison * 100).round(1).to_string())

selection_summary.to_csv(
    TABLE_DIR /  "validation_selection_bias.csv"
)

segment_comparison.to_csv(
    TABLE_DIR / "selection_segment_composition.csv"
)


# %% 6. CHART STYLE AND EXPORT HELPER

print("\n[6] Creating Story Sketches")

# Use savefig instead of plt.show() to avoid Spyder Agg warnings.
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

ORANGE = "#D98B35"
BLUE = "#477FA3"
GREEN = "#398477"
GRAY = "#B8BCC2"
DARK = "#25313D"


def save_sketch(fig, filename):
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# %% 7. SKETCH 1 - SHOPPING BASKET SIMILARITY

fig, ax = plt.subplots(figsize=(10, 6))

values = window["product_jaccard"].dropna()

ax.hist(
    values,
    bins=np.linspace(0, 1, 26),
    color=BLUE,
    edgecolor="white",
    alpha=0.85
)

mean_j = values.mean()

ax.axvline(
    mean_j,
    color=ORANGE,
    linestyle="--",
    linewidth=2.2,
    label=f"Mean = {mean_j:.3f}"
)

ax.axvline(
    order_level_cutoff,
    color=GRAY,
    linestyle=":",
    linewidth=2,
    label=f"Exploratory cutoff = {order_level_cutoff:.3f}"
)

ax.set_title(
    "Consecutive Baskets Often Share Relatively Few Products",
    fontweight="bold",
    loc="left",
    pad=18
)

ax.set_xlabel("Product overlap with previous order (Jaccard similarity)")
ax.set_ylabel("Number of consecutive order pairs")
ax.set_xlim(0, 1)
ax.legend(frameon=False)

ax.text(
    0.98, 0.97,
    f"Orders 5-10 | n = {len(values):,} order pairs | "
    f"{len(profiles):,} shoppers",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color=DARK
)

ax.text(
    0.01,
    -0.17,
    "0 = no shared products; 1 = identical product sets. "
    "The cutoff is an exploratory comparison threshold.",
    transform=ax.transAxes,
    fontsize=9,
    color="gray"
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

save_sketch(
    fig,
    "sketch_01_similarity.png"
)


# %% 8. SKETCH 2 - CONDITIONAL NO-NEW-PRODUCT RATE

plot_summary = (
    conditional_summary
    .set_index("similarity_group")
    .reindex(["Low similarity", "High similarity"])
)

groups = ["Low basket similarity", "High basket similarity"]
no_new_rates = plot_summary["no_new_product_rate"].to_numpy()
has_new_rates = 1 - no_new_rates
pair_counts = plot_summary["order_pairs"].astype(int).to_numpy()

fig, ax = plt.subplots(figsize=(10, 6.3))
y_positions = np.arange(len(groups))

fig.suptitle(
    "A Different Basket Does Not Always Mean Discovery",
    x=0.12,
    y=0.98,
    ha="left",
    fontsize=17,
    fontweight="bold"
)

ax.barh(
    y_positions,
    no_new_rates,
    color=ORANGE,
    height=0.48,
    label="No new-to-history products"
)

ax.barh(
    y_positions,
    has_new_rates,
    left=no_new_rates,
    color="#DCE5EA",
    height=0.48,
    label="At least one new-to-history product"
)

for index, (no_new_rate, has_new_rate, pair_count) in enumerate(
    zip(no_new_rates, has_new_rates, pair_counts)
):
    ax.text(
        no_new_rate / 2,
        index,
        f"{no_new_rate:.1%}",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        no_new_rate + has_new_rate / 2,
        index,
        f"{has_new_rate:.1%}",
        ha="center",
        va="center",
        color=DARK,
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        1.02,
        index,
        f"n = {pair_count:,}",
        va="center",
        fontsize=10,
        color=DARK
    )

low_no_new_pairs = int(
    plot_summary.loc["Low similarity", "no_new_pairs"]
)
low_no_new_shoppers = int(
    plot_summary.loc["Low similarity", "no_new_shoppers"]
)

fig.text(
    0.12,
    0.89,
    f"{low_no_new_pairs:,} low-overlap pairs across "
    f"{low_no_new_shoppers:,} shoppers\n"
    "contained zero new-to-history products",
    color="#A6611E",
    fontsize=11,
    fontweight="bold"
)

ax.set_yticks(y_positions)
ax.set_yticklabels(groups, fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0, 1.13)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xticklabels(
    [f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)]
)
ax.set_xlabel("Share of consecutive order pairs")

ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2
)

fig.text(
    0.12,
    0.025,
    f"Orders 5-10 | Low similarity means product overlap below "
    f"{order_level_cutoff:.3f}. The cutoff is exploratory.",
    fontsize=9,
    color="gray"
)

fig.tight_layout(rect=[0, 0.15, 1, 0.84])

save_sketch(
    fig,
    "sketch_02_conditional_no_new.png"
)


# %% 9. SUPPLEMENT - PRODUCT VS CATEGORY SIMILARITY

# Use the same fixed-window users for all three levels.
similarity_means = pd.Series({
    "Product": profiles["avg_jaccard"].mean(),
    "Aisle": profiles["avg_aisle_jaccard"].mean(),
    "Department": profiles["avg_department_jaccard"].mean(),
})

fig, ax = plt.subplots(figsize=(10, 5.5))

levels = ["Product", "Aisle", "Department"]
values = similarity_means[levels].values

bars = ax.barh(
    levels,
    values,
    color=[ORANGE, BLUE, GREEN],
    height=0.55
)

ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Average Jaccard similarity")

ax.set_title(
    "Different Products Can Belong to Familiar Categories",
    fontweight="bold",
    loc="left",
    pad=18
)

for bar, value in zip(bars, values):
    ax.text(
        value + 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1%}",
        va="center",
        fontsize=12,
        fontweight="bold"
    )

ax.text(
    0.01, -0.19,
    "Category-level overlap is mechanically affected "
    "by classification granularity.",
    transform=ax.transAxes,
    fontsize=9,
    color="gray"
)

fig.tight_layout()

save_sketch(
    fig,
    "sketch_02_categories.png"
)


# %% 10. SUPPLEMENT - ROTATIONAL SHOPPERS

fig, ax = plt.subplots(figsize=(10, 6.5))

background = profiles[
    profiles["segment"] != "Rotational"
]

rotational = profiles[
    profiles["segment"] == "Rotational"
]

ax.scatter(
    background["avg_jaccard"],
    background["avg_novelty"],
    color=GRAY,
    alpha=0.40,
    s=28,
    label="Other shoppers"
)

ax.scatter(
    rotational["avg_jaccard"],
    rotational["avg_novelty"],
    color=ORANGE,
    alpha=0.9,
    s=55,
    label=f"Rotational (n={len(rotational)})"
)

ax.axvline(
    j_cut,
    linestyle="--",
    color="gray",
    alpha=0.5
)

ax.axhline(
    n_cut,
    linestyle="--",
    color="gray",
    alpha=0.5
)

ax.annotate(
    "Different baskets,\nbut fewer new-to-user products",
    xy=(0.10, 0.29),
    xytext=(0.37, 0.72),
    arrowprops={
        "arrowstyle": "->",
        "color": ORANGE,
        "lw": 1.8
    },
    color="#A6611E",
    fontsize=12,
    fontweight="bold"
)

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

ax.set_xlabel("Basket similarity (Product Jaccard)")
ax.set_ylabel("New-to-user product share")

ax.set_title(
    "A Different Basket Doesn't Always Mean a New Product",
    fontweight="bold",
    loc="left",
    pad=18
)

ax.legend(frameon=False, loc="upper right")

ax.text(
    0.02, -0.15,
    f"Orders 5-10 | n = {len(profiles)} shoppers | "
    "Exploratory median-based segments",
    transform=ax.transAxes,
    fontsize=9,
    color="gray"
)

fig.tight_layout()

save_sketch(
    fig,
    "sketch_03_discovery.png"
)


# %% 11. SUPPLEMENT - SIMPLIFIED BEHAVIORAL PERSISTENCE

fig, ax = plt.subplots(figsize=(10, 5.5))

groups = ["Routine", "Rotational"]

# Use previously calculated out-of-sample transition matrix
same = [
    transition.loc["Routine", "Routine"],
    transition.loc["Rotational", "Rotational"]
]

different = [1 - value for value in same]

y = np.arange(len(groups))

ax.barh(
    y,
    same,
    color=GREEN,
    height=0.42,
    label="Same classification"
)

ax.barh(
    y,
    different,
    left=same,
    color="#D9DEE2",
    height=0.42,
    label="Different classification"
)

ax.set_yticks(y)
ax.set_yticklabels(groups, fontsize=12)
ax.invert_yaxis()

ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xticklabels(
    [f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)]
)

# Add percentage labels and sample sizes
for i, group in enumerate(groups):

    percentage = same[i] * 100

    n = int(future_summary.loc[group, "users"])

    ax.text(
        same[i] / 2,
        i,
        f"{percentage:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=14,
        fontweight="bold"
    )

    ax.text(
        1.02,
        i,
        f"n = {n}",
        va="center",
        fontsize=11,
        color=DARK
    )

ax.set_xlim(0, 1.13)

ax.set_xlabel("Share of shoppers")

ax.set_title(
    "Some Shopping Patterns Persist Across Later Orders",
    fontsize=17,
    fontweight="bold",
    loc="left",
    pad=20
)

ax.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.27),
    ncol=2
)

fig.text(
    0.12,
    -0.03,
    "Orders 5-10 vs 11-15 | Complete future histories only.\n"
    "Classification thresholds are fixed; novelty can decline as history grows.",
    fontsize=9,
    color="gray"
)

fig.tight_layout(rect=[0, 0.08, 1, 1])

save_sketch(
    fig,
    "sketch_04_persistence.png"
)

# %% 12A. FIND A ROTATIONAL EXAMPLE

joined_path = JOINED_FILE


joined = pd.read_csv(joined_path)

print("joined_transactions rows:", len(joined))
print("joined_transactions columns:")
print(joined.columns.tolist())

# %% 12B. CANDIDATE ROTATIONAL ORDER PAIRS

# Rotational users from fixed-window profiles
rotational_users = profiles.loc[
    profiles["segment"] == "Rotational",
    "user_id"
]

# Look only at orders 5-10 for those users
rot_pairs = window[window["user_id"].isin(rotational_users)].copy()


# Prefer meaningful baskets, not tiny 1-3-item orders

candidates = rot_pairs[
    (rot_pairs["basket_size"] >= 5) &
    (rot_pairs["previous_basket_size"] >= 5) &
    (rot_pairs["product_jaccard"] <= 0.15) &
    (rot_pairs["novelty_share"] <= 0.20)
].copy()

candidates = candidates.sort_values(
    by=["novelty_share", "product_jaccard", "basket_size"],
    ascending=[True, True, False]
)

cols = [
    "user_id",
    "previous_order_id",
    "order_id",
    "order_number",
    "product_jaccard",
    "novelty_share",
    "basket_size",
    "previous_basket_size"
]

print("Suitable candidates:", len(candidates))
print(candidates[cols].head(20).to_string(index=False))

print(candidates[
    [
        "user_id",
        "previous_order_id",
        "order_id",
        "order_number",
        "product_jaccard",
        "novelty_share",
        "basket_size",
        "previous_basket_size"
    ]
].head(20).to_string(index=False))

# %% 12C. SHOW PRODUCTS FOR ONE ROTATIONAL EXAMPLE

if candidates.empty:
    raise ValueError(
        "No suitable candidates. Relax the filters in 11B."
    )

# Select the highest-ranked candidate from 11B
selected = candidates.iloc[0]

example_user = int(selected["user_id"])
example_prev_order = int(selected["previous_order_id"])
example_order = int(selected["order_id"])

# order_number refers to the current order in the pair
example_order_number = int(selected["order_number"])
example_prev_order_number = example_order_number - 1

print("Selected user:", example_user)
print(
    "Previous order:",
    example_prev_order,
    f"(Order {example_prev_order_number})"
)
print(
    "Current order:",
    example_order,
    f"(Order {example_order_number})"
)
print(
    "Product Jaccard:",
    selected["product_jaccard"]
)
print(
    "Product Novelty:",
    selected["novelty_share"]
)

example_rows = joined[
    (joined["user_id"] == example_user)
    & (
        joined["order_id"].isin([
            example_prev_order,
            example_order,
        ])
    )
].copy()

if example_rows.empty:
    raise ValueError(
        f"No transaction rows found for user {example_user}."
    )

# Confirm that both selected orders exist
found_orders = set(example_rows["order_id"].unique())
expected_orders = {
    example_prev_order,
    example_order,
}

missing_orders = expected_orders - found_orders

if missing_orders:
    raise ValueError(
        f"Missing transaction rows for order IDs: "
        f"{sorted(missing_orders)}"
    )

# Sort by order number and add-to-cart position
sort_cols = ["order_number"]

if "add_to_cart_order" in example_rows.columns:
    sort_cols.append("add_to_cart_order")

example_rows = example_rows.sort_values(sort_cols)

print("\nSelected order-pair transactions:")
print(
    example_rows.head(50).to_string(index=False)
)

# %% 12D. SIDE-BY-SIDE PRODUCT LISTS

prev_items = example_rows.loc[
    example_rows["order_id"] == example_prev_order,
    "product_name"
].tolist()

curr_items = example_rows.loc[
    example_rows["order_id"] == example_order,
    "product_name"
].tolist()

print("\nPREVIOUS ORDER")
for i, item in enumerate(prev_items, 1):
    print(f"{i}. {item}")

print("\nCURRENT ORDER")
for i, item in enumerate(curr_items, 1):
    print(f"{i}. {item}")
    
    
# %% 12E. EXPORT REAL BASKET EXAMPLE

import matplotlib.pyplot as plt

prev = joined[
    joined["order_id"] == example_prev_order
].sort_values("add_to_cart_order")

curr = joined[
    joined["order_id"] == example_order
].sort_values("add_to_cart_order")

prev_names = prev["product_name"].tolist()
curr_names = curr["product_name"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

common = set(prev["product_id"]) & set(curr["product_id"])

for ax, products, title in [
    (
        axes[0],
        prev,
        f"Order {example_prev_order_number} · "
        f"{len(prev)} products",
    ),
    (
        axes[1],
        curr,
        f"Order {example_order_number} · "
        f"{len(curr)} products",
    ),
]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(products) + 1)
    ax.axis("off")

    ax.set_title(title, fontsize=15, fontweight="bold")

    is_current_order = ax is axes[1]

    for i, row in enumerate(products.itertuples(), start=1):
        highlight = row.product_id in common
        y_position = len(products) + 0.4 - i

        if is_current_order:
            ax.scatter(
                0.025,
                y_position,
                s=24,
                color=ORANGE,
                zorder=3
            )

        ax.text(
            0.07 if is_current_order else 0.04,
            y_position,
            row.product_name,
            fontsize=10,
            color="#398477" if highlight else "#25313D",
            fontweight="bold" if highlight else "normal"
        )

fig.suptitle(
    "Different Baskets, Familiar Products",
    fontsize=19,
    fontweight="bold"
)

fig.text(
    0.5, 0.025,
    f"Shopper {example_user} | Jaccard = "
    f"{selected['product_jaccard']:.1%} | "
    f"New-to-history share = "
    f"{selected['novelty_share']:.1%}",
    ha="center",
    fontsize=11
)

fig.text(
    0.5,
    0.072,
    "Green = appears in both adjacent orders | "
    "Orange dot = appeared earlier in the shopper's history",
    ha="center",
    fontsize=9,
    color="gray"
)

fig.tight_layout(rect=[0, 0.11, 1, 0.93])

fig.savefig(
    FIGURE_DIR / "sketch_03_basket_example.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved basket example.")
# %% 13. FINAL SUMMARY

print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)

print(f"Historical fixed-window users: {len(profiles)}")
print(f"Out-of-sample validation users: {len(validation)}")
print(f"Historical Jaccard/Novelty r: {correlation:.3f}")

if "Rotational" in segment_summary.index:
    rotational_share = segment_summary.loc[
        "Rotational", "share"
    ]
    print(f"Rotational share: {rotational_share:.1%}")

print(f"\nCSV outputs: {TABLE_DIR}")
print(f"Story figures: {FIGURE_DIR}")

print("\nIMPORTANT LIMITATIONS:")
print("- First eligible user IDs, not a random sample.")
print("- Segment definitions use sample medians.")
print("- Novelty depends on recorded purchase history.")
print("- Future validation requires complete orders 11-15.")
print("- Category similarity depends on classification granularity.")
print("- No causal evidence about recommendation effectiveness.")
