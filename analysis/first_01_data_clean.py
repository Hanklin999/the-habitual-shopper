
"""
The Habitual Shopper
Part II - Behavioral Validation & Storytelling Sketches

CMU - Telling Stories with Data

Input:
    habitual_results/basket_pairs.csv

Outputs:
    Fixed-window user profiles
    Behavioral segmentation summary
    Out-of-sample validation
    Segment transition matrix
    Basket-size robustness
    Selection-bias check
    Four storytelling sketches

No raw Instacart transaction scanning is required.
"""

# %% 0. SETUP

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(
    r"D:\CMU_2026Fall\9_TellingStories"
    r"\final_project\instcart_dataset"
)

RESULTS_DIR = BASE_DIR / "habitual_results"
SKETCH_DIR = RESULTS_DIR / "story_sketches"

SKETCH_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RESULTS_DIR / "basket_pairs.csv"

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
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

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
    RESULTS_DIR / "fixed_window_user_profiles.csv",
    index=False
)

segment_summary.to_csv(
    RESULTS_DIR / "segment_summary.csv"
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
    RESULTS_DIR / "out_of_sample_validation.csv",
    index=False
)

future_summary.to_csv(
    RESULTS_DIR / "future_segment_summary.csv"
)

transition.to_csv(
    RESULTS_DIR / "segment_transition.csv"
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
    RESULTS_DIR / "basket_size_robustness.csv",
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
    RESULTS_DIR / "validation_selection_bias.csv"
)

segment_comparison.to_csv(
    RESULTS_DIR / "selection_segment_composition.csv"
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
    path = SKETCH_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# %% 7. SKETCH 1 - SHOPPING BASKET SIMILARITY

fig, ax = plt.subplots(figsize=(10, 6))

values = profiles["avg_jaccard"].dropna()

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

ax.set_title(
    "Shopping Baskets Often Change Between Orders",
    fontweight="bold",
    loc="left",
    pad=18
)

ax.set_xlabel("Average product-level basket similarity")
ax.set_ylabel("Number of shoppers")
ax.set_xlim(0, 1)
ax.legend(frameon=False)

ax.text(
    0.98, 0.97,
    f"Orders 5-10 | n = {len(profiles)} shoppers",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color=DARK
)

fig.tight_layout()

save_sketch(
    fig,
    "sketch_01_similarity.png"
)


# %% 8. SKETCH 2 - PRODUCT VS CATEGORY SIMILARITY

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


# %% 9. SKETCH 3 - ROTATIONAL SHOPPERS

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


# %% 10. SKETCH 4 - SIMPLIFIED BEHAVIORAL PERSISTENCE

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

# %% 11A. FIND A ROTATIONAL EXAMPLE

joined_path = RESULTS_DIR / "joined_transactions.csv"

if not joined_path.exists():
    raise FileNotFoundError(f"Cannot find: {joined_path}")

joined = pd.read_csv(joined_path)

print("joined_transactions rows:", len(joined))
print("joined_transactions columns:")
print(joined.columns.tolist())

# %% 11B. CANDIDATE ROTATIONAL ORDER PAIRS

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

# %% 11C. SHOW PRODUCTS FOR ONE ROTATIONAL EXAMPLE

if candidates.empty:
    raise ValueError("No suitable candidates. Relax the filters in 11B.")

selected = candidates.iloc[0]

example_user = int(selected["user_id"])
example_prev_order = int(selected["previous_order_id"])
example_order = int(selected["order_id"])

print("Selected user:", example_user)
print("Previous order:", example_prev_order)
print("Current order:", example_order)
print("Product Jaccard:", selected["product_jaccard"])
print("Product Novelty:", selected["novelty_share"])

example_rows = joined[
    (joined["user_id"] == example_user) &
    (joined["order_id"].isin([example_prev_order, example_order]))
].copy()

# If add_to_cart_order exists, sort by it
sort_cols = ["order_id"]
if "add_to_cart_order" in example_rows.columns:
    sort_cols.append("add_to_cart_order")

example_rows = example_rows.sort_values(sort_cols)

print(example_rows.head(50).to_string(index=False))

# %% 11D. SIDE-BY-SIDE PRODUCT LISTS

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
    
    
# %% 11E. EXPORT REAL BASKET EXAMPLE

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
    (axes[0], prev, "Order 9 · 6 products"),
    (axes[1], curr, "Order 10 · 7 products")
]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(products) + 1)
    ax.axis("off")

    ax.set_title(title, fontsize=15, fontweight="bold")

    for i, row in enumerate(products.itertuples(), start=1):
        highlight = row.product_id in common

        ax.text(
            0.04,
            len(products) + 0.4 - i,
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
    f"User {example_user} | Jaccard = "
    f"{selected['product_jaccard']:.1%} | "
    f"New-to-user share = "
    f"{selected['novelty_share']:.1%}",
    ha="center",
    fontsize=11
)

fig.tight_layout(rect=[0, 0.06, 1, 0.93])

fig.savefig(
    sketch_dir / "sketch_03_basket_example.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved basket example.")
# %% 11. FINAL SUMMARY

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

print(f"\nCSV outputs: {RESULTS_DIR}")
print(f"Story sketches: {SKETCH_DIR}")

print("\nIMPORTANT LIMITATIONS:")
print("- First eligible user IDs, not a random sample.")
print("- Segment definitions use sample medians.")
print("- Novelty depends on recorded purchase history.")
print("- Future validation requires complete orders 11-15.")
print("- Category similarity depends on classification granularity.")
print("- No causal evidence about recommendation effectiveness.")
