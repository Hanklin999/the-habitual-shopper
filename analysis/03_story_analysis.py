"""The Habitual Shopper — Part III: story analysis.

Run from repository root:
    py analysis/03_story_analysis.py

Inputs:
    data/processed/joined_transactions.csv
    data/processed/basket_pairs.csv
Outputs:
    outputs/figures/03_*.png
    outputs/tables/03_*.csv

Analysis: eligible shoppers with Orders 5–10 in basket_pairs.csv and full
Orders 1–10 product history. Shopper-level averages weight each shopper equally.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save figures without requiring a GUI backend.
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
FIGURE_DIR = PROJECT_DIR / 'outputs' / 'figures'
TABLE_DIR = PROJECT_DIR / 'outputs' / 'tables'

LEVELS = {'Product': 'product_id', 'Aisle': 'aisle_id', 'Department': 'department_id'}
STATUS = ['Same as last order', 'Bought before, not last time', 'New to recorded history']
COLORS = {'Product': '#457B9D', 'Aisle': '#E9A23B', 'Department': '#3A9674'}
STATUS_COLORS = ['#457B9D', '#E9A23B', '#8DA6A0']


def run():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ('joined_transactions.csv', 'basket_pairs.csv'):
        if not (PROCESSED_DIR / filename).is_file():
            raise FileNotFoundError(
                f'Missing {PROCESSED_DIR / filename}. Run analysis/01_prepare_data.py first.'
            )

    tx = pd.read_csv(PROCESSED_DIR / 'joined_transactions.csv')
    pairs = pd.read_csv(PROCESSED_DIR / 'basket_pairs.csv')
    required = {'user_id', 'order_id', 'order_number', 'product_id', 'product_name',
                'aisle_id', 'department_id', 'aisle', 'department'}
    if not required.issubset(tx.columns):
        raise ValueError(f'Missing joined_transactions columns: {sorted(required - set(tx.columns))}')
    if not {'user_id', 'order_number'}.issubset(pairs.columns):
        raise ValueError('basket_pairs.csv needs user_id and order_number')
    if tx.duplicated(['user_id', 'order_number', 'product_id']).any():
        raise ValueError('Duplicate user/order/product in joined_transactions.csv')
    if tx[list(LEVELS.values())].isna().any().any():
        raise ValueError('Product, aisle, and department identifiers must not be missing')

    # Use the original fixed-window eligibility, and require full observed history.
    counts = (pairs.loc[pairs.order_number.between(5, 10)]
              .groupby('user_id').order_number.nunique())
    initial_users = set(counts.index[counts.eq(6)])
    available = (tx.loc[tx.order_number.between(1, 10)]
                 .groupby('user_id').order_number.agg(lambda x: set(x)))
    eligible = sorted(uid for uid in initial_users
                      if set(range(1, 11)).issubset(available.get(uid, set())))
    if not eligible:
        raise ValueError('No users with complete Orders 1–10 in both processed input files')
    print(f'Eligible shoppers: {len(eligible)} (initial fixed-window users: {len(initial_users)})')
    if len(eligible) != len(initial_users):
        print('Note: some fixed-window shoppers were excluded for incomplete recorded history.')

    tx = tx[tx.user_id.isin(eligible) & tx.order_number.between(1, 10)].copy()
    baskets = {
        (uid, num): (group.sort_values('add_to_cart_order')
                     if 'add_to_cart_order' in group.columns else group)
        for (uid, num), group in tx.groupby(['user_id', 'order_number'])
    }

    order_rows, item_rows = [], []
    for uid in eligible:
        history = set()
        for num in range(1, 11):
            curr = baskets[(uid, num)]
            current_products = set(curr.product_id)
            if num >= 5:
                prev = baskets[(uid, num - 1)]
                prev_products = set(prev.product_id)
                metrics = {'user_id': uid, 'order_number': num,
                           'basket_size': len(curr)}
                # For each current *product*, ask if its ID/category appeared
                # in the immediately previous basket. Denominator = current products.
                for label, col in LEVELS.items():
                    metrics[label] = curr[col].isin(set(prev[col])).mean()
                for rec in curr.itertuples(index=False):
                    product = rec.product_id
                    status = (STATUS[0] if product in prev_products else
                              STATUS[1] if product in history else STATUS[2])
                    item_rows.append({
                        'user_id': uid, 'order_number': num,
                        'product_id': product, 'product_name': rec.product_name,
                        'aisle': rec.aisle, 'department': rec.department,
                        'status': status
                    })
                order_rows.append(metrics)
            history.update(current_products)

    order_metrics = pd.DataFrame(order_rows)
    items = pd.DataFrame(item_rows)
    assert len(order_metrics) == 6 * len(eligible)
    assert len(items) == int(order_metrics.basket_size.sum())

    # ACT 1 — three aligned repeat-share histograms.
    shopper_repeat = order_metrics.groupby('user_id')[list(LEVELS)].mean() * 100
    ok = ((shopper_repeat['Product'] <= shopper_repeat['Aisle'] + 1e-9) &
          (shopper_repeat['Aisle'] <= shopper_repeat['Department'] + 1e-9))
    if not ok.all():
        raise AssertionError('Product <= Aisle <= Department must hold for nested categories')

    shopper_repeat.to_csv(TABLE_DIR / '03_shopper_repeat_profiles.csv')
    order_metrics.to_csv(TABLE_DIR / '03_order_repeat_metrics.csv', index=False)
    items.to_csv(TABLE_DIR / '03_item_purchase_status.csv', index=False)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, sharey=True)
    edges = np.linspace(0, 100, 6)
    y_max = max(np.histogram(shopper_repeat[c], bins=edges)[0].max() for c in LEVELS)
    for ax, level in zip(axes, LEVELS):
        values = shopper_repeat[level]
        ax.hist(values, bins=edges, color=COLORS[level], edgecolor='white')
        ax.axvline(values.median(), color='#222222', linestyle='--',
                   label=f'Median {values.median():.1f}%')
        ax.set_title(f'{level} level', loc='left', weight='bold')
        ax.set_ylabel('Shoppers')
        ax.set_ylim(0, max(1, y_max * 1.2))
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xticks(edges)
    axes[-1].set_xticklabels([f'{x:.0f}%' for x in edges])
    axes[-1].set_xlabel('Average previous-basket match share')
    fig.suptitle('Different Products, Familiar Categories?', weight='bold', fontsize=16)
    fig.text(.5, .01, f'Same {len(eligible)} shoppers | Orders 5–10 | shopper-weighted',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, .96])
    fig.savefig(FIGURE_DIR / '03_repeat_distributions.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print('Saved figure: 03_repeat_distributions.png')

    # ACT 2 — classify every item by order history, then average shares
    # across six orders per shopper and across shoppers (equal user weight).
    order_composition = (items.groupby(['user_id', 'order_number', 'status'])
                         .size().unstack(fill_value=0).reindex(columns=STATUS, fill_value=0))
    order_composition = order_composition.div(order_composition.sum(axis=1), axis=0)
    shopper_composition = order_composition.groupby(level='user_id').mean()
    if not np.allclose(shopper_composition.sum(axis=1), 1):
        raise AssertionError('Every shopper composition must sum to 100%')
    shopper_composition_pct = shopper_composition * 100
    shopper_composition_pct.to_csv(TABLE_DIR / '03_shopper_basket_composition.csv')

    summary = pd.DataFrame({
        'status': STATUS,
        'mean_share_percent': (shopper_composition_pct.mean().reindex(STATUS).values),
        'median_shopper_share_percent': (shopper_composition_pct.median().reindex(STATUS).values)
    })
    summary.to_csv(TABLE_DIR / '03_basket_composition_summary.csv', index=False)
    assert np.isclose(summary.mean_share_percent.sum(), 100)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    left = 0.0
    for label, color, value in zip(STATUS, STATUS_COLORS, summary.mean_share_percent):
        ax.barh([0], [value], left=left, height=.52, color=color, label=label)
        if value >= 6:
            ax.text(left + value / 2, 0, f'{value:.1f}%', ha='center', va='center',
                    weight='bold', color='white' if color == '#457B9D' else '#212121')
        left += value
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Average share of products in each basket (%)')
    ax.set_title('What Is Actually New in the Shopping Basket?', loc='left',
                 weight='bold', fontsize=15)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, -.22),
              frameon=False, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.5, .02, f'Orders 5–10 | {len(eligible)} shoppers | equal shopper weights',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, 1])
    fig.savefig(FIGURE_DIR / '03_basket_composition.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print('Saved figure: 03_basket_composition.png')


    # ACT 3 — Real shopper example
    # Shopper 977 was selected after reviewing the observed data.
    # This is an illustrative case, not a representative sample.

    case_uid = 977

    example = items[
        (items["user_id"] == case_uid)
        & (items["order_number"].between(8, 10))
    ].copy()

    if example.empty or example["order_number"].nunique() != 3:
        raise ValueError(
            "Shopper 977 is not available in the current data."
        )

    # Save product details for the website
    example = example.sort_values(
        ["order_number", "status", "product_name"]
    )

    example.to_csv(
        TABLE_DIR / "03_shopper_example_items.csv",
        index=False
    )

    # Count the three types of products in each order
    case_counts = (
        example.groupby(["order_number", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS, fill_value=0)
    )

    case_shares = (
        case_counts.div(case_counts.sum(axis=1), axis=0)
        * 100
    )

    case_shares.to_csv(
        TABLE_DIR / "03_shopper_example_summary.csv"
    )

    # Create a 100% stacked bar for Orders 8–10
    fig, ax = plt.subplots(figsize=(9, 4.5))

    left = np.zeros(len(case_shares))

    for label, color in zip(STATUS, STATUS_COLORS):
        values = case_shares[label].to_numpy()

        bars = ax.barh(
            [f"Order {n}" for n in case_shares.index],
            values,
            left=left,
            color=color,
            label=label,
            height=0.55
        )

        ax.bar_label(
            bars,
            labels=[
                f"{v:.0f}%" if v >= 9 else ""
                for v in values
            ],
            label_type="center",
            fontsize=10
        )

        left += values

    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of products in each order (%)")

    ax.set_title(
        "One Shopper, Three Different Baskets",
        loc="left",
        fontsize=15,
        weight="bold"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=9
    )

    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "03_shopper_timeline.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Selected shopper: {case_uid}")
    print(case_counts)
    print("Saved figure: 03_shopper_timeline.png")


    # Optional high-level table for writing the story accurately.
    report = pd.DataFrame({
        'level': list(LEVELS),
        'mean_repeat_percent': shopper_repeat.mean().values,
        'median_repeat_percent': shopper_repeat.median().values,
        'shoppers_below_20_percent': (shopper_repeat < 20).sum().values,
        'shoppers_at_least_60_percent': (shopper_repeat >= 60).sum().values,
    })
    report.to_csv(TABLE_DIR / '03_repeat_summary.csv', index=False)

    print('\nRepeat share by level (shopper averages, %):')
    print(report[['level', 'mean_repeat_percent', 'median_repeat_percent']].round(1).to_string(index=False))
    print('\nBasket composition (shopper-weighted, %):')
    print(summary[['status', 'mean_share_percent']].round(1).to_string(index=False))
    print(f'\nExample shopper ID: {case_uid} | Orders 8–10 | {len(example)} item rows')
    print(f'Figures: {FIGURE_DIR}')
    print(f'Tables: {TABLE_DIR}')
    print('Note: category-level match rates mechanically rise with broader categories.')
    print('Recommendation strategies are hypotheses, not estimated treatment effects.')


if __name__ == '__main__':
    run()
