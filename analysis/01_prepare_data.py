from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(RAW_DIR / "orders.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
aisles = pd.read_csv(RAW_DIR / "aisles.csv")
departments = pd.read_csv(RAW_DIR / "departments.csv")

#%%

# Select first 100 users

prior_orders = orders[
    orders["eval_set"] == "prior"
].copy()

selected_users = (
    prior_orders["user_id"]
    .drop_duplicates()
    .sort_values()
    .head(1000)
)

selected_orders = prior_orders[
    prior_orders["user_id"].isin(selected_users)
].copy()

selected_order_ids = set(selected_orders["order_id"])

chunks = []

for chunk in pd.read_csv(
    RAW_DIR / "order_products__prior.csv",
    chunksize=1_000_000,
):
    keep = chunk[
        chunk["order_id"].isin(selected_order_ids)
    ]

    if not keep.empty:
        chunks.append(keep)

order_products = pd.concat(chunks, ignore_index=True)

#%%
# join product & category

product_metadata = (
    products
    .merge(aisles, on="aisle_id", how="left")
    .merge(departments, on="department_id", how="left")
)

joined = (
    order_products
    .merge(selected_orders, on="order_id", how="left")
    .merge(product_metadata, on="product_id", how="left")
)

joined.to_csv(
    PROCESSED_DIR / "joined_transactions.csv",
    index=False,
)

#%%%
# join order level

order_baskets = (
    joined
    .groupby(
        [
            "user_id",
            "order_id",
            "order_number",
            "days_since_prior_order",
        ],
        dropna=False,
    )
    .agg(
        products=("product_id", lambda x: set(x)),
        aisles=("aisle_id", lambda x: set(x)),
        departments=("department_id", lambda x: set(x)),
        basket_size=("product_id", "size"),
        novelty_share=("reordered", lambda x: (x == 0).mean()),
    )
    .reset_index()
    .sort_values(["user_id", "order_number"])
)

#%%
# join Jaccard function

def jaccard(left, right):
    union = left | right

    if not union:
        return 0.0

    return len(left & right) / len(union)

#%%
# adding near order

rows = []

for user_id, user_orders in order_baskets.groupby("user_id"):
    user_orders = user_orders.sort_values("order_number")
    previous = None

    for current in user_orders.itertuples():
        if previous is not None:
            rows.append({
                "user_id": user_id,
                "previous_order_id": previous.order_id,
                "order_id": current.order_id,
                "order_number": current.order_number,
                "product_jaccard": jaccard(
                    previous.products,
                    current.products,
                ),
                "aisle_jaccard": jaccard(
                    previous.aisles,
                    current.aisles,
                ),
                "department_jaccard": jaccard(
                    previous.departments,
                    current.departments,
                ),
                "novelty_share": current.novelty_share,
                "basket_size": current.basket_size,
                "previous_basket_size": previous.basket_size,
                "days_since_prior_order": current.days_since_prior_order,
            })

        previous = current

basket_pairs = pd.DataFrame(rows)

basket_pairs.to_csv(
    PROCESSED_DIR / "basket_pairs.csv",
    index=False,
)

print(f"Users: {basket_pairs['user_id'].nunique():,}")
print(f"Order pairs: {len(basket_pairs):,}")
print("Data preparation complete.")