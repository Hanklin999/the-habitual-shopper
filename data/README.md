# Data Setup

This project uses the Instacart Market Basket Analysis dataset.

Download the data from:

https://www.kaggle.com/competitions/instacart-market-basket-analysis/data

Place the following files in `data/raw/`:

- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

The raw data is not included in this repository because of its size.

Run:

```powershell
py analysis/01_prepare_data.py
py analysis/02_analyze_story.py