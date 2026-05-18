"""
generate_data.py
----------------
Generates a realistic e-commerce dataset for conversion optimisation analysis.
Designed to match the IBM Python for Data Science project on Aditi Tyagi's resume.

Dataset: 50,000 user records with A/B test groups, pricing, and conversion labels.
"""

import numpy as np
import pandas as pd

def generate_dataset(n=50000, seed=42):
    np.random.seed(seed)

    # ── User demographics ───────────────────────────────────────────────────
    user_id       = np.arange(1, n + 1)
    age           = np.random.randint(18, 65, n)
    gender        = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.48, 0.48, 0.04])
    country       = np.random.choice(
        ['India', 'US', 'UK', 'Germany', 'Australia'],
        n, p=[0.40, 0.25, 0.15, 0.10, 0.10]
    )

    # ── A/B test group (split 50/50 — 10K shown to each in test, rest baseline) ──
    # 10K users in the A/B test (5K control, 5K treatment), rest are observation
    ab_group = np.where(
        user_id <= 10000,
        np.where(user_id <= 5000, 'control', 'treatment'),
        'observation'
    )

    # ── Pricing & product ───────────────────────────────────────────────────
    base_price    = np.random.uniform(5, 200, n).round(2)
    # Introduce 17% pricing errors (wrong values) — to be detected in EDA
    pricing_error = np.random.choice([True, False], n, p=[0.17, 0.83])
    listed_price  = np.where(
        pricing_error,
        (base_price * np.random.uniform(1.5, 3.0, n)).round(2),  # inflated error
        base_price
    )
    category      = np.random.choice(
        ['Electronics', 'Fashion', 'Home & Kitchen', 'Books', 'Sports'],
        n, p=[0.25, 0.30, 0.20, 0.10, 0.15]
    )
    discount_pct  = np.random.choice([0, 5, 10, 15, 20, 25, 30], n,
                                      p=[0.30, 0.10, 0.20, 0.15, 0.10, 0.10, 0.05])

    # ── Session behaviour ───────────────────────────────────────────────────
    time_on_site  = np.random.exponential(5, n).round(2)          # minutes
    pages_viewed  = np.random.poisson(4, n) + 1
    device        = np.random.choice(['Mobile', 'Desktop', 'Tablet'], n, p=[0.55, 0.35, 0.10])
    prev_purchases= np.random.poisson(1.5, n)

    # ── Conversion logic ────────────────────────────────────────────────────
    # Base probability
    prob = (
        0.05
        + 0.10 * (discount_pct / 30)
        + 0.08 * (time_on_site / time_on_site.max())
        + 0.06 * (pages_viewed / pages_viewed.max())
        + 0.05 * (prev_purchases / (prev_purchases.max() + 1))
        - 0.04 * (listed_price / listed_price.max())
        # Treatment group gets +12% lift
        + np.where(ab_group == 'treatment', 0.12, 0.0)
    )
    prob = np.clip(prob, 0.01, 0.95)
    converted = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        'user_id':        user_id,
        'age':            age,
        'gender':         gender,
        'country':        country,
        'ab_group':       ab_group,
        'category':       category,
        'base_price':     base_price,
        'listed_price':   listed_price,
        'pricing_error':  pricing_error,
        'discount_pct':   discount_pct,
        'time_on_site_min': time_on_site,
        'pages_viewed':   pages_viewed,
        'device':         device,
        'prev_purchases': prev_purchases,
        'converted':      converted
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/ecommerce_data_raw.csv", index=False)
    print(f"Dataset generated: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Conversion rate:   {df['converted'].mean():.2%}")
    print(f"Pricing errors:    {df['pricing_error'].mean():.2%}")
    print(f"A/B test users:    {(df['ab_group'] != 'observation').sum():,}")
