# Ecommerce-conversion-analysis
# E-Commerce Conversion Optimisation & Predictive Modelling

**Author:** Aditi Tyagi  
**Tools:** Python · Pandas · NumPy · SciPy · scikit-learn · Matplotlib · Seaborn  
**Dataset:** Synthetic e-commerce dataset (50,000 user records)

---

## Project Overview

This project performs end-to-end data analysis on an e-commerce dataset covering:

1. **Exploratory Data Analysis (EDA)** — conversion rates by category, device, and discount
2. **Data Cleaning** — automated detection and removal of pricing errors (~17% error rate)
3. **A/B Hypothesis Testing** — statistical validation of a new pricing variant (T-test, p-value)
4. **Predictive Modelling** — Linear Regression to predict conversion likelihood (R²=0.87)

---

## Key Results

| Metric | Result |
|---|---|
| Dataset size | 50,000 records × 15 features |
| Pricing error rate detected | **16.8%** (8,395 records flagged & removed) |
| A/B test conversion lift | **+12 percentage points** (treatment vs control) |
| T-test p-value | **p < 0.0001** (statistically significant) |
| Regression R² | **0.87** on 20% held-out test set |
| RMSE | **0.04** |

---

## Project Structure

```
ecommerce_conversion_analysis/
│
├── data/
│   ├── ecommerce_data_raw.csv       # Original 50K dataset
│   └── ecommerce_data_clean.csv     # After pricing error removal
│
├── src/
│   ├── generate_data.py             # Dataset generation script
│   └── analysis.py                  # Main analysis (EDA → Clean → A/B → Model)
│
├── notebooks/
│   └── analysis.ipynb               # Jupyter notebook walkthrough
│
├── outputs/
│   ├── 01_eda_conversion_overview.png
│   ├── 02_eda_price_discount.png
│   ├── 03_pricing_error_analysis.png
│   ├── 04_ab_test_results.png
│   ├── 05_regression_model.png
│   └── 06_correlation_heatmap.png
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-conversion-analysis.git
cd ecommerce-conversion-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate dataset
```bash
python src/generate_data.py
```

### 4. Run full analysis
```bash
python src/analysis.py
```

### 5. Or open the Jupyter notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Analysis Walkthrough

### EDA
- Conversion rate varies significantly by product category and device type
- Mobile users have the highest share but lower conversion vs desktop
- Discount percentage positively correlates with conversion

### Data Cleaning
- 16.8% of records had inflated listed prices (1.5×–3× the base price)
- Detected using `listed_price > base_price * 1.3` threshold
- Removed before modelling to avoid bias

### A/B Test
```
Null Hypothesis (H₀):  No difference in conversion between control and treatment
Alt  Hypothesis (H₁):  Treatment has higher conversion rate

Result: t=14.56, p<0.0001 → Reject H₀
Conclusion: Treatment variant delivers a statistically significant +12pp lift
```

### Linear Regression
- **Target:** Estimated conversion probability score (continuous 0–1)
- **Features:** Age, device, discount %, time on site, pages viewed, previous purchases, A/B group
- **Split:** 80% train / 20% test
- **Best predictors:** Discount %, time on site, previous purchases

---

## Charts

All visualisations are saved in the `/outputs` folder and include:
- Conversion rate by category and device
- Pricing error scatter plot (base vs listed price)
- A/B test bar chart with p-value annotation
- Feature coefficient chart
- Correlation heatmap

---

## Skills Demonstrated

`Python` `Pandas` `NumPy` `SciPy` `scikit-learn` `Matplotlib` `Seaborn`  
`EDA` `Data Cleaning` `Hypothesis Testing` `A/B Testing` `Linear Regression`  
`Statistical Analysis` `Data Visualisation` `Jupyter Notebook`

---

## Contact

**Aditi Tyagi**  
[LinkedIn](https://www.linkedin.com/in/aditi-tyagi-3bb12636b) · aditityagi1001@gmail.com
