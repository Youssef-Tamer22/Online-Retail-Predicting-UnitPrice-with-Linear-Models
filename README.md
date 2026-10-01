# Online Retail: Predicting UnitPrice with Linear Models

Predict the `UnitPrice` of an item in the Online Retail dataset using the linear regression family
(LinearRegression, Ridge, Lasso), evaluated with **R²** and **RMSE** on the original price scale.

## Results

Held-out test set (20% of the cleaned data, 106,248 rows):

| Model | R² | RMSE |
|---|---|---|
| LinearRegression | 0.8584 | 1.603 |
| **Ridge (alpha = 0.3)** | 0.8579 | 1.606 |
| Lasso (alpha = 1e-5) | 0.8399 | 1.705 |

Optional variant: adding TF-IDF word features from `Description` (not used in the simple version)
raises Ridge to R² 0.860 / RMSE 1.593. The gain is small because `StockCode` already identifies the product.

For comparison, the first version of the notebook scored very low. With the accounting rows left in,
every model stayed near R² 0.02 (RMSE about 156), because a handful of rows dominated the metric.

## Dataset

- File: `Online_Retail.xlsx` (541,909 rows, 8 columns)
- Columns used: `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, `Country`
- After cleaning: 531,239 rows (424,991 train / 106,248 test)

## Why the first version scored badly

1. **The strongest predictors were dropped.** `StockCode` was removed, but the same product sells at
   nearly the same price. Quantity, time and country say very little about price.
2. **A few non-product rows dominated the metric.** Rows priced above £100 carry 99.8% of the price
   variance, and 87% of them are accounting entries (postage, manual entries, Amazon fees, bank
   charges). The largest are a MANUAL entry at £38,970 and an AMAZON FEE at £17,836, which no model
   can predict.
3. **Clipping hid problems.** Clipping negative quantities to 1 and prices to 0-1000 kept the bad rows.
4. **Converting log prices back adds bias** if done with a plain `expm1`.

## Method

1. **Cleaning**
   - Drop exact duplicate rows
   - Remove prices <= 0 (negative and zero prices)
   - Remove accounting codes: `POST, DOT, M, C2, D, S, BANK CHARGES, AMAZONFEE, CRUK, B, ADJUST, ADJUST2`
     (2,890 rows)
   - Cancelled invoices are **kept**. A cancelled flag changed R² by less than 0.001.
2. **Features**
   - `StockCode`, `Country`, `month` (one-hot encoded)
   - `lq`: `log1p(|Quantity|)`, capped at the 99.9th percentile to tame extreme orders
   - `cancel`: 1 if the invoice number starts with "C"
3. **Split:** 80/20 with `random_state=42`. Encoders are fitted on the training data only (inside a
   pipeline), so nothing leaks from the test set.
4. **Target:** `log1p(UnitPrice)`, because prices are heavily skewed.
5. **Models:** LinearRegression, Ridge, Lasso. Alphas (0.3 and 1e-5) were tuned on a validation split
   inside the training data, so the test set was not used for tuning.
6. **Back-conversion:** predictions are converted to pounds with `exp(pred) * correction - 1`, where
   `correction = mean(exp(training residuals))` (Duan smearing). This removes the downward bias from
   exponentiating log predictions. Predictions are clipped at 0.
7. **Metrics:** R² and RMSE on the original £ scale.

## Key findings

- Using the product identity and removing accounting entries is what lifts the score.
- LinearRegression and Ridge score almost the same. With about 425,000 training rows, overfitting is
  small, so regularisation barely matters.
- Lasso is slightly lower because it zeroes out some product columns that carry price information.
- Cancelled invoices do not affect the prediction.

## Limitations

- Rare products with few sales are hard to predict.
- Products whose price changed over time cannot be captured well by a linear model.
- Accounting rows were removed because they are not product selling prices. If they are kept in the
  evaluation, R² drops to about 0.02 for every model.
- Alphas were tuned for the TF-IDF feature set and were not retuned for the simple version.

## How to run

```bash
pip install pandas numpy scikit-learn openpyxl matplotlib
```

1. Put `Online_Retail.xlsx` next to the notebook (or set `DATA_PATH` in the notebook).
2. Run all cells. A full run takes about 4 minutes, mostly the Lasso fit.

## Files

- `retail_unitprice_linear.ipynb`: the notebook (includes the optional TF-IDF features)
- `Online_Retail.xlsx`: the data
- `README.md`: this file
