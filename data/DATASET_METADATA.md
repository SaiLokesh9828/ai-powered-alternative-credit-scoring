# Dataset metadata

Source: UCI Machine Learning Repository — Default of Credit Card Clients
Source URL: https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients

- Observations: 30,000
- Predictors: 23 (excluding ID and target)
- Total source columns: 25 (ID + 23 predictors + target)
- Target: `default.payment.next.month` (renamed to `default_payment_next_month`)
- Class 0 / non-default: 23,364 (77.88%)
- Class 1 / default: 6,636 (22.12%)
- Missing values reported by UCI: none
- Historical window: April–September 2005; target is next-month default
- Geography/context: credit-card clients in Taiwan

Predictor columns:
`LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`, `BILL_AMT1`, `BILL_AMT2`, `BILL_AMT3`, `BILL_AMT4`, `BILL_AMT5`, `BILL_AMT6`, `PAY_AMT1`, `PAY_AMT2`, `PAY_AMT3`, `PAY_AMT4`, `PAY_AMT5`, `PAY_AMT6`

Limitations: no transaction-level behavioral feed, no thin-file indicator, no consent/alternative-data fields, historical population, and no direct measurement of NPA reduction or decision turnaround time.
