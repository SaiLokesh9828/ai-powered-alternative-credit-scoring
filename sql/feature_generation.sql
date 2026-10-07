-- Illustrative feature-generation logic for a production warehouse.
-- Adapt table/column names to the bank's schema. The source dataset itself is not transactional.
SELECT
    customer_id,
    credit_limit,
    AVG(monthly_bill_amount) AS avg_bill_amount,
    STDDEV(monthly_bill_amount) AS bill_volatility,
    AVG(monthly_payment_amount) AS avg_payment_amount,
    SUM(CASE WHEN repayment_status > 0 THEN 1 ELSE 0 END) AS months_with_delay,
    MAX(repayment_status) AS max_payment_delay,
    SUM(monthly_payment_amount) / NULLIF(ABS(SUM(monthly_bill_amount)), 0) AS total_payment_ratio_6m,
    AVG(monthly_bill_amount / NULLIF(credit_limit, 0)) AS avg_utilization
FROM customer_credit_history
WHERE observation_date <= :decision_date
GROUP BY customer_id, credit_limit;
