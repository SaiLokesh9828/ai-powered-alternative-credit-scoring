import joblib
import pandas as pd


MODEL_PATH = "models/credit_risk_model.joblib"
THRESHOLD = 0.60


# Exact features used by the trained model
FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "max_payment_delay",
    "avg_payment_delay",
    "months_with_delay",
    "severe_delay_months",
    "avg_bill_amount",
    "bill_volatility",
    "avg_payment_amount",
    "payment_volatility",
    "avg_utilization",
    "max_utilization",
    "avg_payment_ratio",
    "recent_bill_change",
    "recent_payment_change",
    "total_billed_6m",
    "total_paid_6m",
    "total_payment_ratio_6m",
]


# Load trained pipeline
model = joblib.load(MODEL_PATH)


def predict_credit_risk(input_data: dict):
    """
    Predict credit risk for one customer.
    """

    # Check for missing features
    missing_features = [
        feature for feature in FEATURES
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Keep only required features and preserve training order
    df = pd.DataFrame(
        [[input_data[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    # Get probability of risky class
    probability = model.predict_proba(df)[0][1]

    # Apply selected threshold
    prediction = int(probability >= THRESHOLD)

    # Risk category
    if probability < 0.30:
        risk_category = "Low Risk"
    elif probability < 0.60:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    return {
        "risk_probability": round(float(probability), 4),
        "prediction": prediction,
        "risk_category": risk_category,
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":

    # Example customer
    sample_customer = {
        "LIMIT_BAL": 50000,
        "AGE": 25,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,

        "BILL_AMT1": 20000,
        "BILL_AMT2": 18000,
        "BILL_AMT3": 16000,
        "BILL_AMT4": 15000,
        "BILL_AMT5": 14000,
        "BILL_AMT6": 13000,

        "PAY_AMT1": 2000,
        "PAY_AMT2": 2000,
        "PAY_AMT3": 1800,
        "PAY_AMT4": 1800,
        "PAY_AMT5": 1700,
        "PAY_AMT6": 1600,

        "max_payment_delay": 0,
        "avg_payment_delay": 0,
        "months_with_delay": 0,
        "severe_delay_months": 0,

        "avg_bill_amount": 16000,
        "bill_volatility": 2500,
        "avg_payment_amount": 1766.67,
        "payment_volatility": 166.67,

        "avg_utilization": 0.32,
        "max_utilization": 0.40,
        "avg_payment_ratio": 0.11,

        "recent_bill_change": -1000,
        "recent_payment_change": -100,

        "total_billed_6m": 96000,
        "total_paid_6m": 10600,
        "total_payment_ratio_6m": 0.11,
    }

    result = predict_credit_risk(sample_customer)

    print()
    print("=" * 50)
    print("AI-POWERED ALTERNATIVE CREDIT SCORING")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 50)