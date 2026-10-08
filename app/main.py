from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict_credit_risk


app = FastAPI(
    title="AI-Powered Alternative Credit Scoring API",
    description="API for predicting customer credit risk using a tuned XGBoost model.",
    version="1.0.0"
)


class CreditApplication(BaseModel):

    LIMIT_BAL: float
    AGE: float

    PAY_0: float
    PAY_2: float
    PAY_3: float
    PAY_4: float
    PAY_5: float
    PAY_6: float

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

    max_payment_delay: float
    avg_payment_delay: float
    months_with_delay: float
    severe_delay_months: float

    avg_bill_amount: float
    bill_volatility: float

    avg_payment_amount: float
    payment_volatility: float

    avg_utilization: float
    max_utilization: float
    avg_payment_ratio: float

    recent_bill_change: float
    recent_payment_change: float

    total_billed_6m: float
    total_paid_6m: float
    total_payment_ratio_6m: float


@app.get("/")
def home():
    return {
        "message": "AI-Powered Alternative Credit Scoring API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Tuned XGBoost",
        "threshold": 0.60
    }


@app.post("/predict")
def predict(application: CreditApplication):

    try:
        input_data = application.model_dump()

        result = predict_credit_risk(input_data)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )