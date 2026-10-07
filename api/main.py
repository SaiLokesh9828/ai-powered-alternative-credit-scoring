from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from src.predict import predict_credit_risk

app = FastAPI(title='Alternative Credit Scoring API', version='1.0.0')

class CustomerInput(BaseModel):
    model_config = ConfigDict(extra='allow')
    LIMIT_BAL: float = Field(gt=0)
    SEX: int | None = None
    EDUCATION: int | None = None
    MARRIAGE: int | None = None
    AGE: int = Field(ge=18, le=100)
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float = 0
    BILL_AMT2: float = 0
    BILL_AMT3: float = 0
    BILL_AMT4: float = 0
    BILL_AMT5: float = 0
    BILL_AMT6: float = 0
    PAY_AMT1: float = 0
    PAY_AMT2: float = 0
    PAY_AMT3: float = 0
    PAY_AMT4: float = 0
    PAY_AMT5: float = 0
    PAY_AMT6: float = 0

@app.get('/health')
def health():
    return {'status': 'ok', 'model_exists': Path('models/credit_risk_model.joblib').exists()}

@app.post('/predict')
def predict(customer: CustomerInput):
    try:
        return predict_credit_risk(customer.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
