# AI-Powered Alternative Credit Scoring System

**Repository:** `alternative-credit-scoring-ml`

A portfolio-ready credit-risk ML system that estimates the probability of next-month credit-card default and exposes the result through FastAPI. It is designed as a **decision-support prototype**, not as a production lending decision engine.

## Business problem

Banks need consistent probability-of-default estimates to support faster and more risk-aware credit decisions. The project focuses on customers with limited traditional information as a motivating use case, but the selected public dataset does **not** contain bank-account transaction streams or an explicit thin-file flag. Therefore, this repository does not claim to measure thin-file performance or any NPA improvement.

## Dataset

**UCI Default of Credit Card Clients**: 30,000 observations and 23 explanatory variables, with no missing values reported by UCI. The target is whether the customer defaulted on the next month's payment. The dataset contains six months of repayment-status history, bill amounts and previous payment amounts. It was collected for credit-card clients in Taiwan and does not contain modern open-banking transaction data, merchant categories, account-level cash-flow history, or a thin-file indicator.

Source: https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients

The repository downloads the public file at runtime rather than fabricating or silently replacing it. Dataset licensing/attribution should be retained when redistributing the data.

## Pipeline

```text
UCI dataset
   -> validation / duplicate handling
   -> leakage checks
   -> train/validation/test split (stratified)
   -> feature engineering
   -> sklearn ColumnTransformer
   -> Logistic Regression / Random Forest / XGBoost
   -> validation comparison
   -> CV hyperparameter tuning
   -> threshold analysis: 0.30 ... 0.80
   -> final test evaluation
   -> SHAP global + individual explanations
   -> Joblib model
   -> FastAPI /predict
```

### Leakage prevention

The target is excluded before feature generation. Splitting happens before model fitting and all preprocessing is fitted inside sklearn Pipelines/ColumnTransformers. No SMOTE is applied before splitting. This version uses class weights / XGBoost `scale_pos_weight` instead of synthetic oversampling. Obvious target-like columns are checked before training.

### Features

The UCI fields include credit limit, repayment status for six months, six months of bill statements and six months of previous payments. The feature layer adds:

- maximum/average payment delay
- months with any delay and severe delays
- average bill/payment amount
- bill/payment volatility
- average/max credit utilization proxy
- payment-to-bill ratios
- recent bill/payment changes
- six-month total billed and paid amounts

Features such as transaction frequency, spending variability, account age and true RFM are **not claimed as present in the UCI data**. `sql/feature_generation.sql` shows how similar features could be generated from a real transactional warehouse while enforcing a credit-decision cutoff.

## Fair lending / responsible AI

The baseline scoring pipeline excludes `SEX`, `EDUCATION`, and `MARRIAGE` from model training. They remain available in the input schema only for compatibility with the source data and potential offline fairness audits. This is not a guarantee of fairness: other variables can be proxies, and a real bank would need legal/compliance review, adverse-action reasoning, subgroup performance tests, privacy controls, human oversight, and governance before deployment.

## Evaluation

The pipeline reports:

- Precision, Recall, F1
- ROC-AUC
- PR-AUC
- confusion matrix counts
- ROC and Precision-Recall curves
- threshold analysis at 0.30, 0.40, 0.50, 0.60, 0.70 and 0.80

**No metrics are hard-coded in this repository.** `data/processed/evaluation_results.json` and figures are generated after you run the pipeline. Until then, results should be described as **To be generated after execution**.

Threshold selection is demonstrated using validation F1 as a transparent default. In a bank, the threshold should instead be chosen using approved loss/cost functions, capacity constraints, risk appetite, calibration, fairness constraints and policy rules. A higher threshold generally reduces the number of customers classified as high risk but can increase false negatives; a lower threshold generally catches more potential defaults but can increase false positives.

## Risk categories

- **LOW:** probability < 0.30
- **MEDIUM:** 0.30 <= probability < 0.50
- **HIGH:** probability >= 0.50

These are portfolio-demo thresholds, not regulatory or bank policy thresholds.

## SHAP

`src/shap_explain.py` creates a global SHAP summary plot and supports individual explanations. The API returns the top individual SHAP factors as `top_risk_factors`.

## Installation

### Local

```bash
git clone <your-repository-url>
cd alternative-credit-scoring-ml
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_download
python -m src.eda
python -m src.pipeline
python -m src.shap_explain
pytest -q
```

### Google Colab

Upload/extract the repository, then:

```python
%cd /content/alternative-credit-scoring-ml
!pip install -r requirements.txt
!python -m src.data_download
!python -m src.pipeline
!python -m src.shap_explain
!pytest -q
```

Training is CPU-compatible; XGBoost can use a GPU only if you intentionally configure a compatible environment. The repository does not require a GPU.

## FastAPI

After training:

```bash
uvicorn api.main:app --reload
```

Open `/docs` in the browser. Endpoints:

- `GET /health`
- `POST /predict`

Example JSON is in `api/example_request.json`.

## Streamlit (optional)

```bash
streamlit run streamlit_app.py
```

## Docker

```bash
docker build -t alternative-credit-scoring-ml .
docker run --rm -p 8000:8000 alternative-credit-scoring-ml
```

The image expects a trained `models/credit_risk_model.joblib` to be present if you want `/predict` to score immediately. For a reproducible container workflow, train the model first and include the artifact only in a controlled deployment package; it is ignored by Git in this repository.

## Project structure

```text
alternative-credit-scoring-ml/
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── example_request.json
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── config.py
│   ├── data_download.py
│   ├── data_processing.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── features.py
│   ├── leakage.py
│   ├── modeling.py
│   ├── pipeline.py
│   ├── predict.py
│   └── shap_explain.py
├── sql/
│   └── feature_generation.sql
├── tests/
├── Dockerfile
├── README.md
├── requirements.txt
├── streamlit_app.py
└── .gitignore
```

## Limitations

1. The source population is Taiwanese credit-card clients from 2005, not Indian bank customers.
2. It is not an open-banking/transaction dataset and has no explicit thin-file label.
3. The public dataset does not establish causal business impact, NPA reduction, faster turnaround time, or approval lift.
4. Calibration, drift monitoring, reject inference, stability testing, PSI, regulatory validation and production data governance are outside this portfolio implementation.
5. Probability thresholds are illustrative and must not be used for real lending without governance.

## Future improvements

- Add bank-approved transaction/open-banking features.
- Calibrate PD with Platt/isotonic methods and evaluate Brier score/calibration curves.
- Add PSI and population drift monitoring.
- Add subgroup fairness metrics and documented mitigation.
- Add model registry/versioning and data contracts.
- Add human-review and adverse-action reason workflows.
- Evaluate cost-sensitive thresholds using an approved credit-loss matrix.
- Add temporal validation when a longitudinal production dataset is available.

## Interview material

### 30-second explanation

“I built an end-to-end credit-risk decision-support system using the UCI Default of Credit Card Clients dataset. I engineered repayment-delay, utilization and payment-ratio features, prevented leakage, compared Logistic Regression, Random Forest and XGBoost, tuned the selected model with cross-validation, evaluated ROC-AUC and PR-AUC, and tested operating thresholds. I then added SHAP explanations and deployed the scoring pipeline through FastAPI with validation and Docker support. I explicitly avoid claiming NPA reduction because the public dataset cannot measure that business outcome.”

### 2-minute explanation

“The business problem is estimating probability of default consistently enough to support credit decisions. I chose a real UCI credit-default dataset with 30,000 observations and six months of repayment, billing and payment history. The dataset does not contain true bank transaction feeds or a thin-file flag, so I treat thin-file lending as the motivating use case rather than making an unsupported performance claim. I first validate the target and check obvious leakage. I split the data into train, validation and test sets before fitting preprocessing. Feature engineering creates interpretable repayment-delay, utilization, payment-ratio and trend features. I exclude direct demographic variables from the baseline model to reduce unnecessary lending-risk concerns. I compare Logistic Regression, Random Forest and XGBoost using class weighting, then tune the strongest validation model with stratified cross-validation. Evaluation includes ROC-AUC, PR-AUC, precision, recall, F1, confusion matrices and threshold analysis from 0.30 to 0.80. SHAP provides global and customer-level explanations. Finally, the trained pipeline is serialized with Joblib and served using FastAPI. In a real bank I would add calibration, temporal validation, drift monitoring, fairness testing, approved loss functions, privacy controls and human oversight before deployment.”

## 20 technical interview Q&A

1. **Why PR-AUC as well as ROC-AUC?** PR-AUC is more informative when the positive class is relatively uncommon because it focuses on precision-recall tradeoffs.
2. **Why split before preprocessing?** Fitting imputers/scalers on all data leaks validation/test information into training.
3. **Why use a Pipeline?** It keeps preprocessing and the estimator together and makes CV safer and reproducible.
4. **Why class weights?** They make mistakes on the minority class more costly without synthesizing observations.
5. **Why not SMOTE before splitting?** Synthetic samples could incorporate information derived from validation/test observations.
6. **Why Logistic Regression?** It is a strong, interpretable credit-risk baseline.
7. **Why Random Forest?** It captures nonlinear interactions with limited preprocessing.
8. **Why XGBoost?** Gradient-boosted trees are strong on structured tabular data and support nonlinear relationships.
9. **What is target leakage?** Using information unavailable at decision time or directly derived from the outcome.
10. **What is PD?** Probability of default over a defined future horizon; here the source target is next-month default.
11. **Why optimize a threshold?** A 0.5 cutoff is arbitrary; credit decisions have asymmetric business costs.
12. **What happens when the threshold decreases?** More cases are labeled high risk, usually increasing recall and potentially increasing false positives.
13. **What is calibration?** Agreement between predicted probabilities and observed event frequencies.
14. **Why is calibration important in credit?** A PD estimate is more useful for risk pricing, provisioning and policy when it has probabilistic meaning.
15. **What is SHAP?** A feature-attribution framework based on Shapley-value ideas that explains a prediction relative to a baseline.
16. **Why use a holdout test set?** It provides a final estimate after model selection/tuning.
17. **Why stratify?** It preserves the class ratio across splits.
18. **What is drift?** A change in data or relationships over time that can degrade model performance.
19. **Why not claim NPA reduction?** NPA impact requires real operational deployment and measured outcomes, not just model metrics.
20. **How would you productionize it?** Version data/model artifacts, enforce schemas, monitor drift and calibration, log decisions, govern access, test fairness, and establish human oversight.

## 15 banking / credit-risk interview Q&A

1. **What is a credit score?** A summarized risk indicator used with policy and other information to support credit decisions.
2. **What is default?** Failure to meet contractual repayment obligations under a defined policy definition.
3. **What is NPA?** A banking asset whose repayment performance has deteriorated enough to meet the applicable non-performing classification rules.
4. **PD vs LGD vs EAD?** PD is likelihood of default, LGD is loss severity conditional on default, and EAD is exposure at default.
5. **What is underwriting?** The process of evaluating borrower risk and determining credit terms.
6. **What is a thin-file customer?** A customer with limited traditional credit-history information; alternative signals may help, subject to consent, legality and fairness.
7. **Why are repayment histories valuable?** They directly reflect prior payment behavior, a strong risk signal in many credit settings.
8. **What is credit utilization?** Balance relative to available revolving credit; high utilization can signal financial stress.
9. **What is reject inference?** The problem of estimating outcomes for applicants who were rejected and therefore have no observed repayment outcome.
10. **Why monitor vintage/cohort performance?** Risk can vary by origination period and macroeconomic conditions.
11. **What is PSI?** Population Stability Index, a common indicator for changes in feature distributions between reference and current populations.
12. **What is risk appetite?** The level/type of credit risk an institution is willing to accept.
13. **Why human oversight?** Automated risk scores can be wrong, biased, or incomplete and should operate within approved policy and review controls.
14. **Why protect borrower data?** Credit data is sensitive; collection and processing should follow applicable privacy, security and consent requirements.
15. **What would you do before bank deployment?** Validate data lineage, temporal performance, calibration, fairness, stability, explainability, security, compliance and operational impact.

## ATS-friendly resume bullets

- Built an end-to-end credit-risk ML pipeline using a real UCI credit-default dataset, including leakage checks, feature engineering, class-imbalance handling, model comparison and threshold analysis.
- Compared Logistic Regression, Random Forest and XGBoost with ROC-AUC, PR-AUC, precision, recall, F1 and confusion-matrix evaluation; added SHAP-based global and individual explanations.
- Deployed the trained scoring pipeline through FastAPI with Pydantic validation, Joblib model loading, pytest tests, Docker support and an optional Streamlit interface.

## GitHub metadata

**Repository:** `alternative-credit-scoring-ml`

**Description (<=160 chars):** End-to-end credit-risk ML system with feature engineering, XGBoost/Random Forest/Logistic Regression, SHAP, FastAPI, tests and Docker.

**Topics:** `credit-risk`, `credit-scoring`, `machine-learning`, `data-science`, `xgboost`, `random-forest`, `logistic-regression`, `shap`, `fastapi`, `scikit-learn`, `banking`, `risk-management`, `explainable-ai`, `mlops`, `python`
