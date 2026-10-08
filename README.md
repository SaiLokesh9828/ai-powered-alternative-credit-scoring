# \# AI-Powered Alternative Credit Scoring System

# 

# An end-to-end machine learning system for predicting customer credit risk using behavioral and financial repayment patterns. The project combines feature engineering, preprocessing, model comparison, hyperparameter tuning, threshold optimization, and a FastAPI inference API.

# 

# > \*\*Portfolio Project:\*\* Developed as a machine learning project focused on credit-risk prediction and decision support. This system is not intended for real-world lending decisions without further validation, calibration, fairness analysis, regulatory review, and production monitoring.

# 

# \---

# 

# \## 🚀 Project Overview

# 

# Traditional credit scoring can depend heavily on established credit-history variables. This project explores an alternative approach by extracting additional behavioral indicators from customers' repayment and billing history.

# 

# The system analyzes:

# 

# \* Credit limit

# \* Customer age

# \* Historical payment delays

# \* Monthly billing amounts

# \* Monthly payment amounts

# \* Payment behavior

# \* Credit utilization

# \* Payment-to-bill ratios

# \* Recent changes in billing/payment behavior

# \* Six-month aggregate financial behavior

# 

# The final model is a \*\*tuned XGBoost classifier\*\* deployed through a \*\*FastAPI REST API\*\*.

# 

# \---

# 

# \## 🎯 Problem Statement

# 

# Financial institutions need reliable methods to identify customers who may have a higher probability of credit repayment difficulties.

# 

# The objective of this project is to build a machine learning classification system that:

# 

# 1\. Processes customer financial behavior.

# 2\. Engineers meaningful credit-risk features.

# 3\. Compares multiple machine learning models.

# 4\. Selects and tunes the strongest model.

# 5\. Optimizes the classification threshold.

# 6\. Evaluates performance on unseen test data.

# 7\. Exposes the trained model through an API.

# 

# \---

# 

# \## 🏗️ System Architecture

# 

# ```text

# Customer Financial Data

# &#x20;         │

# &#x20;         ▼

# &#x20;  Feature Engineering

# &#x20;         │

# &#x20;         ▼

# &#x20;  Data Preprocessing

# &#x20;  ├── Missing-value imputation

# &#x20;  └── Standard scaling

# &#x20;         │

# &#x20;         ▼

# &#x20;    Model Training

# &#x20;  ├── Logistic Regression

# &#x20;  ├── Random Forest

# &#x20;  └── XGBoost

# &#x20;         │

# &#x20;         ▼

# &#x20;  Hyperparameter Tuning

# &#x20;         │

# &#x20;         ▼

# &#x20;   Tuned XGBoost

# &#x20;         │

# &#x20;         ▼

# &#x20;  Threshold Selection

# &#x20;      Threshold = 0.60

# &#x20;         │

# &#x20;         ▼

# &#x20;    FastAPI Backend

# &#x20;         │

# &#x20;         ▼

# &#x20;   Credit Risk Result

# ```

# 

# \---

# 

# \## 📊 Dataset

# 

# The project uses customer credit-card behavioral and repayment information.

# 

# The original dataset contains:

# 

# \* \*\*30,000 customer records\*\*

# \* \*\*41 columns\*\*

# \* Financial, demographic, billing, and repayment variables

# 

# The final model uses \*\*36 engineered/input features\*\*.

# 

# Raw datasets are intentionally excluded from this repository.

# 

# \---

# 

# \## 🧠 Feature Engineering

# 

# Several behavioral features were created from the customer's six-month payment and billing history.

# 

# \### Payment behavior

# 

# \* `max\_payment\_delay`

# \* `avg\_payment\_delay`

# \* `months\_with\_delay`

# \* `severe\_delay\_months`

# 

# \### Billing behavior

# 

# \* `avg\_bill\_amount`

# \* `bill\_volatility`

# \* `total\_billed\_6m`

# \* `recent\_bill\_change`

# 

# \### Payment behavior

# 

# \* `avg\_payment\_amount`

# \* `payment\_volatility`

# \* `total\_paid\_6m`

# \* `recent\_payment\_change`

# 

# \### Credit utilization

# 

# \* `avg\_utilization`

# \* `max\_utilization`

# 

# \### Payment ratios

# 

# \* `avg\_payment\_ratio`

# \* `total\_payment\_ratio\_6m`

# 

# These features attempt to capture not only the customer's current financial state but also repayment patterns over time.

# 

# \---

# 

# \## ⚙️ Preprocessing

# 

# The final model is stored as a complete scikit-learn Pipeline.

# 

# \### Numerical preprocessing

# 

# ```text

# Missing Values

# &#x20;     ↓

# Median Imputation

# &#x20;     ↓

# StandardScaler

# &#x20;     ↓

# XGBoost

# ```

# 

# Using a single pipeline ensures that the same preprocessing applied during training is automatically applied during inference.

# 

# \---

# 

# \# 🤖 Models Evaluated

# 

# Three classification models were compared.

# 

# | Model               |    ROC-AUC |     PR-AUC |

# | ------------------- | ---------: | ---------: |

# | Logistic Regression |     0.7613 |     0.5199 |

# | Random Forest       |     0.7812 |     0.5465 |

# | XGBoost             |     0.7830 |     0.5551 |

# | \*\*Tuned XGBoost\*\*   | \*\*0.7859\*\* | \*\*0.5584\*\* |

# 

# The tuned XGBoost model achieved the strongest validation performance.

# 

# \---

# 

# \# 🔧 Hyperparameter Tuning

# 

# The final XGBoost configuration:

# 

# | Parameter            | Value |

# | -------------------- | ----: |

# | Learning Rate        |  0.03 |

# | Max Depth            |     4 |

# | Number of Estimators |   250 |

# 

# Validation performance after tuning:

# 

# \* \*\*ROC-AUC:\*\* 0.7859

# \* \*\*PR-AUC:\*\* 0.5584

# 

# \---

# 

# \# 🎚️ Threshold Optimization

# 

# Instead of relying on the default classification threshold of 0.50, multiple thresholds were evaluated.

# 

# The final threshold was selected as:

# 

# ```text

# Threshold = 0.60

# ```

# 

# This provided a better precision/recall balance for the project's objective.

# 

# The threshold should not be considered a production lending policy. In a real financial system, threshold selection would need to incorporate expected loss, business costs, approval rates, calibration, regulatory requirements, and fairness constraints.

# 

# \---

# 

# \# 📈 Final Test Performance

# 

# The final tuned XGBoost model was evaluated on unseen test data.

# 

# | Metric                   | Test Result |

# | ------------------------ | ----------: |

# | ROC-AUC                  |  \*\*0.7811\*\* |

# | PR-AUC                   |  \*\*0.5613\*\* |

# | Precision                |  \*\*54.31%\*\* |

# | Recall                   |  \*\*54.56%\*\* |

# | F1 Score                 |  \*\*54.44%\*\* |

# | Classification Threshold |    \*\*0.60\*\* |

# 

# \### Confusion Matrix

# 

# ```text

# &#x20;                Predicted

# &#x20;                0       1

# 

# Actual 0       4064     609

# Actual 1        603     724

# ```

# 

# Where:

# 

# \* True Negatives = 4064

# \* False Positives = 609

# \* False Negatives = 603

# \* True Positives = 724

# 

# \---

# 

# \## 📊 Evaluation Visualizations

# 

# \### ROC Curve

# 

# !\[ROC Curve](reports/figures/test\_roc\_curve.png)

# 

# \### Precision-Recall Curve

# 

# !\[Precision-Recall Curve](reports/figures/test\_pr\_curve.png)

# 

# \### Confusion Matrix

# 

# !\[Confusion Matrix](reports/figures/test\_confusion\_matrix.png)

# 

# \---

# 

# \# 🔌 FastAPI

# 

# The trained model is exposed through a REST API using FastAPI.

# 

# \## Available Endpoints

# 

# \### Health Check

# 

# ```text

# GET /health

# ```

# 

# Example response:

# 

# ```json

# {

# &#x20; "status": "healthy",

# &#x20; "model": "Tuned XGBoost",

# &#x20; "threshold": 0.6

# }

# ```

# 

# \### Credit Risk Prediction

# 

# ```text

# POST /predict

# ```

# 

# The endpoint accepts the 36 features required by the trained model.

# 

# Example response:

# 

# ```json

# {

# &#x20; "risk\_probability": 0.2227,

# &#x20; "prediction": 0,

# &#x20; "risk\_category": "Low Risk",

# &#x20; "threshold

# ```



