End-to-End AI Fraud Detection SystemThis project implements a comprehensive real-time credit card fraud detection system. It covers the entire machine learning lifecycle: from handling extreme data imbalance and preventing data leakage during preprocessing to deploying a live API with SQL logging.

Key Features

Production-Ready AI: Built with a "Safety-First" approach, prioritizing Recall to minimize financial loss from undetected fraud.

Robust Preprocessing: Implements a strict pipeline that splits data before scaling, preventing Data Leakage—a common mistake in junior-level projects.

Real-Time Inference: Powered by FastAPI, providing high-concurrency and low-latency predictions.

Automated Data Logging: Integrates SQLAlchemy (ORM) to asynchronously log every transaction and prediction into a SQL database (SQLite/PostgreSQL).

Performance Optimization: Uses FastAPI BackgroundTasks for database operations to ensure the API response time remains ultra-fast.

Tech Stack

Machine Learning: Python, Scikit-learn, Pandas, NumPy, Joblib.
Backend & API: FastAPI, Uvicorn, Pydantic.
Database: SQLAlchemy (ORM), PostgreSQL.

Model Performance

The model was trained on the Kaggle Credit Card Fraud Detection dataset. Given the 99.8% class imbalance, the system focuses on the Recall metric for the minority class (Fraud).

Metric      Class 0 (Legit) Class 1 (Fraud)
Precision       1.00            0.04
Recall          0.97            0.93
F1-Score        0.98            0.08

Analysis: A 93% Recall means the system successfully flags nearly all fraudulent attempts. The lower precision is a calculated trade-off in fraud detection systems to ensure maximum security coverage.

Project StructurePlaintextfraud_ai/
├── data/               # Raw dataset (creditcard.csv)
├── models/             # Serialized model and scaler (.pkl)
├── train.py            # Model training, evaluation, and export script
├── app.py              # FastAPI server with SQL integration
├── requirements.txt    # Project dependencies
└── fraud_history.db    # Auto-generated SQL database for logs

Installation & Usage

1. Install Dependencies:

    pip install -r requirements.txt

2. Train the Model:

    python train.py

3. Launch the API:

    python -m uvicorn app:app --reload

4. Interactive Documentation:

    Navigate to http://127.0.0.1:8000/docs to test the API via the built-in Swagger UI.

API SpecificationEndpoint: 

POST /predictSample 

Request Body:

{
  "data": [0.0, -1.35, 1.1, -1.2, ..., 149.62]
}

Sample Response:

{
  "is_fraud": 1,
  "fraud_probability": 0.9325,
  "verdict": "High Risk",
  "status": "Logged to Database"
}