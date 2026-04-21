# 🚨 End-to-End AI Fraud Detection System

A production-oriented **real-time credit card fraud detection system** that demonstrates the full machine learning lifecycle — from data preprocessing to API deployment and database logging.

---

## 📌 Overview

This project is built with a **"Safety-First" philosophy**, prioritizing fraud detection (**high recall**) to minimize financial loss.

It showcases strong engineering practices expected in real-world AI systems:
- Preventing **data leakage**
- Handling **extreme class imbalance**
- Deploying a **low-latency API**
- Logging predictions into a **SQL database**

---

## ⚙️ Key Features

### 🧠 Production-Ready AI
- Optimized for **high Recall (Fraud class)**
- Designed to minimize **false negatives** (missed fraud)

### 🧹 Robust Preprocessing
- Train/Test split **before scaling**
- Eliminates **data leakage** (common junior mistake)

### ⚡ Real-Time Inference API
- Built with **FastAPI**
- Supports **high concurrency** and **low latency**

### 🗄️ Automated Data Logging
- Uses **SQLAlchemy (ORM)**
- Stores:
  - Transaction data
  - Prediction results
  - Fraud probability

### 🚀 Performance Optimization
- Uses **FastAPI BackgroundTasks**
- Database writes are **asynchronous**
- Keeps API response time **fast**

---

## 🧰 Tech Stack

### Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Backend & API
- FastAPI
- Uvicorn
- Pydantic

### Database
- SQLAlchemy (ORM)
- SQLite / PostgreSQL

---

## 📊 Model Performance

Trained on the **Kaggle Credit Card Fraud Detection dataset**  
(Highly imbalanced: ~99.8% legitimate transactions)

| Metric     | Class 0 (Legit) | Class 1 (Fraud) |
|------------|-----------------|-----------------|
| Precision  | 1.00            | 0.04            |
| Recall     | 0.97            | 0.93            |
| F1-Score   | 0.98            | 0.08            |

### 🔍 Analysis
- ✅ **93% Recall** → Most fraudulent transactions are detected  
- ⚠️ **Low Precision (0.04)** → More false positives  
- 🎯 Trade-off is intentional: better to **flag suspicious transactions** than miss fraud  

---

## 📁 Project Structure


fraud_ai/
├── data/ # Raw dataset (creditcard.csv)
├── models/ # Serialized model and scaler (.pkl)
├── train.py # Training, evaluation, export
├── app.py # FastAPI server + SQL logging
├── requirements.txt # Dependencies
└── fraud_history.db # Auto-generated database


---

## 🚀 Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
2. Train the Model
python train.py
3. Run the API
python -m uvicorn app:app --reload
4. Open Interactive Docs

http://127.0.0.1:8000/docs

🔌 API Specification
Endpoint
POST /predict
📥 Request Body
{
  "data": [0.0, -1.35, 1.1, -1.2, ..., 149.62]
}
📤 Response
{
  "is_fraud": 1,
  "fraud_probability": 0.9325,
  "verdict": "High Risk",
  "status": "Logged to Database"
}