import joblib
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# --- CONFIG DATABASE (Ở đây dùng SQLite cho nhanh, đổi sang Postgres nếu muốn) ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./fraud_history.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Định nghĩa bảng trong SQL
class FraudLog(Base):
    __tablename__ = "fraud_logs"
    id = Column(Integer, primary_key=True, index=True)
    transaction_amount = Column(Float)
    prediction = Column(Integer)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Tạo bảng
Base.metadata.create_all(bind=engine)

# --- APP & AI MODEL ---
app = FastAPI()
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class Transaction(BaseModel):
    data: list # Danh sách 30 giá trị [Time, V1...V28, Amount]

# Hàm lưu log vào DB (Chạy ngầm để API phản hồi nhanh hơn)
def log_to_db(amount: float, pred: int, conf: float):
    db = SessionLocal()
    new_log = FraudLog(transaction_amount=amount, prediction=pred, confidence=conf)
    db.add(new_log)
    db.commit()
    db.close()

@app.post("/predict")
async def predict(transaction: Transaction, background_tasks: BackgroundTasks):
    # 1. Tiền xử lý
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = pd.DataFrame([transaction.data], columns=cols)
    
    # 2. Scale & Predict
    input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    # 3. Lưu vào DB (sử dụng BackgroundTask để không làm chậm API)
    amount_val = transaction.data[-1] # Lấy giá trị Amount từ input
    background_tasks.add_task(log_to_db, amount_val, prediction, probability)

    return {
        "is_fraud": prediction,
        "confidence": round(probability, 4),
        "status": "Logged to Database"
    }