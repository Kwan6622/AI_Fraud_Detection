import pandas as pd 
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model():
    df = pd.read_csv("data/creditcard.csv")

    x = df.drop("Class", axis=1)
    y = df["Class"]
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    x_train = x_train_raw.copy()
    x_test = x_test_raw.copy()

    x_train[["Amount","Time"]] = scaler.fit_transform(x_train[["Amount","Time"]])
    x_test[["Amount","Time"]] = scaler.transform(x_test[["Amount","Time"]])

    train_data = pd.concat([x_train, y_train], axis=1)
    fraud_indices = train_data[train_data.Class == 1].index
    normal_indices = train_data[train_data.Class == 0].index

    np.random.seed(42)
    random_normal_indices = np.random.choice(normal_indices, len(fraud_indices), replace=False)
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    under_sample_train = train_data.loc[under_sample_indices]

    x_train_final = under_sample_train.drop("Class", axis=1)
    y_train_final = under_sample_train["Class"]
    #Training using RandomForest 
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train_final, y_train_final)

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

if __name__ == "__main__":
    train_model()
