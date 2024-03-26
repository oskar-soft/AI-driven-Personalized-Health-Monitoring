# AI-driven-Personalized-Health-Monitoring
开发一个AI驱动的个性化健康监测系统，利用可穿戴设备和机器学习来检测早期疾病迹象。
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Simulate data collection from a wearable device
# This would be replaced by actual data collection in a real application
def collect_data():
    # Simulating health metrics: heart rate, body temperature, steps, sleep hours
    data = {
        'heart_rate': np.random.randint(60, 100, 1000),
        'body_temperature': np.random.normal(36.5, 0.5, 1000),
        'steps': np.random.randint(0, 20000, 1000),
        'sleep_hours': np.random.randint(4, 10, 1000),
        'health_status': np.random.choice(['Healthy', 'Unhealthy'], 1000)
    }
    df = pd.DataFrame(data)
    return df

# Data preprocessing
def preprocess_data(df):
    # Convert categorical data to numerical (Healthy=0, Unhealthy=1)
    df['health_status'] = df['health_status'].map({'Healthy': 0, 'Unhealthy': 1})
    return df

# Train a simple machine learning model
def train_model(df):
    X = df.drop('health_status', axis=1)
    y = df['health_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")
    
    return model

# Anomaly detection (simulated as a simple prediction for demo)
def detect_anomalies(model, new_data):
    prediction = model.predict([new_data])
    if prediction == 1:
        print("Anomaly detected: Potential health issue")
    else:
        print("No anomaly detected: Healthy")

# Main flow
if __name__ == "__main__":
    df = collect_data()
    df = preprocess_data(df)
    model = train_model(df)
    
    # Simulating new data from a wearable device
    new_data = [75, 36.7, 5000, 7]  # Example: heart rate, body temp, steps, sleep hours
    detect_anomalies(model, new_data)
