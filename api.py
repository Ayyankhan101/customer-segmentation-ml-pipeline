from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import json

app = FastAPI(title="Customer Segment Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    annual_spend: float
    visits_per_month: int
    items_per_order: int
    support_tickets: int

def get_processed_data():
    df = pd.read_csv("customer_segments.csv")
    df_clean = df.dropna(subset=['segment'])
    
    # Impute missing values for calculations
    imputer = KNNImputer(n_neighbors=5)
    cols_to_impute = ['annual_spend', 'visits_per_month', 'items_per_order', 'support_tickets']
    df_clean[cols_to_impute] = imputer.fit_transform(df_clean[cols_to_impute])
    
    return df_clean

@app.on_event("startup")
def load_model():
    global model, scaler, label_encoder
    
    df_clean = get_processed_data()
    X = df_clean[['annual_spend', 'visits_per_month', 'items_per_order', 'support_tickets']]
    
    le = LabelEncoder()
    y = le.fit_transform(df_clean['segment'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    label_encoder = le

@app.get("/data/summary")
def get_summary():
    df = get_processed_data()
    
    best_idx = 0 # Placeholder for evaluation results
    
    return {
        "total_customers": int(len(df)),
        "total_revenue": float(df['annual_spend'].sum()),
        "avg_spend": float(df['annual_spend'].mean()),
        "segment_counts": df['segment'].value_counts().to_dict(),
        "feature_stats": df.describe().to_dict()
    }

@app.get("/data/charts")
def get_chart_data():
    df = get_processed_data()
    
    # Segment distribution
    segment_dist = df['segment'].value_counts().to_dict()
    
    # Average spend by segment
    avg_spend_segment = df.groupby('segment')['annual_spend'].mean().to_dict()
    
    # Correlations
    corr = df[['annual_spend', 'visits_per_month', 'items_per_order', 'support_tickets']].corr().to_dict()
    
    return {
        "segments": segment_dist,
        "revenue_by_segment": avg_spend_segment,
        "correlations": corr
    }

@app.post("/predict")
def predict(data: CustomerData):
    features = np.array([[data.annual_spend, data.visits_per_month, 
                         data.items_per_order, data.support_tickets]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    predicted_segment = label_encoder.inverse_transform(prediction)[0]
    
    probabilities = model.predict_proba(features_scaled)[0]
    class_probs = {
        cls: float(prob) 
        for cls, prob in zip(label_encoder.classes_, probabilities)
    }
    
    return {
        "prediction": predicted_segment,
        "probabilities": class_probs
    }

@app.get("/")
def root():
    return {"message": "Customer Segment Prediction API", 
            "endpoints": ["/predict", "/data/summary", "/data/charts"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)