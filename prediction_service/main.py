from fastapi import FastAPI
from loguru import logger
import joblib
import json
import pandas as pd


logger.debug("Loading classification model and other artifacts...")
feat_cols = json.load(open('./models/feat_cols.json'))
logger.debug(f"Feature columns from training process: {feat_cols}")
scaler = joblib.load('./models/scaler.pkl')
model = joblib.load('./models/xgboost_clf_model.pkl')
logger.debug("Model loaded successfully!")

app = FastAPI()

@app.post("/predict")
def predict(data:dict):
    logger.debug(f"Received data: {data}")
    data = pd.DataFrame([data])
    data = data[feat_cols]
    data = scaler.transform(data)
    prediction = model.predict(data).item() 
    # need to get to the value to avoid error below
    # ValueError: [TypeError('cannot convert dictionary update sequence element #0 to a sequence'), TypeError('vars() argument must have __dict__ attribute')]
    logger.debug(f"Prediction: {prediction}")
    return {"prediction": prediction}