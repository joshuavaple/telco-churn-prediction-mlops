from fastapi import FastAPI
from loguru import logger
import joblib
import json
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv


logger.debug("Loading classification model and other artifacts...")
feat_cols = json.load(open('./models/feat_cols.json'))
logger.debug(f"Feature columns from training process: {feat_cols}")
scaler = joblib.load('./models/scaler.pkl')
model = joblib.load('./models/xgboost_clf_model.pkl')
logger.debug("Model loaded successfully!")

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key")
# API_KEY = os.environ.get("API_KEY")

# import API_KEYS from environment:
load_dotenv('./api_keys.env', override=True)
API_KEYS = os.getenv('API_KEY').split(',')


def get_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key

@app.post("/predict_auth")
def predict(data: dict, api_key: str = Depends(get_api_key)):
    logger.debug(f"Received data: {data}")
    data = pd.DataFrame([data])
    data = data[feat_cols]
    data = scaler.transform(data)
    prediction = model.predict(data).item()
    logger.debug(f"Prediction: {prediction}")
    return {"prediction": prediction}