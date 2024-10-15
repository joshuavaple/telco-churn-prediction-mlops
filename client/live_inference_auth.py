import requests
import pandas as pd
import json
from dotenv import load_dotenv
import os

load_dotenv('./client_api_key.env', override=True)
API_KEY = os.getenv('API_KEY')
headers = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
}

df_inference_raw = pd.read_csv('../data/inference/df_inference_raw.csv')
df_inference_raw = df_inference_raw.drop(columns=['Churn'])

def run_live_inference():
    url = "http://localhost:8082/predict-auth"
    sample_row = df_inference_raw.sample(1)
    sample_row_dict = sample_row.to_dict(orient='records')[0]
    data_json = json.dumps(sample_row_dict)
    response = requests.post(url, data=data_json, headers=headers)
    return response.json()

if __name__ == '__main__':
    print(run_live_inference())