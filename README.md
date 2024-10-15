# telco-churn-prediction-mlops
A simple project to practice MLOps

## 1. DEV setup
- cd to the project root
- Creata a conda environment by `conda env create -f environment_dev.yaml`
- Activate the environment above by `conda activate telco-churn-prediction`

## 2. Building a simple classifier
- You can recreate the model and other artifacts by rerunning the 2 notebooks: `eda_clean_featurize.ipynb` and `train_eval.ipynb`.
- Running them will replace pretrained artifacts from the `prediction_service/models` folder
- However, they are not the focus of this project. You can use pretrained artifacts above.

## 3. Serving the model locally outside a container
- cd to prediction_service/
- Run the command `make run-dev` to start a FastAPI service.

## 4. Build and run model prediction service locally in a container
- cd to prediction_service/
- Run the command `make run-live` to build the service image (containing the pretrained artifacts) and run it in a local container.

## 5. How to use the inference endpoints
- There are 2 endpoints when the service is running: `/predict` (without API key) and `/predict-auth` (with API key)
- They accept inference data in the form of a dictionary, with keys being the columns that the model was trained.
- You can find examples in the `client/` folder. They use a subset of the original dataset `df_inference_raw.csv` (held out from the train-eval process) to simulate client requests. The client sends raw data to the service. The service will do the necessary preprocessing and featurizing (persisted from the training process) before invoking the model and returns the prediction.
- An example request body is `{"AccountWeeks": 80, "ContractRenewal": 1, "DataPlan": 0, "DataUsage": 0.0, "CustServCalls": 0, "DayMins": 194.8, "DayCalls": 116, "MonthlyCharge": 51.0, "OverageFee": 10.5, "RoamMins": 12.8}`
- The example response is `{'prediction': 0}`

### Note
- the api keys stored in .env files are for demonstration purposes, they are not api keys to any other services.
- in practice, they are not committed to the repos, but managed in suitable secret management services from the server and client side.