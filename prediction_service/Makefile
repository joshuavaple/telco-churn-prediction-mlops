run-dev:
	uvicorn main:app --host 0.0.0.0 --port 8082

build: 
	docker build --no-cache -t telco-churn-prediction-service .

run-live: build
	docker run -p 8082:8082 telco-churn-prediction-service