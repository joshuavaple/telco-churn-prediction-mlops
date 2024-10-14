run-dev:
	uvicorn main:app --host 0.0.0.0 --port 8081

run-dev-auth:
	uvicorn main_auth:app --host 0.0.0.0 --port 8082

build: 
	docker build --no-cache -t prediction-service .

run-live: build
	docker run -p 8082:8082 prediction-service