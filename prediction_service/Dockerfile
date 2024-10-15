# let's start with a slim python image for 3.12:
FROM python:3.11-slim

# set the working directory in the container
WORKDIR /app

# Install build-essentials, includinng gcc and other tools
RUN apt-get update && apt-get install -y build-essential

# copy the source code to the working directory
COPY . .

EXPOSE 8082

RUN pip install -r requirements.txt --no-cache-dir

# command to run on container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]