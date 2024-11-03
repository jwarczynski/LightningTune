# Dockerfile for Optuna worker
FROM python:3.12-slim
LABEL authors="jwarczynski"

# Set environment variables
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Install system dependencies
RUN apt-get update && apt-get install -y libpq-dev python3-dev && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch cpu-only version
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file
COPY ./requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy your code into the container
COPY ./src/download_dataset.py .

RUN python download_dataset.py

COPY ./src/optimize.py ./src/models.py ./

# Default command to start the training script
CMD ["python", "optimize.py"]
