# LightningTune

A PyTorch Lightning project for efficient model training and hyperparameter optimization (HPO) using Optuna. This project supports live custom logging during HPO runs, with containerized deployments to ensure a streamlined and reproducible workflow. You can monitor experiments using **Weights & Biases** (wandb).

## Overview

This project leverages **PyTorch Lightning** for streamlined model training, **Optuna** for hyperparameter optimization, and **Docker** for containerization. Containerized HPO enables you to efficiently run experiments in isolated environments, ensuring consistency and scalability across multiple workers. Real-time logging with **Weights & Biases** (wandb) provides valuable insights during HPO runs.

## Key Features

- **Containerized HPO**: Easily run hyperparameter tuning experiments in Docker, simplifying deployment and scaling.
- **Real-Time Custom Logging**: Track metrics and model performance live during each tuning run with wandb.
- **Reproducibility**: Configured for deterministic results, leveraging environment variables for consistency.

## Setup Instructions

### 1. Create a `.env` file

Before running the project, create a `.env` file in the root directory to store necessary environment variables. At a minimum, include the following variables:

```plaintext
WANDB_API_KEY=<your_wandb_api_key>
CUBLAS_WORKSPACE_CONFIG=:4096:8  # For reproducibility (optional)
```

### 2. Run with Docker Compose
Use Docker Compose to build and run the project in containers. This will set up the Optuna database, connect the containers, and start the HPO worker(s).

To start the containers, run:
```bash
docker-compose up --build optuna-worker
```

This command builds and launches the HPO worker containers, which will communicate with the Optuna database and log metrics to wandb in real time.

## Architecture

- **Optuna**: Handles hyperparameter optimization across multiple workers using a PostgreSQL database for shared study storage.
- **PyTorch Lightning**: Simplifies model training with an easy-to-use framework that integrates seamlessly with Optuna for HPO.
- **Weights & Biases**: Provides real-time experiment tracking with custom logging during HPO runs.
- **Docker Compose**: Manages service dependencies and scales multiple Optuna workers for parallelized tuning.

## Dependencies

The Docker setup automatically handles dependencies. For local installations, refer to the `requirements.txt` file or Dockerfile for required packages.

## Example Commands

To adjust the number of trials, limit training batches, or customize other parameters, update the `command` section in the Docker Compose file.

Example:

```yaml
command: python optimize.py -n 10 -ltb 0.2 -lvb 0.2 -pn MyProject -sn MyStudy
```
This command sets 10 trials with 20% of training and validation batches and logs results under the project MyProject with study name MyStudy.
