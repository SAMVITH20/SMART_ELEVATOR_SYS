# Smart Elevator Scheduling System using Reinforcement Learning

## Project Overview

This project implements a Smart Elevator Scheduling System using Reinforcement Learning techniques such as Q-Learning and SARSA.

The system learns optimal elevator movement strategies to reduce passenger waiting time, improve movement efficiency, and minimize unnecessary elevator operations.

The project also integrates MLOps concepts including:

- Experiment Tracking
- Model Versioning
- CI/CD Automation
- API Deployment
- Monitoring Logs
- Docker Containerization

---

# SDG Mapping

This project supports the following United Nations Sustainable Development Goals (SDGs):

## SDG 9 — Industry, Innovation and Infrastructure

Efficient intelligent infrastructure using AI-driven elevator scheduling.

## SDG 11 — Sustainable Cities and Communities

Improves smart building automation and reduces waiting time.

## SDG 7 — Affordable and Clean Energy

Optimizes elevator movement to reduce unnecessary energy consumption.

---

# Problem Statement

Traditional elevator systems follow fixed scheduling strategies which may lead to:

- Increased passenger waiting time
- Inefficient movement
- Higher energy consumption
- Poor traffic handling during peak hours

This project uses Reinforcement Learning to enable elevators to learn efficient movement policies dynamically.

---

# Reinforcement Learning Approach

## Agent

Elevator

## Environment

Building floors and passenger requests

## States

Current elevator floor

## Actions

- Move Up
- Move Down
- Stay/Open Door

## Reward Function

- +100 for reaching target floor
- Negative reward for inefficient movement

---

# Algorithms Used

## 1. Q-Learning

A model-free reinforcement learning algorithm that learns optimal actions using Q-value updates.

## 2. SARSA

An on-policy reinforcement learning algorithm that updates values based on the action actually taken.

---

# Project Architecture

```text
Passenger Request
        ↓
Environment
        ↓
RL Agent (QLearning / SARSA)
        ↓
Action Selection
        ↓
Reward Calculation
        ↓
Q-table Update
```

---

# Project Structure

```text
SMART_ELEVATOR_SYS/

├── api/
├── configs/
├── experiments/
├── logs/
├── models/
├── plots/
├── sim/
├── .github/workflows/
│
├── Dockerfile
├── requirements.txt
├── README.md
├── train.py
└── LICENSE
```

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- FastAPI
- MLflow
- Docker
- GitHub Actions

---

# Installation Steps

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/SMART_ELEVATOR_SYS.git
```

## Move into Project Folder

```bash
cd SMART_ELEVATOR_SYS
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running Experiments

## Experiment 1 — QLearning Baseline

```bash
python train.py configs/qlearning_v1.yaml
```

---

## Experiment 2 — QLearning Optimized

```bash
python train.py configs/qlearning_v2.yaml
```

---

## Experiment 3 — SARSA

```bash
python train.py configs/sarsa_v1.yaml
```

---

# Experiment Tracking

The project stores experiment results inside:

```text
experiments/
```

Tracked metrics:

- Average reward
- Episodes
- Learning rate
- Epsilon
- Algorithm used

---

# MLflow Experiment Tracking

## Start MLflow UI

```bash
mlflow ui
```

Open browser:

```text
http://127.0.0.1:5000
```

MLflow tracks:

- Hyperparameters
- Metrics
- Reward performance
- Experiment comparisons

---

# FastAPI Deployment

## Run API

```bash
uvicorn api.app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

The API predicts elevator movement decisions.

---

# Monitoring and Logging

Prediction logs are stored in:

```text
logs/prediction_logs.csv
```

Tracked monitoring data:

- Timestamp
- Current floor
- Target floor
- Recommended action

---

# Docker Support

## Build Docker Image

```bash
docker build -t smart-elevator .
```

## Run Docker Container

```bash
docker run smart-elevator
```

---

# CI/CD Automation

GitHub Actions is used for CI/CD automation.

Whenever code is pushed:

- Dependencies are installed automatically
- Training pipeline executes automatically
- Workflow validates reproducibility

Workflow file:

```text
.github/workflows/train.yml
```

---

# Experiment Comparison

| Experiment | Algorithm | Learning Rate | Epsilon |
|---|---|---|---|
| Exp1 | QLearning | 0.1 | 0.2 |
| Exp2 | QLearning | 0.3 | 0.05 |
| Exp3 | SARSA | 0.1 | 0.1 |

---

# Challenges Faced

- Balancing exploration vs exploitation
- Avoiding infinite training loops
- Hyperparameter tuning
- CI/CD workflow setup
- Docker configuration issues

---

# Lessons Learned

- Importance of reproducibility in ML systems
- CI/CD automation for ML workflows
- Reinforcement Learning training stability
- Experiment tracking using MLflow
- Monitoring and deployment practices

---

# Future Improvements

- Deep Q-Networks (DQN)
- Multi-elevator coordination
- Real-time passenger traffic prediction
- Energy-aware scheduling
- Cloud deployment

---

# Screenshots

(Add screenshots here)

Recommended screenshots:

- MLflow UI
- FastAPI Swagger UI
- Reward Graphs
- GitHub Actions
- Docker Running
- GitHub Branches
- Pull Requests

---

# Author

SAMVITH

---

# License

MIT License