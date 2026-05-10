# Smart Elevator Scheduling System using Q-Learning

## Problem Statement

This project implements a Smart Elevator Scheduling System using Reinforcement Learning (Q-Learning).

The objective is to reduce passenger waiting time and optimize elevator movement.

---

# SDG Mapping

This project supports:

- SDG 9 — Industry, Innovation and Infrastructure
- SDG 11 — Sustainable Cities and Communities
- SDG 7 — Affordable and Clean Energy

---

# Project Structure

smart-elevator-rl/

├── sim/
├── configs/
├── experiments/
├── plots/
├── train.py

---

# Reinforcement Learning Components

## State

Current elevator floor.

## Actions

- Move Up
- Move Down
- Stay/Open Door

## Reward Function

- +100 for reaching target floor
- Negative reward for inefficient movement

---

# Installation

## Create Virtual Environment

python -m venv venv

## Activate

### Windows

venv\Scripts\activate

### Linux/Mac

source venv/bin/activate

## Install Requirements

pip install -r requirements.txt

---

# Run Experiment 1

python train.py configs/qlearning_v1.yaml

---

# Run Experiment 2

python train.py configs/qlearning_v2.yaml

---

# Experiments

## Experiment 1

- Learning Rate: 0.1
- Epsilon: 0.2
- Higher exploration

## Experiment 2

- Learning Rate: 0.3
- Epsilon: 0.05
- Lower exploration

---

# Experiment Tracking

Stored in:

experiments/summary.csv

Tracked Parameters:

- run-id
- episodes
- average reward
- learning rate
- epsilon

---

# Reproducibility

To reproduce the same experiment:

python train.py configs/qlearning_v1.yaml

or

python train.py configs/qlearning_v2.yaml

---

# Monitoring Plan

If deployed in a real-world building, the following metrics would be monitored:

- Average waiting time
- Elevator idle time
- Energy consumption
- Queue length
- Passenger congestion
- Elevator efficiency

---

# Versioning

Git tags are used for experiment tracking.

Examples:

- exp-qlearning-1
- exp-qlearning-2