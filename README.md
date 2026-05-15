# Autonomous Drone Navigation in Dynamic Environments (RL)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-green)
![PyBullet](https://img.shields.io/badge/Physics-PyBullet-orange)
![Status](https://img.shields.io/badge/Status-Work_in_Progress_(FYP--I_Completed)-yellow)

> **Note:** This repository contains the Phase 1 (FYP-I) deliverables for my Computer Systems Engineering Final Year Project. It is actively being developed as we transition into FYP-II.

## Overview

This project implements a **Hierarchical Reinforcement Learning** framework for autonomous quadrotor navigation in dynamic, unpredictable wind environments. 

Instead of relying on brittle end-to-end learning, this system splits the intelligence into two layers:
1. **High-Level Brain (PPO Neural Network):** Observes a 16D state vector (including target error and wind forces) and outputs continuous 3D velocity intentions.
2. **Low-Level Muscles (Classical PID):** Uses a custom **Kinematic Look-ahead Bridge** (0.25s projection) to smoothly translate the AI's velocity commands into deterministic motor RPMs to stabilize flight.

## Key Features (FYP-I)

* **Hierarchical PPO-PID Architecture:** Drastically accelerates training convergence and ensures physical flight safety.
* **Continuous Wind Disturbance Modeling:** Injects time-varying sinusoidal aerodynamic forces (`p.applyExternalForce`) directly into the PyBullet rigid-body physics engine.
* **Multi-Objective Reward Shaping:** A carefully balanced reward function utilizing a "ticking clock" distance penalty to prevent hover-farming and ensure efficient target-reaching.
* **Structured Evaluation Modes:** Built-in testing for deterministic (Fixed Wind) and stochastic (Random Wind) scenarios to prove policy generalization.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lumi010/autonomous-drone-rl.git
cd autonomous-drone-rl
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install numpy pybullet gymnasium stable-baselines3
```

## Usage: Testing the Pre-trained Agent

You can test the pre-trained PPO agent using `main.py`. The environment supports three structured evaluation modes:

**1. No Wind (Baseline Test):**
```bash
python main.py --use-ppo --no-wind
```

**2. Fixed Wind (Deterministic Disturbance):**
Applies a constant-phase sinusoidal wind. The storm is exactly the same every time.
```bash
python main.py --use-ppo
```

**3. Random Wind (Stochastic Disturbance):**
Randomizes wind phase and amplitude upon every reset. Proves the agent generalizes to unseen physics.
```bash
python main.py --use-ppo --random-wind
```

*(Add the `--no-gui` flag if you want to run the tests headlessly without the PyBullet viewer).*

## Usage: Training a New Agent

To train a new PPO policy from scratch:

```bash
python train/train_ppo.py --timesteps 200000 --random-wind
```
*   Use `--resume` to continue training from an existing model in `models/drone_model.zip`.
*   Logs will be saved to `logs/ppo_monitor.csv`.

## Future Work Roadmap (FYP-II)

The foundation built in FYP-I will be expanded next semester:
- [ ] **Dynamic Obstacle Avoidance:** Expanding the observation space to include 3D raycasting sensors.
- [ ] **Moving Targets:** Training the agent to track non-stationary goals.
- [ ] **Aviation-Grade Wind:** Upgrading the sinusoidal wind to the DOD-standard **Dryden Turbulence Model** for high-fidelity aerodynamic realism.
- [ ] **Algorithm Benchmarking:** Comparing PPO's on-policy performance against off-policy algorithms like Soft Actor-Critic (SAC).

## License
MIT License
