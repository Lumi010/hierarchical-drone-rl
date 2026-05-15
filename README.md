# Autonomous Drone Navigation in Dynamic Environments

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-green)
![PyBullet](https://img.shields.io/badge/Physics-PyBullet-orange)
![Status](https://img.shields.io/badge/Status-Work_in_Progress_(FYP--I_Completed)-yellow)

> **Note:** This repository contains the Phase 1 (FYP-I) deliverables for my Computer Systems Engineering Final Year Project. It is actively being developed as we transition into FYP-II.

---

## 🚁 Project Overview
This project implements a **Hierarchical Reinforcement Learning** framework for autonomous quadrotor navigation in dynamic, unpredictable wind environments. The core engineering contribution is solving the "Sim-to-Real" gap by training the drone in a continuously changing aerodynamic environment.

## 🧠 Core Architecture: The PPO-PID Bridge
Instead of relying on brittle end-to-end learning, this system splits the intelligence into two distinct layers:
1. **High-Level Brain (PPO Neural Network):** Observes a 16-Dimensional state vector (including target error and wind forces) and outputs a continuous 3D velocity intention.
2. **Low-Level Muscles (Classical PID Controller):** Uses a custom **Kinematic Look-ahead Bridge** (0.25s projection) to smoothly translate the AI's velocity commands into deterministic motor RPMs to stabilize flight.

## 🌪️ Evaluation Results
The environment features a continuous sinusoidal wind disturbance model injected directly into the PyBullet rigid-body physics engine. 
* **🟢 No Wind:** Baseline navigation check (**100% Success**).
* **🟡 Fixed Wind:** Deterministic, repeatable sinusoidal disturbance (**100% Success**).
* **🔴 Random Wind:** Highly stochastic storms where wind phase and amplitude are randomized. The agent achieved a **90% success rate** against conditions it had never seen during training, proving true physical generalization.

---

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/Lumi010/hierarchical-drone-rl.git
cd hierarchical-drone-rl
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install numpy pybullet gymnasium stable-baselines3
```

## 🎮 Usage: Testing the Pre-trained Agent

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
*(Add the `--no-gui` flag to run the tests headlessly without the PyBullet viewer).*

## 🏋️ Usage: Training a New Agent

To train a new PPO policy from scratch:
```bash
python train/train_ppo.py --timesteps 200000 --random-wind
```
*   Use `--resume` to continue training from an existing model in `models/drone_model.zip`.
*   Logs will be saved to `logs/ppo_monitor.csv`.

---

## 🚀 Future Work Roadmap (FYP-II)
The foundation built in FYP-I will be expanded next semester:
- [ ] **Dynamic Obstacle Avoidance:** Expanding the observation space to include 3D raycasting sensors.
- [ ] **Moving Targets:** Training the agent to track non-stationary goals.
- [ ] **Aviation-Grade Wind:** Upgrading the sinusoidal wind to the DOD-standard **Dryden Turbulence Model** for high-fidelity realism.
- [ ] **Algorithm Benchmarking:** Comparing PPO's on-policy performance against off-policy algorithms like Soft Actor-Critic (SAC).
