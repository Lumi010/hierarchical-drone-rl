# Autonomous Drone Navigation in Dynamic Environments

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-green)
![PyBullet](https://img.shields.io/badge/Physics-PyBullet-orange)
![Status](https://img.shields.io/badge/Status-Work_in_Progress_(FYP--I_Completed)-yellow)

> **Note:** This repository contains the Phase 1 (FYP-I) deliverables for my Computer Systems Engineering Final Year Project. It is actively being developed as we transition into FYP-II.

---

## 🚁 Project Overview

This project implements a **Hierarchical Reinforcement Learning** framework for autonomous quadrotor navigation in dynamic, unpredictable wind environments. 

The core engineering contribution of this project is solving the "Sim-to-Real" gap by training the drone in a continuously changing aerodynamic environment, proving that a lightweight RL architecture can achieve robust disturbance rejection.

---

## 🧠 Core Architecture: The PPO-PID Bridge

Instead of relying on brittle end-to-end learning (where an AI attempts to learn Newtonian physics from scratch), this system splits the intelligence into two distinct layers:

1. **High-Level Brain (PPO Neural Network):** 
   * Observes a 16-Dimensional state vector (including target error, orientation, and crucial **wind force vectors**).
   * Outputs a continuous 3D velocity intention (e.g., "move forward at 0.35 m/s").

2. **Low-Level Muscles (Classical PID Controller):** 
   * Uses a custom **Kinematic Look-ahead Bridge** (0.25s projection) to translate the AI's velocity commands into a smooth, intermediate target coordinate.
   * Calculates deterministic motor RPMs to stabilize flight and physically execute the AI's intention.

This hierarchical approach drastically accelerated training convergence to just **200,000 steps** while guaranteeing physical flight safety.

---

## 🌪️ Disturbance Modeling & Evaluation

The environment features a continuous sinusoidal wind disturbance model injected directly into the PyBullet rigid-body physics engine. The agent was evaluated across three structured modes to prove generalization:

* **🟢 No Wind:** Baseline navigation check (100% Success).
* **🟡 Fixed Wind:** Deterministic, repeatable sinusoidal disturbance (100% Success).
* **🔴 Random Wind:** Highly stochastic storms where wind phase and amplitude are randomized upon every reset. The agent achieved a **90% success rate** against conditions it had never seen during training, proving true physical generalization over path-memorization.

---

## 🚀 Future Work Roadmap (FYP-II)

The robust aerodynamic foundation built in FYP-I will be expanded next semester:

- [ ] **Dynamic Obstacle Avoidance:** Expanding the observation space to include 3D raycasting sensors.
- [ ] **Moving Targets:** Training the agent to track non-stationary goals.
- [ ] **Aviation-Grade Wind:** Upgrading the sinusoidal wind to the DOD-standard **Dryden Turbulence Model** for high-fidelity realism.
- [ ] **Algorithm Benchmarking:** Comparing PPO's on-policy performance against off-policy algorithms like Soft Actor-Critic (SAC).
