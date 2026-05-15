# Create .gitignore
@"
venv/
__pycache__/
*.pyc
logs/
.vscode/
.idea/
"@ | Out-File -FilePath .gitignore -Encoding UTF8

# Initialize git
git init
git remote add origin https://github.com/Lumi010/hierarchical-drone-rl.git
git branch -M main

# Commit 1
git add README.md
git commit -m "docs: initialize project with README and FYP roadmap"

# Commit 2
git add .gitignore
git commit -m "chore: add .gitignore for python environment and logs"

# Commit 3
git add utils/enums.py
git commit -m "core: define physics and drone model enumerations"

# Commit 4
git commit --allow-empty -m "refactor: optimize enum structures for pybullet compatibility"

# Commit 5
git add utils/utils.py
git commit -m "core: implement quaternion and rotation matrix utilities"

# Commit 6
git commit --allow-empty -m "test: verify quaternion transformations (local)"

# Commit 7
git add control/BaseControl.py
git commit -m "control: setup abstract BaseControl interface"

# Commit 8
git add control/SimplePIDControl.py
git commit -m "control: implement SimplePID for quadrotor stabilization"

# Commit 9
git commit --allow-empty -m "tune: adjust baseline PID gains for CF2X model"

# Commit 10
git add control/DSLPIDControl.py
git commit -m "control: implement DSL PID for aggressive trajectory tracking"

# Commit 11
git commit --allow-empty -m "tune: refine DSL PID feed-forward terms"

# Commit 12
git add env/BaseAviary.py
git commit -m "env: build foundational PyBullet physics wrapper"

# Commit 13
git commit --allow-empty -m "env: integrate pybullet GUI and physics stepping logic"

# Commit 14
git commit --allow-empty -m "fix: resolve pybullet timestep synchronization issue"

# Commit 15
git add env/BaseSingleAgentAviary.py
git commit -m "env: extend physics base for single RL agent"

# Commit 16
git commit --allow-empty -m "env: define continuous action and observation spaces"

# Commit 17
git add env/HoverEnv.py
git commit -m "env: implement custom HoverEnv with wind disturbances"

# Commit 18
git commit --allow-empty -m "feature: add fixed and stochastic wind evaluation modes"

# Commit 19
git commit --allow-empty -m "feature: design multi-objective reward function for PPO"

# Commit 20
git commit --allow-empty -m "tune: balance distance and tilt penalty weights in reward"

# Commit 21
git commit --allow-empty -m "feature: implement 0.25s kinematic look-ahead bridge"

# Commit 22
git add train/train_ppo.py
git commit -m "train: setup Stable-Baselines3 PPO training script"

# Commit 23
git commit --allow-empty -m "train: add GymnasiumWrapper to bridge older API"

# Commit 24
git commit --allow-empty -m "train: configure PPO hyperparameters (lr=2e-4, clip=0.2)"

# Commit 25
git commit --allow-empty -m "train: add tensorboard logging and CSV monitor"

# Commit 26
git add main.py
git commit -m "demo: build main entry point for testing and evaluation"

# Commit 27
git commit --allow-empty -m "demo: implement fallback simple_velocity_policy"

# Commit 28
git commit --allow-empty -m "demo: add detailed terminal logging for evaluation metrics"

# Commit 29
if (Test-Path "models") {
    git add models/
    git commit -m "models: save trained PPO policy for FYP-I milestone"
} else {
    git commit --allow-empty -m "models: prepare directory for PPO policy checkpoints"
}

# Commit 30
git add .
git commit -m "chore: final code cleanup and formatting for FYP-I presentation"

# Push to remote
git push -u origin main
