import numpy as np
import pybullet as p
from gym import spaces

from env.BaseSingleAgentAviary import ActionType, BaseSingleAgentAviary, ObservationType
from utils.enums import DroneModel, Physics


class HoverEnv(BaseSingleAgentAviary):
    """Simple FYP-1 drone navigation environment.

    PPO learns only high-level velocity commands:
        action = [vx, vy, vz]

    PID converts those commands into stable thrust/attitude/RPM control.
    Wind is modeled as one external force applied to the drone body.
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,
        target_xyz=None,
        wind_enabled=True,
        wind_strength=0.006,
        random_wind=False,
        max_xy_speed=0.35,
        max_z_speed=0.22,
        log_every_steps=120,
    ):
        self.TARGET_POS = np.array(target_xyz if target_xyz is not None else [1.0, 0.6, 1.0], dtype=np.float32)
        self.WIND_ENABLED = bool(wind_enabled)
        self.WIND_STRENGTH = float(wind_strength)
        self.RANDOM_WIND = bool(random_wind)
        self.MAX_XY_SPEED = float(max_xy_speed)
        self.MAX_Z_SPEED = float(max_z_speed)
        self.LOG_EVERY_STEPS = int(log_every_steps)

        self.VELOCITY_LOOKAHEAD_SEC = 0.25
        self.GOAL_RADIUS = 0.30
        self.EPISODE_LEN_SEC = 12

        self.last_wind = np.zeros(3, dtype=np.float32)
        self.last_desired_velocity = np.zeros(3, dtype=np.float32)
        self.wind_phase = np.zeros(2, dtype=np.float32)
        self.current_wind_strength = self.WIND_STRENGTH
        self.previous_distance = None
        self.episode_reward = 0.0
        self.episode_min_distance = np.inf
        self.episode_max_tilt = 0.0
        self.episode_steps = 0
        self.episode_success = False
        self._wind_line_id = -1
        # FYP-II Phase 2: Temporal awareness — track the previous PPO action
        self.last_ppo_action = np.zeros(3, dtype=np.float32)
        self.action_difference = np.zeros(3, dtype=np.float32)

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.25]])

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        self.EPISODE_LEN_SEC = 12

    def reset(self):
        obs = super().reset()
        if hasattr(self, "ctrl"):
            self.ctrl.reset()

        self.previous_distance = self._distance_to_target()
        self.episode_reward = 0.0
        self.episode_min_distance = self.previous_distance
        self.episode_max_tilt = 0.0
        self.episode_steps = 0
        self.episode_success = False
        self.last_wind = np.zeros(3, dtype=np.float32)
        self.last_desired_velocity = np.zeros(3, dtype=np.float32)
        # FYP-II Phase 2: Reset temporal action tracking
        self.last_ppo_action = np.zeros(3, dtype=np.float32)
        self.action_difference = np.zeros(3, dtype=np.float32)
        self._reset_wind()
        self._draw_target()
        return obs

    def _actionSpace(self):
        return spaces.Box(low=-np.ones(3), high=np.ones(3), dtype=np.float32)

    def _observationSpace(self):
        # FYP-II Phase 2: Expanded from 16D to 19D (adds prev action [vx, vy, vz])
        return spaces.Box(low=-np.ones(19), high=np.ones(19), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        target_error = self.TARGET_POS - pos
        distance = np.linalg.norm(target_error)
        wind_scale = max(self.WIND_STRENGTH, 0.001)

        obs = np.hstack([
            np.clip(target_error / 3.0, -1.0, 1.0),      # dims  0-2:  target error
            np.clip(distance / 3.0, 0.0, 1.0),            # dim   3:    scalar distance
            np.clip(vel / 3.0, -1.0, 1.0),                # dims  4-6:  linear velocity
            np.clip(rpy / np.pi, -1.0, 1.0),              # dims  7-9:  roll/pitch/yaw
            np.clip(ang_vel / 8.0, -1.0, 1.0),            # dims 10-12: angular velocity
            np.clip(self.last_wind / wind_scale, -1.0, 1.0),  # dims 13-15: wind vector
            self.last_ppo_action,                          # dims 16-18: prev PPO action (already in [-1,1])
        ])
        return obs.astype(np.float32)

    def _preprocessAction(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(3,)
        action = np.clip(action, -1.0, 1.0)

        # FYP-II Phase 2: Compute action difference for smoothness penalty, then update
        self.action_difference = action - self.last_ppo_action
        self.last_ppo_action = action.copy()

        # FYP-II Phase 1: Residual Reinforcement Learning (RRL)
        # Compute baseline proportional action to target position
        state = self._getDroneStateVector(0)
        target_error = self.TARGET_POS - state[0:3]
        
        base_action = np.zeros(3, dtype=np.float32)
        base_action[0] = np.clip(0.55 * target_error[0], -0.8, 0.8)
        base_action[1] = np.clip(0.55 * target_error[1], -0.8, 0.8)
        base_action[2] = np.clip(0.85 * target_error[2], -0.9, 0.9)

        # Total action = baseline action + PPO residual correction action
        total_action = np.clip(base_action + action, -1.0, 1.0)

        desired_velocity = np.array([
            total_action[0] * self.MAX_XY_SPEED,
            total_action[1] * self.MAX_XY_SPEED,
            total_action[2] * self.MAX_Z_SPEED,
        ], dtype=np.float32)

        state = self._getDroneStateVector(0)
        target_pos = state[0:3] + desired_velocity * self.VELOCITY_LOOKAHEAD_SEC
        target_pos[2] = np.clip(target_pos[2], 0.25, 2.2)
        self.last_desired_velocity = desired_velocity

        rpm, _, _ = self.ctrl.computeControl(
            control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, state[9]]),
            target_vel=np.zeros(3),
            target_rpy_rates=np.zeros(3),
        )
        return rpm

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        self._apply_wind(nth_drone)

    def _apply_wind(self, nth_drone):
        if not self.WIND_ENABLED:
            self.last_wind = np.zeros(3, dtype=np.float32)
            return

        t = self.step_counter * self.TIMESTEP
        wind = np.array([
            self.current_wind_strength * np.sin(0.7 * t + self.wind_phase[0]),
            self.current_wind_strength * np.cos(0.5 * t + self.wind_phase[1]),
            0.0,
        ], dtype=np.float32)

        self.last_wind = wind
        p.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            -1,
            forceObj=wind,
            posObj=self.pos[nth_drone].tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT,
        )
        self._draw_wind()

    def _reset_wind(self):
        if not self.RANDOM_WIND:
            self.wind_phase = np.zeros(2, dtype=np.float32)
            self.current_wind_strength = self.WIND_STRENGTH
            return

        self.wind_phase = np.random.uniform(0.0, 2.0 * np.pi, size=2).astype(np.float32)
        strength_scale = np.random.uniform(0.75, 1.25)
        self.current_wind_strength = self.WIND_STRENGTH * strength_scale

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        rpy = state[7:10]
        ang_vel = state[13:16]

        distance = np.linalg.norm(self.TARGET_POS - pos)
        progress = 0.0 if self.previous_distance is None else self.previous_distance - distance
        direction = self.TARGET_POS - pos
        direction_norm = np.linalg.norm(direction)
        unit_direction = direction / direction_norm if direction_norm > 1e-6 else np.zeros(3)
        action_alignment = float(np.dot(self.last_desired_velocity, unit_direction))
        tilt = np.linalg.norm(rpy[0:2])

        reward = 30.0 * progress
        reward += 2.0 * action_alignment
        reward -= 0.5 * distance
        reward -= 0.05 * np.linalg.norm(vel)
        reward -= 0.3 * tilt
        reward -= 0.02 * np.linalg.norm(ang_vel)

        # FYP-II Phase 2: Action smoothness penalty — penalises jerky changes between steps
        # Inspired by SimpleFlight (2025): action smoothness is #1 factor for sim-to-real transfer
        smoothness_penalty = -0.1 * float(np.sum(np.square(self.action_difference)))
        reward += smoothness_penalty

        # FYP-II Phase 2: Energy regularisation — penalises unnecessarily high motor RPMs
        # Inspired by Energy-Aware UAV Navigation DRL (IEEE 2024)
        motor_rpms = np.clip(self.last_action[0, :], 0, None)  # shape (4,), clamp to non-negative
        energy_penalty = -0.01 * float(np.mean(np.square(motor_rpms / 10000.0)))
        reward += energy_penalty

        if progress < 0.0:
            reward += 15.0 * progress

        if distance < self.GOAL_RADIUS:
            reward += 100.0
            self.episode_success = True

        if pos[2] < 0.10 or distance > 5.0 or tilt > 1.4:
            reward -= 100.0

        self.previous_distance = distance
        self.episode_reward += float(reward)
        self.episode_min_distance = min(self.episode_min_distance, float(distance))
        self.episode_max_tilt = max(self.episode_max_tilt, float(tilt))
        self.episode_steps += 1
        return float(reward)

    def _computeDone(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        tilt = np.linalg.norm(state[7:9])
        distance = np.linalg.norm(self.TARGET_POS - pos)

        if distance < self.GOAL_RADIUS:
            self.episode_success = True
            return True
        if pos[2] < 0.08:
            return True
        if distance > 5.0:
            return True
        if tilt > 1.4:
            return True
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        return {
            "distance": self._distance_to_target(),
            "target": self.TARGET_POS.copy(),
            "position": state[0:3].copy(),
            "velocity": state[10:13].copy(),
            "rpy": state[7:10].copy(),
            "angular_velocity": state[13:16].copy(),
            "wind": self.last_wind.copy(),
            "desired_velocity": self.last_desired_velocity.copy(),
            "episode_reward": float(self.episode_reward),
            "episode_min_distance": float(self.episode_min_distance),
            "episode_max_tilt": float(self.episode_max_tilt),
            "success": bool(self.episode_success),
            "steps": int(self.episode_steps),
        }

    def _distance_to_target(self):
        state = self._getDroneStateVector(0)
        return float(np.linalg.norm(self.TARGET_POS - state[0:3]))

    def _draw_target(self):
        if not self.GUI:
            return

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.GOAL_RADIUS,
            rgbaColor=[0.1, 0.8, 0.2, 0.45],
            physicsClientId=self.CLIENT,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.TARGET_POS.tolist(),
            physicsClientId=self.CLIENT,
        )

    def _draw_wind(self):
        if not self.GUI or self.step_counter % 12 != 0:
            return

        start = self.pos[0].tolist()
        end = (np.array(start) + 70.0 * self.last_wind).tolist()
        self._wind_line_id = p.addUserDebugLine(
            start,
            end,
            lineColorRGB=[0.1, 0.35, 1.0],
            lineWidth=3,
            lifeTime=0.12,
            replaceItemUniqueId=int(self._wind_line_id),
            physicsClientId=self.CLIENT,
        )
