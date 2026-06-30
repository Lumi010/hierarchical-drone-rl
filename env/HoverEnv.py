import numpy as np
import pybullet as p
from gym import spaces
from scipy.signal import tf2ss, cont2discrete

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
        randomize_dynamics=False,
        obstacles_enabled=True,
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

        # FYP-II Phase 4: Curriculum Learning & Domain Randomization
        self.total_env_steps = 0
        self.RANDOMIZE_DYNAMICS = bool(randomize_dynamics)
        self.OBSTACLES = bool(obstacles_enabled)

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

        # FYP-II Phase 4: Store nominal properties loaded from URDF
        self.NOMINAL_M = self.M
        self.NOMINAL_J_DIAG = np.array([self.J[0,0], self.J[1,1], self.J[2,2]], dtype=np.float32)
        self.NOMINAL_KF = self.KF
        self.NOMINAL_KM = self.KM

        # FYP-II Phase 3: Initialize continuous turbulence filters
        self._init_dryden_filters()

        self.EPISODE_LEN_SEC = 12

    def reset(self):
        # FYP-II Phase 5: Clear stale obstacle IDs before super().reset()
        # because BaseAviary.reset() calls p.resetSimulation() (destroys all bodies)
        # then _housekeeping() -> _addObstacles() re-creates them.
        self.obstacle_ids = []

        obs = super().reset()  # resetSimulation → _housekeeping → _addObstacles
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

        # FYP-II Phase 5: Position camera to focus directly on the drone, obstacle, and target
        if self.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=-45,
                cameraPitch=-20,
                cameraTargetPosition=[0.5, 0.3, 0.7],
                physicsClientId=self.CLIENT
            )
        return obs

    def _actionSpace(self):
        return spaces.Box(low=-np.ones(3), high=np.ones(3), dtype=np.float32)

    def _observationSpace(self):
        # Base observation space is 19D (adds prev action to 16D base)
        # FYP-II Phase 5: Expanded to 27D when obstacles are enabled (adds 8 normalized raycast distances)
        dim = 27 if getattr(self, "OBSTACLES", False) else 19
        return spaces.Box(low=-np.ones(dim), high=np.ones(dim), dtype=np.float32)

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

        # FYP-II Phase 5: Query raycast sensors and append to observations (dims 19-26)
        if self.OBSTACLES:
            ray_obs = self._get_raycast_observations()
            obs = np.hstack([obs, ray_obs])

        return obs.astype(np.float32)

    def _preprocessAction(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(3,)
        action = np.clip(action, -1.0, 1.0)

        # FYP-II Phase 4: Track total training steps
        self.total_env_steps += 1

        # FYP-II Phase 2: Compute action difference for smoothness penalty, then update
        self.action_difference = action - self.last_ppo_action
        self.last_ppo_action = action.copy()

        # FYP-II Phase 1: Residual Reinforcement Learning (RRL) with Vortex APF (Phase 5)
        # Compute baseline navigation using Vortex Artificial Potential Fields (VAPF)
        state = self._getDroneStateVector(0)
        target_error = self.TARGET_POS - state[0:3]
        
        # Calculate attractive force (target pull)
        F_a = 0.55 * target_error
        
        # Calculate repulsive and vortex forces (obstacle push and swirl)
        F_r = np.zeros(3, dtype=np.float32)
        F_v = np.zeros(3, dtype=np.float32)
        
        if self.OBSTACLES and hasattr(self, "obstacle_positions"):
            d_inf = 1.2  # distance of influence — start steering 1.2m away
            k_r = 0.8    # repulsive force scaling (strong push away)
            k_v = 1.2    # vortex force scaling (strong tangential swirl)
            
            for obs_pos in self.obstacle_positions:
                to_drone = state[0:2] - obs_pos[0:2]
                dist = np.linalg.norm(to_drone)
                
                if dist < d_inf:
                    # Clamp minimum distance to obstacle radius to prevent division blowup
                    dist = max(dist, self.obstacle_radius + 0.02)
                    dir_away = to_drone / dist
                    
                    # Repulsive force: F_r = k_r * (1/d - 1/d_inf) * (1/d^2)
                    fr_mag = k_r * (1.0 / dist - 1.0 / d_inf) * (1.0 / (dist ** 2))
                    fr_2d = fr_mag * dir_away
                    F_r[0] += fr_2d[0]
                    F_r[1] += fr_2d[1]
                    
                    # Vortex force: tangential vector (rotated by 90 degrees)
                    # Rotate dynamically toward target side to prevent orbiting away from target
                    vortex_dir = np.array([-dir_away[1], dir_away[0]], dtype=np.float32)
                    to_target = self.TARGET_POS[0:2] - state[0:2]
                    if np.dot(vortex_dir, to_target) < 0:
                        vortex_dir = -vortex_dir
                    
                    fv_2d = k_v * fr_mag * vortex_dir
                    F_v[0] += fv_2d[0]
                    F_v[1] += fv_2d[1]
        
        base_action = np.zeros(3, dtype=np.float32)
        base_action[0] = np.clip(F_a[0] + F_r[0] + F_v[0], -0.8, 0.8)
        base_action[1] = np.clip(F_a[1] + F_r[1] + F_v[1], -0.8, 0.8)
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

        # FYP-II Phase 3: MIL-F-8785C Dryden Turbulence model
        # Generate independent Gaussian white noise scaled to discrete time step
        dt = self.TIMESTEP
        w_scale = 1.0 / np.sqrt(dt)
        w_u = np.random.normal(0.0, 1.0) * w_scale
        w_v = np.random.normal(0.0, 1.0) * w_scale

        # Step the discrete-time state-space system equations
        self.dryden_x_u = np.dot(self.Ad_u, self.dryden_x_u) + self.Bd_u * w_u
        self.dryden_x_v = np.dot(self.Ad_v, self.dryden_x_v) + self.Bd_v * w_v

        # Calculate outputs (gust velocities)
        gust_u = float(np.dot(self.Cd_u, self.dryden_x_u) + self.Dd_u * w_u)
        gust_v = float(np.dot(self.Cd_v, self.dryden_x_v) + self.Dd_v * w_v)

        # Normalize outputs and scale to convert to wind force in Newtons
        force_x = self.current_wind_strength * (gust_u / self.sigma_u)
        force_y = self.current_wind_strength * (gust_v / self.sigma_v)

        wind = np.array([force_x, force_y, 0.0], dtype=np.float32)
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
        # FYP-II Phase 4: Curriculum Learning
        # Define 3 stages based on total environment steps
        if self.total_env_steps < 30000:
            # Stage 1: Calm air, no dynamics randomization
            self.current_curriculum_stage = 1
            active_wind_strength = 0.0
            active_rand_scale = 0.0
        elif self.total_env_steps < 80000:
            # Stage 2: Light wind, moderate dynamics randomization (50% bounds)
            self.current_curriculum_stage = 2
            active_wind_strength = self.WIND_STRENGTH * 0.5
            active_rand_scale = 0.5
        else:
            # Stage 3: Full storm, full dynamics randomization (100% bounds)
            self.current_curriculum_stage = 3
            active_wind_strength = self.WIND_STRENGTH
            active_rand_scale = 1.0

        # Compute wind strength scale (apply random fluctuations if RANDOM_WIND is enabled)
        if not self.RANDOM_WIND:
            self.wind_phase = np.zeros(2, dtype=np.float32)
            self.current_wind_strength = active_wind_strength
        else:
            self.wind_phase = np.random.uniform(0.0, 2.0 * np.pi, size=2).astype(np.float32)
            strength_scale = np.random.uniform(0.75, 1.25)
            self.current_wind_strength = active_wind_strength * strength_scale

        # Reset Dryden filter states
        if hasattr(self, "dryden_x_u") and self.dryden_x_u is not None:
            self.dryden_x_u.fill(0.0)
            self.dryden_x_v.fill(0.0)

        # FYP-II Phase 4: Domain Randomization
        # Randomize mass, inertia, and motor coefficients if enabled
        if self.RANDOMIZE_DYNAMICS and active_rand_scale > 0.0:
            # 1. Mass randomization: +/-15% scaled by curriculum
            mass_scale = 1.0 + np.random.uniform(-0.15, 0.15) * active_rand_scale
            self.M = self.NOMINAL_M * mass_scale
            self.GRAVITY = self.G * self.M
            p.changeDynamics(self.DRONE_IDS[0], -1, mass=self.M, physicsClientId=self.CLIENT)

            # 2. Inertia diagonal randomization: +/-10% scaled by curriculum
            inertia_scale = 1.0 + np.random.uniform(-0.10, 0.10, size=3) * active_rand_scale
            random_inertia = self.NOMINAL_J_DIAG * inertia_scale
            p.changeDynamics(self.DRONE_IDS[0], -1, localInertiaDiagonal=random_inertia.tolist(), physicsClientId=self.CLIENT)

            # 3. Motor coefficient randomization: +/-8% scaled by curriculum
            kf_scale = 1.0 + np.random.uniform(-0.08, 0.08) * active_rand_scale
            km_scale = 1.0 + np.random.uniform(-0.08, 0.08) * active_rand_scale
            self.KF = self.NOMINAL_KF * kf_scale
            self.KM = self.NOMINAL_KM * km_scale
        else:
            # Restore nominal parameters if randomization is disabled or at Stage 1
            self.M = self.NOMINAL_M
            self.GRAVITY = self.G * self.M
            self.KF = self.NOMINAL_KF
            self.KM = self.NOMINAL_KM
            if hasattr(self, "DRONE_IDS") and len(self.DRONE_IDS) > 0:
                p.changeDynamics(self.DRONE_IDS[0], -1, mass=self.NOMINAL_M, localInertiaDiagonal=self.NOMINAL_J_DIAG.tolist(), physicsClientId=self.CLIENT)

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

        # Check collision with obstacles
        collision_with_obstacle = False
        if self.OBSTACLES and hasattr(self, "obstacle_ids"):
            for obs_id in self.obstacle_ids:
                contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=obs_id, physicsClientId=self.CLIENT)
                if len(contacts) > 0:
                    collision_with_obstacle = True
                    reward -= 100.0  # Big penalty for crashing into obstacle
                    break

        if pos[2] < 0.10 or distance > 5.0 or tilt > 1.4 or collision_with_obstacle:
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
        # Check collision with obstacles
        if self.OBSTACLES and hasattr(self, "obstacle_ids"):
            for obs_id in self.obstacle_ids:
                contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=obs_id, physicsClientId=self.CLIENT)
                if len(contacts) > 0:
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

    def _init_dryden_filters(self):
        # Nominal parameters for low altitude (under 1000 ft) per MIL-F-8785C
        h_ft = 20.0  # nominal altitude in feet (approx 6 meters)
        V_a = 1.0    # nominal airspeed in m/s (clamped to prevent division by zero in hover)

        # Conversion: 1 knot = 1.68781 ft/s
        W_20_ft_per_sec = 15.0 * 1.68781  # light turbulence wind speed at 20 ft
        V_a_ft_per_sec = V_a * 3.28084

        # Turbulence scale lengths (L_u, L_v in feet)
        L_u = h_ft / ((0.177 + 0.000823 * h_ft) ** 1.2)
        L_v = L_u

        # Turbulence intensities (sigma_u, sigma_v in ft/s)
        sigma_w = 0.1 * W_20_ft_per_sec
        sigma_u = sigma_w / ((0.177 + 0.000823 * h_ft) ** 0.4)
        sigma_v = sigma_u

        self.sigma_u = sigma_u
        self.sigma_v = sigma_v

        # Time constants (T_u, T_v in seconds)
        T_u = L_u / V_a_ft_per_sec
        T_v = L_v / V_a_ft_per_sec

        # Longitudinal transfer function: H_u(s) = sigma_u * sqrt(2 * T_u / pi) / (T_u * s + 1)
        num_u = [sigma_u * np.sqrt(2.0 * T_u / np.pi)]
        den_u = [T_u, 1.0]

        # Lateral transfer function: H_v(s) = sigma_v * sqrt(T_v / pi) * (sqrt(3) * T_v * s + 1) / (T_v^2 * s^2 + 2 * T_v * s + 1)
        num_v = [sigma_v * np.sqrt(T_v / np.pi) * np.sqrt(3.0) * T_v, sigma_v * np.sqrt(T_v / np.pi)]
        den_v = [T_v ** 2, 2.0 * T_v, 1.0]

        # Convert to continuous state-space representation
        Ac_u, Bc_u, Cc_u, Dc_u = tf2ss(num_u, den_u)
        Ac_v, Bc_v, Cc_v, Dc_v = tf2ss(num_v, den_v)

        # Discretize continuous state space at 240Hz using bilinear transform
        dt = 1.0 / self.SIM_FREQ
        self.Ad_u, self.Bd_u, self.Cd_u, self.Dd_u, _ = cont2discrete((Ac_u, Bc_u, Cc_u, Dc_u), dt, method="bilinear")
        self.Ad_v, self.Bd_v, self.Cd_v, self.Dd_v, _ = cont2discrete((Ac_v, Bc_v, Cc_v, Dc_v), dt, method="bilinear")

        # Initialize filter state vectors
        self.dryden_x_u = np.zeros((Ac_u.shape[0], 1), dtype=np.float32)
        self.dryden_x_v = np.zeros((Ac_v.shape[0], 1), dtype=np.float32)

    def _addObstacles(self):
        """Add static cylindrical column obstacles in PyBullet.
        
        Called after every reset() because BaseAviary.reset() calls
        p.resetSimulation() which destroys all previously created bodies.
        """
        if not self.OBSTACLES:
            return

        self.obstacle_radius = 0.20
        self.obstacle_positions = [
            np.array([0.5, 0.3, 1.0], dtype=np.float32)  # half-way pillar directly in path
        ]
        self.obstacle_ids = []

        # Cylindrical collision and visual shape definitions
        col_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.obstacle_radius,
            height=2.0,
            physicsClientId=self.CLIENT
        )
        
        vis_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.obstacle_radius,
            length=2.0,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],  # solid opaque red for maximum visibility
            physicsClientId=self.CLIENT
        )

        for pos in self.obstacle_positions:
            body_id = p.createMultiBody(
                baseMass=0.0,  # Mass = 0 makes it static
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=pos.tolist(),
                physicsClientId=self.CLIENT
            )
            self.obstacle_ids.append(body_id)
            print(f"[INFO] Spawned cylinder obstacle with ID {body_id} at position {pos}")

    def _get_raycast_observations(self):
        """Returns 8 normalized distance measurements around the drone's current yaw."""
        if not self.OBSTACLES or not hasattr(self, "obstacle_ids") or len(self.obstacle_ids) == 0:
            return np.ones(8, dtype=np.float32)

        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        yaw = rpy[2]

        max_ray_dist = 1.5
        ray_from = []
        ray_to = []

        # Spacing rays at 45 degree intervals relative to current heading (yaw)
        for i in range(8):
            angle = yaw + i * (2.0 * np.pi / 8.0)
            dx = max_ray_dist * np.cos(angle)
            dy = max_ray_dist * np.sin(angle)
            ray_from.append(pos)
            ray_to.append(pos + np.array([dx, dy, 0.0], dtype=np.float32))

        # Query batch raycasts from PyBullet
        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.CLIENT)
        
        ray_obs = []
        for res in results:
            hit_fraction = res[2]  # float value in [0.0, 1.0] representing distance
            ray_obs.append(float(hit_fraction))

        return np.array(ray_obs, dtype=np.float32)
