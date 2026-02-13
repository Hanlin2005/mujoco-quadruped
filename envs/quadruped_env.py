"""Custom Gymnasium environment for Unitree Go1 quadruped locomotion."""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class QuadrupedEnv(gym.Env):
    """Gymnasium environment for training a Unitree Go1 quadruped to walk.

    Observation space (48 dims):
        - joint positions (12)
        - joint velocities (12)
        - trunk orientation quaternion (4)
        - trunk angular velocity (3)
        - trunk linear velocity (3)
        - foot contact forces (4) - binary
        - gravity vector in body frame (3)
        - commanded velocity (3) - [vx, vy, yaw_rate]
        - previous actions (12) - for smoothness
        Total: 12+12+4+3+3+4+3+3+12 = 56

    Action space (12 dims):
        - Target joint position offsets added to default standing pose
        - Clipped to actuator control ranges
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # Default standing joint positions from the Go1 keyframe
    DEFAULT_JOINT_POS = np.array([
        0.0, 0.9, -1.8,  # FR: hip, thigh, knee
        0.0, 0.9, -1.8,  # FL
        0.0, 0.9, -1.8,  # RR
        0.0, 0.9, -1.8,  # RL
    ])

    # Foot site names for contact detection
    FOOT_SITES = ["FR", "FL", "RR", "RL"]

    # Foot geom names for contact detection
    FOOT_GEOMS = ["FR", "FL", "RR", "RL"]

    def __init__(
        self,
        xml_path=None,
        render_mode=None,
        max_episode_steps=1000,
        action_scale=0.3,
        ctrl_cost_weight=0.01,
        alive_bonus=1.0,
        velocity_tracking_weight=2.0,
        fall_penalty=5.0,
        smoothness_weight=0.05,
        foot_slip_weight=0.1,
        orientation_weight=0.5,
        height_weight=0.5,
        cmd_vx=1.0,
        cmd_vy=0.0,
        cmd_yaw_rate=0.0,
        domain_randomize=False,
        terrain_xml=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale

        # Reward weights
        self.ctrl_cost_weight = ctrl_cost_weight
        self.alive_bonus = alive_bonus
        self.velocity_tracking_weight = velocity_tracking_weight
        self.fall_penalty = fall_penalty
        self.smoothness_weight = smoothness_weight
        self.foot_slip_weight = foot_slip_weight
        self.orientation_weight = orientation_weight
        self.height_weight = height_weight

        # Commanded velocity
        self.cmd_vx = cmd_vx
        self.cmd_vy = cmd_vy
        self.cmd_yaw_rate = cmd_yaw_rate

        # Domain randomization
        self.domain_randomize = domain_randomize

        # Load MuJoCo model
        if terrain_xml is not None:
            self._xml_path = terrain_xml
        elif xml_path is not None:
            self._xml_path = xml_path
        else:
            self._xml_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "models", "scene.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(self._xml_path)
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.sim_steps_per_action = 10  # 50 Hz control at 0.002s timestep
        self.control_dt = self.dt * self.sim_steps_per_action

        # Body/joint/site IDs
        self.trunk_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self.foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.FOOT_SITES
        ]
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.FOOT_GEOMS
        ]
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        # Store default model parameters for domain randomization
        self._default_body_mass = self.model.body_mass.copy()
        self._default_dof_damping = self.model.dof_damping.copy()
        self._default_geom_friction = self.model.geom_friction.copy()
        self._default_actuator_gainprm = self.model.actuator_gainprm.copy()

        # Previous action for smoothness penalty
        self._prev_action = np.zeros(12)

        # Step counter
        self._step_count = 0

        # Action space: position offsets for 12 joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Observation space
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Renderer for rgb_array mode
        self._renderer = None
        if self.render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _get_obs(self):
        """Construct observation vector."""
        # Joint positions (12) - relative to default standing
        joint_pos = self.data.qpos[7:] - self.DEFAULT_JOINT_POS

        # Joint velocities (12)
        joint_vel = self.data.qvel[6:] * 0.1  # scale down

        # Trunk orientation quaternion (4)
        trunk_quat = self.data.qpos[3:7]

        # Trunk angular velocity (3)
        trunk_angvel = self.data.qvel[3:6] * 0.25

        # Trunk linear velocity (3)
        trunk_linvel = self.data.qvel[0:3]

        # Foot contact binary (4)
        foot_contacts = self._get_foot_contacts().astype(np.float32)

        # Gravity vector in body frame (3)
        gravity_body = self._get_gravity_body_frame()

        # Commanded velocity (3)
        cmd = np.array([self.cmd_vx, self.cmd_vy, self.cmd_yaw_rate])

        # Previous action (12)
        prev_action = self._prev_action

        obs = np.concatenate([
            joint_pos,        # 12
            joint_vel,        # 12
            trunk_quat,       # 4
            trunk_angvel,     # 3
            trunk_linvel,     # 3
            foot_contacts,    # 4
            gravity_body,     # 3
            cmd,              # 3
            prev_action,      # 12
        ]).astype(np.float32)

        return obs

    def _get_foot_contacts(self):
        """Return binary array of foot-ground contacts."""
        contacts = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            for j, foot_geom_id in enumerate(self.foot_geom_ids):
                if (geom1 == foot_geom_id and geom2 == self.floor_geom_id) or \
                   (geom2 == foot_geom_id and geom1 == self.floor_geom_id):
                    contacts[j] = True
        return contacts

    def _get_gravity_body_frame(self):
        """Get gravity vector rotated into the trunk body frame."""
        trunk_quat = self.data.qpos[3:7]
        # Rotate world gravity [0, 0, -9.81] into body frame
        gravity_world = np.array([0.0, 0.0, -1.0])
        rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rot_matrix, trunk_quat)
        rot_matrix = rot_matrix.reshape(3, 3)
        gravity_body = rot_matrix.T @ gravity_world
        return gravity_body

    def _apply_domain_randomization(self):
        """Randomize physical parameters for sim-to-real transfer."""
        if not self.domain_randomize:
            return

        rng = self.np_random

        # Body mass: ±15%
        mass_scale = rng.uniform(0.85, 1.15, size=self.model.body_mass.shape)
        self.model.body_mass[:] = self._default_body_mass * mass_scale

        # Joint damping: ±20%
        damping_scale = rng.uniform(0.8, 1.2, size=self.model.dof_damping.shape)
        self.model.dof_damping[:] = self._default_dof_damping * damping_scale

        # Ground friction: ±30%
        friction_scale = rng.uniform(0.7, 1.3, size=self.model.geom_friction.shape)
        self.model.geom_friction[:] = self._default_geom_friction * friction_scale

        # Motor strength (gain): ±10%
        gain_scale = rng.uniform(0.9, 1.1, size=self.model.actuator_gainprm.shape)
        self.model.actuator_gainprm[:] = self._default_actuator_gainprm * gain_scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Apply domain randomization
        self._apply_domain_randomization()

        # Reset to home keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Add small random perturbation to initial joint positions
        noise = self.np_random.uniform(-0.05, 0.05, size=12)
        self.data.qpos[7:] += noise
        self.data.ctrl[:] = self.data.qpos[7:]

        # Randomize initial orientation slightly
        euler_noise = self.np_random.uniform(-0.05, 0.05, size=3)
        quat_noise = np.zeros(4)
        mujoco.mju_euler2Quat(quat_noise, euler_noise, "xyz")
        result_quat = np.zeros(4)
        mujoco.mju_mulQuat(result_quat, self.data.qpos[3:7], quat_noise)
        self.data.qpos[3:7] = result_quat

        mujoco.mj_forward(self.model, self.data)

        # Reset internal state
        self._prev_action = np.zeros(12)
        self._step_count = 0

        # Optionally randomize commanded velocity
        if options and options.get("randomize_cmd", False):
            self.cmd_vx = self.np_random.uniform(0.0, 1.5)
            self.cmd_vy = self.np_random.uniform(-0.3, 0.3)
            self.cmd_yaw_rate = self.np_random.uniform(-0.5, 0.5)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Scale action and add to default standing position
        target_pos = self.DEFAULT_JOINT_POS + action * self.action_scale

        # Clip to actuator control ranges
        ctrl_range_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_range_high = self.model.actuator_ctrlrange[:, 1]
        target_pos = np.clip(target_pos, ctrl_range_low, ctrl_range_high)

        # Store pre-step state for reward computation
        trunk_pos_before = self.data.qpos[0:3].copy()

        # Apply action and simulate
        self.data.ctrl[:] = target_pos
        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Post-step state
        trunk_pos_after = self.data.qpos[0:3].copy()

        # Compute reward components
        reward, reward_info = self._compute_reward(
            action, trunk_pos_before, trunk_pos_after
        )

        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= self.max_episode_steps

        # Update previous action
        self._prev_action = action.copy()

        obs = self._get_obs()
        info = {
            "step": self._step_count,
            "trunk_height": self.data.qpos[2],
            "trunk_x": self.data.qpos[0],
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action, pos_before, pos_after):
        """Compute shaped reward for locomotion."""
        info = {}

        # Forward velocity reward (track commanded velocity)
        dt = self.control_dt
        velocity = (pos_after - pos_before) / dt
        vx, vy = velocity[0], velocity[1]

        # Velocity tracking: reward for matching commanded velocity
        vx_error = (vx - self.cmd_vx) ** 2
        vy_error = (vy - self.cmd_vy) ** 2

        # Yaw rate tracking
        yaw_rate = self.data.qvel[5]
        yaw_error = (yaw_rate - self.cmd_yaw_rate) ** 2

        velocity_reward = np.exp(-2.0 * (vx_error + vy_error + 0.5 * yaw_error))
        info["r_velocity"] = velocity_reward

        # Alive bonus
        alive = self.alive_bonus
        info["r_alive"] = alive

        # Control cost (energy)
        ctrl_cost = self.ctrl_cost_weight * np.sum(action ** 2)
        info["r_ctrl_cost"] = -ctrl_cost

        # Action smoothness penalty
        action_diff = action - self._prev_action
        smoothness_penalty = self.smoothness_weight * np.sum(action_diff ** 2)
        info["r_smoothness"] = -smoothness_penalty

        # Orientation penalty: penalize tilting
        trunk_quat = self.data.qpos[3:7]
        gravity_body = self._get_gravity_body_frame()
        # How much gravity deviates from pointing straight down in body frame
        orientation_error = np.sum(gravity_body[:2] ** 2)
        orientation_penalty = self.orientation_weight * orientation_error
        info["r_orientation"] = -orientation_penalty

        # Height reward: stay near target height
        target_height = 0.27
        height_error = (self.data.qpos[2] - target_height) ** 2
        height_penalty = self.height_weight * height_error
        info["r_height"] = -height_penalty

        # Foot slip penalty
        foot_contacts = self._get_foot_contacts()
        slip_penalty = 0.0
        for i, site_id in enumerate(self.foot_site_ids):
            if foot_contacts[i]:
                foot_vel = np.zeros(6)
                mujoco.mj_objectVelocity(
                    self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site_id,
                    foot_vel, 0
                )
                # Linear velocity of contact foot
                slip_speed = np.linalg.norm(foot_vel[3:5])  # only x,y
                slip_penalty += slip_speed
        slip_penalty *= self.foot_slip_weight
        info["r_foot_slip"] = -slip_penalty

        # Total reward
        reward = (
            self.velocity_tracking_weight * velocity_reward
            + alive
            - ctrl_cost
            - smoothness_penalty
            - orientation_penalty
            - height_penalty
            - slip_penalty
        )

        info["r_total"] = reward
        return reward, info

    def _check_termination(self):
        """Check if episode should terminate (robot fell)."""
        trunk_height = self.data.qpos[2]
        trunk_quat = self.data.qpos[3:7]

        # Fell over: trunk too low
        if trunk_height < 0.12:
            return True

        # Excessive tilt: check body z-axis alignment with world z
        rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rot_matrix, trunk_quat)
        rot_matrix = rot_matrix.reshape(3, 3)
        body_z = rot_matrix[:, 2]
        # Dot product with world z-axis
        z_alignment = body_z[2]
        if z_alignment < 0.3:  # roughly >70 degree tilt
            return True

        return False

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data, camera="tracking")
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
