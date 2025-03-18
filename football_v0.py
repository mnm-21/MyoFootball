import collections
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat
from scipy.spatial.transform import Rotation as R


class FootballEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = [
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'torso_angle',
        'feet_heights',
        'height',
        'feet_rel_positions',
        'phase_var',
        'muscle_length',
        'muscle_velocity',
        'muscle_force',
        'football_pos',
        'football_vel',
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 5.0,
        "done": -100.0,
        "cyclic_hip": -8.0,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0,
        "in_goal": 200.0,
        "kick": 20.0,
        "kick_force": 120.0,
        "ball_vel_reward": 10.0 # 30m/s kick in exact goal direction will give 120 reward
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               min_height = 0.8,
               max_rot = 0.8,
               hip_period = 100,
               reset_type='init',
               target_x_vel=0.0,
               target_y_vel=1.2,
               target_rot = None,
               **kwargs,
               ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

        # move heightfield down if not used
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])
        obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()])
        obs_dict['feet_heights'] = self._get_feet_heights().copy()
        obs_dict['height'] = np.array([self._get_height()]).copy()
        obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy()
        obs_dict['phase_var'] = np.array([(self.steps/self.hip_period) % 1]).copy()
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()
        football_body_id = self.sim.model.body_name2id("football")  # Get body ID
        obs_dict['football_pos'] = self.sim.data.body_xpos[football_body_id].copy()
        obs_dict['football_vel'] = self.sim.data.qvel[self.sim.model.body_dofadr[football_body_id] : self.sim.model.body_dofadr[football_body_id] + 6].copy()
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l',
                                                       'hip_rotation_r'])
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('vel_reward', vel_reward),   # add reward for moving foot to ball
            ('cyclic_hip',  cyclic_hip),                      
            ('ref_rot',  ref_rot),
            ('joint_angle_rew', joint_angle_rew),
            ('act_mag', act_mag), # not used 
            ('sparse',  vel_reward), # not used
            ('solved',    vel_reward >= 1.0), # not used 
            ('done',  self._get_done()),
            ('in_goal', self._get_goal_reward()),
            ('kick', self._get_kick_reward()),
            ('kick_force', self._get_kick_force()),
            ('ball_vel_reward', self._get_ball_vel_reward())
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def _get_goal_reward(self):
        model = self.sim.model
        data = self.sim.data

        crossbar_pos = data.geom_xpos[model.geom_name2id("crossbar")].copy()
        left_post_pos = data.geom_xpos[model.geom_name2id("left_post")].copy()
        right_post_pos = data.geom_xpos[model.geom_name2id("right_post")].copy()

        # Define goal boundaries
        goal_x_min, goal_x_max = left_post_pos[0], right_post_pos[0]  # X-bounds between the posts
        goal_y = left_post_pos[1]  # Y-position of the goal line
        goal_z_min, goal_z_max = 0, crossbar_pos[2]  # Ground to crossbar

        # Get ball position 
        ball_pos = data.body_xpos[model.body_name2id("football")].copy()

        # Check if the ball is inside the goal area
        in_goal = (goal_x_min <= ball_pos[0] <= goal_x_max and
                ball_pos[1] <= goal_y and  # Ball must be behind the goal line
                goal_z_min <= ball_pos[2] <= goal_z_max)
        
        return 1 if in_goal else 0

    def _get_kick_reward(self):

        football_geom_id = self.sim.model.geom_name2id("football_geom")
        foot_geom_ids = [self.sim.model.geom_name2id(name) for name in [
            "l_foot_col1", "r_foot_col1",  # Forefoot 
            "l_foot_col3", "r_foot_col3",  # Midfoot
            "l_foot_col4", "r_foot_col4",  # Heel 
            "l_bofoot_col1", "r_bofoot_col1",  # Toe tip 
            "l_bofoot_col2", "r_bofoot_col2",  # Toe base
            "l_talus", "r_talus",  # Ankle
            "l_foot", "r_foot",  # Full foot
            "l_bofoot", "r_bofoot"
        ]]
        reward = 0

        for contact in self.sim.data.contact:
            if (contact.geom1 == football_geom_id and contact.geom2 in foot_geom_ids) or (contact.geom2 == football_geom_id and contact.geom1 in foot_geom_ids):
                reward = 1  # Assign reward for a successful kick
                break  # No need to check further if a valid contact is found
        return reward

    def _get_kick_force(self):
        """
        Computes the reward based on the force applied by the foot to the ball.
        Rewards stronger and better-aligned kicks.
        """
        football_geom_id = self.sim.model.geom_name2id("football_geom")
        foot_geom_ids = [self.sim.model.geom_name2id(name) for name in [
            "l_foot_col1", "r_foot_col1",  
            "l_foot_col3", "r_foot_col3",  
            "l_foot_col4", "r_foot_col4", 
            "l_bofoot_col1", "r_bofoot_col1",  
            "l_bofoot_col2", "r_bofoot_col2",  
            "l_talus", "r_talus",  
            "l_foot", "r_foot",  
            "l_bofoot", "r_bofoot"
        ]]

        # Initialize total force vector (x, y components only)
        total_force_vec = np.zeros(2)
        contact_force = np.zeros(6)  # MuJoCo stores force as (3 force + 3 torque)

        # Sum forces across all contacts
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if (contact.geom1 == football_geom_id and contact.geom2 in foot_geom_ids) or \
            (contact.geom2 == football_geom_id and contact.geom1 in foot_geom_ids):

                mujoco.mj_contactForce(self.sim.model, self.sim.data, i, contact_force)

                if contact.geom1 == football_geom_id:
                    # Force exerted by football on foot; negate to get foot on football
                    total_force_vec += -contact_force[:2]  # Only x, y components
                else:
                    # Force exerted by foot on football; use as is
                    total_force_vec += contact_force[:2]

        # Compute total force magnitude
        total_force_mag = np.linalg.norm(total_force_vec)

        # Force Alignment with Goal (x, y only)
        ball_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("football")][:2]  # (x, y)
        goal_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("goalpost")][:2]  # (x, y)
        
        goal_dir_vec = goal_pos - ball_pos  # Vector from ball to goal
        goal_dir_norm = np.linalg.norm(goal_dir_vec) + 1e-6  # Avoid division by zero
        goal_dir_vec /= goal_dir_norm  # Normalize

        # dot product between force direction and goal direction
        if total_force_mag > 1e-6:  # Avoid division by zero
            force_alignment = np.dot(total_force_vec / total_force_mag, goal_dir_vec)
        else:
            force_alignment = 0.0  # No contact force, so no meaningful alignment

        force_alignment = np.clip(force_alignment, -1, 1)  # Ensure valid range

        # Compute Final Reward 
        if total_force_mag < 300:
            force_reward = (total_force_mag / 300) ** 2  # Quadratic increase up to 300N
        else:
            force_reward = np.exp(-0.002 * (total_force_mag - 300) ** 2)  # Decay beyond 300N

        # Scale reward based on alignment
        final_reward = force_reward * (force_alignment + 1) ** 2  # Squared to prioritize good alignment

        return final_reward

    def _get_ball_vel_reward(self):
        football_body_id = self.sim.model.body_name2id("football")  # Get body ID
        football_vel = self.sim.data.qvel[self.sim.model.body_dofadr[football_body_id] : self.sim.model.body_dofadr[football_body_id] + 6].copy()
        football_pos = self.sim.data.body_xpos[football_body_id].copy()
        linear_vel = football_vel[:3] 
        speed = np.linalg.norm(linear_vel)  # Speed in m/s

        # Define goal direction (should be unit vector)
        goal_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("goalpost")].copy()
        goal_dir = goal_pos - football_pos  # Vector from ball to goal
        goal_dir /= np.linalg.norm(goal_dir)  # unit vector

        goal_alignment = np.dot(linear_vel / (speed + 1e-6), goal_dir)  # Cosine similarity
        goal_alignment = np.clip(goal_alignment, -1, 1)

        alignment_reward = (goal_alignment + 1) ** 2  # Non-linear scaling
        base_reward = 0.1 * speed  # linear scaling
        reward = base_reward * alignment_reward
        return reward

    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if  self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()

        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

    def reset(self, **kwargs):
        self.steps = 0
        if self.reset_type == 'random':
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == 'init':
                qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_done(self):
        height = self._get_height()
        if self._get_goal_reward() == 1:
            return 1
        if height < self.min_height:
            return -1
               
        return 0

    def _get_joint_angle_rew(self, joint_names):
        """
        Penalize extreme joint angles. Unless the agent is kicking
        """
        mag = 0
        joint_angles = self._get_angle(joint_names)
        mag = np.mean(np.abs(joint_angles))
        if self._is_kicking():
            return np.exp(-2 * mag)
        return np.exp(-5 * mag)

    def _is_kicking(self):
        """
        Determines if the agent is in the kick phase based on distance to the ball.
        """
        com_pos = self._get_com()[:2]  # Get agent's center of mass (x, y)
        ball_id = self.sim.model.body_name2id("football")
        ball_pos = self.sim.data.body_xpos[ball_id][:2].copy()  # Ball position (x, y)

        # Compute Euclidean distance
        distance_to_ball = np.linalg.norm(com_pos - ball_pos)

        # Define kick phase when within 1m of the ball
        return distance_to_ball < 1.0

    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        return np.array([self.sim.data.body_xpos[foot_id_l][2], self.sim.data.body_xpos[foot_id_r][2]])

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        pelvis = self.sim.model.body_name2id('pelvis')
        return np.array([self.sim.data.body_xpos[foot_id_l]-self.sim.data.body_xpos[pelvis], self.sim.data.body_xpos[foot_id_r]-self.sim.data.body_xpos[pelvis]])
    
    def _get_vel_reward(self):
        """
        Reward function that prioritizes movement towards the ball
        rather than an arbitrary target velocity.
        """
        vel = self._get_com_velocity()  # (vx, vy)
        speed = np.linalg.norm(vel) + 1e-6  # Avoid division by zero

        #Directional reward (Encourage movement toward the ball
        football_id = self.sim.model.body_name2id("football")
        football_pos = self.sim.data.body_xpos[football_id][:2]  # (x, y)

        com_pos = self._get_com()[:2]  

        dir_to_ball = football_pos - com_pos
        dir_to_ball /= np.linalg.norm(dir_to_ball) + 1e-6  

        # alignment (cosine similarity)
        alignment = np.dot(vel / speed, dir_to_ball)  
        alignment = np.clip(alignment, -1, 1)  

        # Transform alignment to a positive reward (range [0, 4])
        alignment_reward = (alignment + 1) ** 2

        # Speed penalty  (promote stable movement)
        target_speed = np.linalg.norm([self.target_x_vel, self.target_y_vel])
        speed_penalty = np.exp(-0.5 * np.square(target_speed - speed))  

        return alignment_reward * speed_penalty

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        phase_var = (self.steps/self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.float32)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)


    def _get_ref_rotation_rew(self):
        """Reward for facing the ball while keeping an upright posture."""
        # Get positions (3D for proper vector math)
        com_pos = self._get_com()[:3]
        football_id = self.sim.model.body_name2id("football")
        football_pos = self.sim.data.body_xpos[football_id][:3]

        # Yaw Alignment
        dir_to_ball_xy = football_pos[:2] - com_pos[:2]
        dir_to_ball_xy /= np.linalg.norm(dir_to_ball_xy) + 1e-6  # Normalize
        target_yaw = np.arctan2(dir_to_ball_xy[1], dir_to_ball_xy[0])

        # Upright Posture
        target_pitch = 0  # No forward/backward lean
        target_roll = 0   # No side-to-side tilt

        target_rot = R.from_euler('ZYX', [target_yaw, target_pitch, target_roll]).as_quat()
        target_rot = np.roll(target_rot, 1)  # Convert to MuJoCo format [w,x,y,z]

        current_rot = self.sim.data.qpos[3:7]  # Already in MuJoCo format [w, x, y, z]

        cos_theta = np.clip(np.dot(current_rot, target_rot), -1.0, 1.0) 
        full_angle = 2 * np.arccos(cos_theta)  # Angular deviation in radians

        # Posture Penalty 
        current_euler = R.from_quat(current_rot[[1,2,3,0]]).as_euler('ZYX')  # [yaw, pitch, roll]
        pitch_penalty = np.exp(-2 * abs(current_euler[1]))  # Allow slight pitch tilts
        roll_penalty = np.exp(-2 * abs(current_euler[2]))   # Allow slight roll tilts

        ANGLE_THRESH = 0.35  # ~20° total deviation
        if full_angle <= ANGLE_THRESH:
            posture_reward = 0.4 * pitch_penalty + 0.4 * roll_penalty
            yaw_reward = np.exp(-2 * full_angle)
            return 0.5 * yaw_reward + 0.5 * posture_reward 
        else:
            return np.exp(-5 * (full_angle - ANGLE_THRESH))   
    
    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id('torso')
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model. exluding the football and goalpost
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1).copy()
        football_id = self.sim.model.body_name2id("football")
        goalpost_id = self.sim.model.body_name2id("goalpost")
        mass[football_id] = 0
        mass[goalpost_id] = 0
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

    def _get_height(self):
        """
        Get center-of-mass height.
        """
        return self._get_com()[2]

    def _get_com(self):
        """
        Compute the center of mass of the robot. Excluding the football and goalpost.
        """
        mass = np.expand_dims(self.sim.model.body_mass.copy(), -1) 
        com = self.sim.data.xipos.copy()  

        # Exclude football and goalpost by setting their mass to zero
        football_id = self.sim.model.body_name2id("football")
        goalpost_id = self.sim.model.body_name2id("goalpost")
        mass[football_id] = 0
        mass[goalpost_id] = 0

        return np.sum(mass * com, 0) / np.sum(mass)
    
    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]] for name in names])
