import collections
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat
from scipy.spatial.transform import Rotation as R


#  //  python -m deprl.main myoFootball.json
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
        'kicked', #v2
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        #"vel_reward": 5.0, #v2
        "done": 100.0, # multiplied by -3 if agent falls and +1 if goal is scored
        #"act_reg": 1.0,
        #"cyclic_hip": -10.0, #v2
        "yaw_align": 10.0,
        "posture_rew": 25.0,
        #"ref_rot": 5.0,  #v2
        "joint_angle_rew": 5.0,
        "in_goal": 400.0,
        "kick": 4.0,
        "kick_force": 10.0,
        "ball_vel_reward": 15.0,
        "dist_from_ball": 8.0, 
        "post_kick_time": 1.0,  
        "ball_to_goal_reward": 20.0, 
        "torso_pelvis_vel": 15.0,
        "feet_contact_rew": 5.0,
        "root_change": 2.5,
        "com_height": 20.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        self.prev_ball_distance = 0
        self.kick_step = 0
        #self.kicked = kwargs.pop("kicked", 0)
        self.kicked = 0
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        #self.kicked = 0 if self._is_kicking() else 1 # set kicked to false if in kick region else true

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
        obs_dict['kicked'] = np.array([self.kicked])
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        # if self.steps - self.kick_step > 100: # give 100 timesteps for ball to move away from agent to prevent rapid transitions
        #     self.kicked = 0 if self._is_kicking() else 1 # set kicked to false if in kick region else true
        vel_reward = self._get_vel_reward()
        #cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l',
                                                       'hip_rotation_r'])
        #act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0 
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        act_mag = act_mag*10.0
        rwd_dict = collections.OrderedDict((
            
            #('vel_reward', vel_reward),  
            #('cyclic_hip',  cyclic_hip),                      
            # ('ref_rot',  ref_rot),
            ('joint_angle_rew', joint_angle_rew),
            ('act_reg', act_mag), #
            ('sparse',  vel_reward), #
            ('solved',    vel_reward >= 1.0),#
            ('done',  self._get_done()),
            ('in_goal', self._get_goal_reward()),
            ('kick', self._get_kick_reward()),
            ('kick_force', self._get_kick_force()),
            ('ball_vel_reward', self._get_ball_vel_reward()),
            ('dist_from_ball', self._get_dist_from_ball()),
            ('post_kick_time', self._get_post_kick_time_reward()),
            ('ball_to_goal_reward', self._get_ball_to_goal_reward()),
            ('com_height', self._com_height_reward()),
            ('torso_pelvis_vel',self._torso_pevlis_velocity()),
            ('feet_contact_rew', self._feet_contact()),
            ('yaw_align', self._get_yaw_alignment_rew()),
            ('posture_rew', self._get_posture_rew()),
            ('root_change', self._get_root_change()),
            ('dense', 0.0),

        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0) - act_mag
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
    
    def _get_dist_from_ball(self):

        if self.kicked:
            return 0
        
        # Get ball position
        ball_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("football")].copy()

        # Get positions of both feet
        left_foot_pos = self.sim.data.geom_xpos[self.sim.model.geom_name2id("l_foot")].copy()
        right_foot_pos = self.sim.data.geom_xpos[self.sim.model.geom_name2id("r_foot")].copy()

        dist_left = np.linalg.norm(ball_pos - left_foot_pos)
        dist_right = np.linalg.norm(ball_pos - right_foot_pos)

        # Take the minimum distance (closer foot)
        min_dist = min(dist_left, dist_right)

        # Reward shaping: Negative distance to encourage proximity
        return -min_dist
    
    def _get_ball_to_goal_reward(self):
        """
        Reward for moving the ball closer to the goal.
        Encourages reducing the distance between the ball and the goal over time.
        """
        if self.steps == 0:
            self.prev_ball_distance = np.linalg.norm(self.sim.data.body_xpos[self.sim.model.body_name2id("football")].copy() - self.sim.data.body_xpos[self.sim.model.body_name2id("goalpost")].copy())

        ball_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("football")].copy()

        goal_center = self.sim.data.body_xpos[self.sim.model.body_name2id("goalpost")].copy()
        
        # Compute Euclidean distance from the ball to the goal center
        current_distance = np.linalg.norm(ball_pos[:2] - goal_center[:2])  

        # Compute distance change (improvement)
        distance_change = self.prev_ball_distance - current_distance  # Positive if moving closer
        stagnation_penalty = -0.05 if np.abs(distance_change) < 1e-3 else 0.0
        # Reward based on improvement
        reward = np.tanh(20.0 * distance_change)  #scales between -1 and 1

        self.prev_ball_distance = current_distance

        return reward + stagnation_penalty

    def _com_height_reward(self):
        com_height = self._get_height()
        target_height = 0.94
        # if com height in +- 0.2 of target height no penalty
        # if com_height > target_height - 0.05 :
        #     return 1
        reward = np.exp(-20 * (com_height - target_height)**2)
        return reward
    def _get_kick_reward(self):

        football_geom_id = self.sim.model.geom_name2id("football_geom")
        foot_geom_ids = [self.sim.model.geom_name2id(name) for name in [
            "l_foot_col1", "r_foot_col1",  
            "l_foot_col3", "r_foot_col3",  
            "l_foot_col4", "r_foot_col4",  
            "l_bofoot_col1", "r_bofoot_col1",   
            "l_bofoot_col2", "r_bofoot_col2",  
            "l_talus", "r_talus",  # Ankle
            "l_foot", "r_foot",  # Full foot
            "l_bofoot", "r_bofoot"
        ]]
        reward = 0

        for contact in self.sim.data.contact:
            if (contact.geom1 == football_geom_id and contact.geom2 in foot_geom_ids) or (contact.geom2 == football_geom_id and contact.geom1 in foot_geom_ids):
                reward = 1  # Assign reward for a successful kick
                self.kicked = 1 # Set kicked flag to True
                self.kick_step = self.steps 
                break  # No need to check further if a valid contact is found
        return reward

    def _get_kick_force(self):
        """
        Computes the reward based on the force applied by the foot to the ball.
        Rewards stronger and better-aligned kicks.
        """
        if self._get_kick_reward() == 0:
            return 0
        
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
        contact_force = np.zeros(6, dtype=np.float64)  # Ensure correct shape

        # Sum forces across all contacts
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if (contact.geom1 == football_geom_id and contact.geom2 in foot_geom_ids) or \
            (contact.geom2 == football_geom_id and contact.geom1 in foot_geom_ids):

                mujoco.mj_contactForce(self.sim.model.ptr, self.sim.data.ptr, i, contact_force)  # Use .ptr

                if contact.geom1 == football_geom_id:
                    # Force exerted by football on foot; negate to get foot on football
                    total_force_vec += -contact_force[:2]  # Only x, y components
                else:
                    # Force exerted by foot on football; use as is
                    total_force_vec += contact_force[:2]  # Only x, y components

        # Compute total force magnitude
        total_force_mag = np.linalg.norm(total_force_vec)

        # Force Alignment with Goal (x, y only)
        ball_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("football")][:2]  
        goal_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("goalpost")][:2]  
        
        goal_dir_vec = goal_pos - ball_pos  # Vector from ball to goal
        goal_dir_norm = np.linalg.norm(goal_dir_vec) + 1e-6  # Avoid division by zero
        goal_dir_vec /= goal_dir_norm  # Normalize

        # dot product between force direction and goal direction
        if total_force_mag > 1e-6:  # Avoid division by zero
            force_alignment = np.dot(total_force_vec / total_force_mag, goal_dir_vec)
        else:
            force_alignment = 0.0  # No contact force, so no meaningful alignment

        force_alignment = np.clip(force_alignment, -1, 1)  # Ensure valid range

        force_reward = np.exp(0.002 * total_force_mag)  
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
            return -3
               
        return 0

    def _get_joint_angle_rew(self, joint_names):
        """
        Penalize extreme joint angles.
        """
        mag = 0
        joint_angles = self._get_angle(joint_names)
        mag = np.mean(np.abs(joint_angles))
        return (1 - np.exp(4 * mag))
         

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
        return distance_to_ball < 0.4

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
        Reward the agent for moving in the direction of the ball after kicking it
        """
        if self.kicked==0:
            return 0
    
        vel = self._get_com_velocity()  # (vx, vy)
        speed = np.linalg.norm(vel) + 1e-6  # Avoid division by zero

        #Directional reward (Encourage movement toward the ball
        football_id = self.sim.model.body_name2id("football")
        football_pos = self.sim.data.body_xpos[football_id][:2]  # (x, y)

        com_pos = self._get_com()[:2]  

        #  direction to football
        dir_to_ball = football_pos - com_pos
        dir_to_ball /= np.linalg.norm(dir_to_ball) + 1e-6  

        # alignment (cosine similarity)
        alignment = np.dot(vel / speed, dir_to_ball)  
        alignment = np.clip(alignment, -1, 1)  

        # Transform alignment to a positive reward (range [0, 4])
        alignment_reward = (alignment + 1) ** 2

        # --- Speed penalty (optional) ---
        target_speed = np.linalg.norm([self.target_x_vel, self.target_y_vel])
        speed_penalty = np.exp(-0.5 * np.square(target_speed - speed))  

        return alignment_reward * speed_penalty

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        if self.kicked==0:
            return 0
        phase_var = ((self.steps)/self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.float32)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)

    def _get_yaw_alignment_rew(self):
        """Reward for aligning yaw direction with the ball."""
        if self.kicked:
            target_rot = [self.target_rot if self.target_rot is not None else self.init_qpos[3:7]][0]
            return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))
            
        com_pos = self._get_com()[:2]  # Center of mass (XY plane)
        football_id = self.sim.model.body_name2id("football")
        football_pos = self.sim.data.body_xpos[football_id][:2]  # Ball position (XY plane)

        # Compute target yaw angle
        dir_to_ball_xy = football_pos - com_pos
        norm = np.linalg.norm(dir_to_ball_xy)
        if norm < 1e-4:  # Avoid division by zero
            return 0.0
        target_yaw = np.arctan2(dir_to_ball_xy[1], dir_to_ball_xy[0])

        # Extract torso yaw angle using _get_torso_angle()
        torso_quat = self._get_torso_angle()  # [w, x, y, z]
        current_yaw = R.from_quat(torso_quat[[1, 2, 3, 0]]).as_euler('ZYX')[0]  # Extract yaw

        # Compute absolute yaw error
        yaw_error = abs(target_yaw - current_yaw)
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]

        # Reward based on yaw alignment
        ANGLE_THRESH = np.deg2rad(10)  # ~10-degree threshold
        if yaw_error <= ANGLE_THRESH:
            return 1.0 - (yaw_error / ANGLE_THRESH)
        else:
            return np.exp(-4 * (yaw_error - ANGLE_THRESH))  # Exponential decay for large deviations

    def _get_posture_rew(self):
        """Penalty for pitch and roll deviations based on the torso orientation."""
        torso_quat = self._get_torso_angle()  

        # Convert torso quaternion to Euler angles
        current_euler = R.from_quat(torso_quat[[1, 2, 3, 0]]).as_euler('ZYX')  # [yaw, pitch, roll]
        
        pitch_dev = abs(current_euler[1])
        roll_dev = abs(current_euler[2])

        return np.exp(-5 * (pitch_dev**2 + 1.5 * roll_dev**2))
        

    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        # if self.kicked==0 or ((self.steps - self.kick_step)<25):
        #     return 0
        target_rot = [self.target_rot if self.target_rot is not None else self.init_qpos[3:7]][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))
    
    # define a function to penalise too much change in pelvis and torso coordinates
    def _get_root_change(self):
        root_pos = self.sim.data.qpos[:3]
        root_change = np.linalg.norm(root_pos - self.init_qpos[:3])
        # exponential penalty
        return 1-np.exp(3 * root_change)
        
    def _feet_contact(self):
        # Get touch sensor readings (contact forces for each foot)
        
        touch_sensor_names = ["r_foot", "r_toes", "l_foot", "l_toes"]
        touch_sensors = np.array([self.sim.data.sensor(sens_name).data[0].copy() for sens_name in touch_sensor_names])
        right_foot_contact = touch_sensors[0] + touch_sensors[1]
        left_foot_contact = touch_sensors[2] + touch_sensors[3]
        both_feet_contact = 1.0 if right_foot_contact > 0 and left_foot_contact > 0 else 0.0
        if self.kicked==0 or ((self.steps - self.kick_step)<15):
            return 1.0 if left_foot_contact > 0 or right_foot_contact > 0 else 0.0
        return both_feet_contact
    
    def _torso_pevlis_velocity(self):
        
        torso_vel = self.sim.data.cvel[self.sim.model.body_name2id('torso')]
        pelvis_vel = self.sim.data.cvel[self.sim.model.body_name2id('pelvis')]
        velocity_penalty = -0.1 * (np.linalg.norm(pelvis_vel) + np.linalg.norm(torso_vel))
        return velocity_penalty

    def _get_post_kick_time_reward(self):
        """
        Reward for surviving after a successful kick.
        """
        if self.kicked==0 or (self.steps - self.kick_step<=1): 
            return 0 

        time_bonus = 0.01 * (self.steps - self.kick_step)

        return time_bonus
    
    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id('torso')
        return self.sim.data.body_xquat[body_id].copy()

    def _quat_to_euler(self, quat):
        """
        Convert a quaternion (w, x, y, z) to roll, pitch (Euler angles).
        """
        w, x, y, z = quat

        # Roll (rotation around x-axis)  
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (rotation around y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))  # Avoid NaN issues

        return roll, pitch

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
        mass = np.expand_dims(self.sim.model.body_mass.copy(), -1)  # Shape: (num_bodies, 1)
        com = self.sim.data.xipos.copy()  # Shape: (num_bodies, 3)

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