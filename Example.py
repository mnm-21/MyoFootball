from myosuite.utils import gym
# Add the following lines to /Users/mayankchandak/miniconda3/envs/MyoSuite/lib/python3.8/site-packages/myosuite/envs/myo/myobase/__init__.py file to register env
# register = gym.register
# register(id='Football-v0',
#     entry_point='myosuite.envs.myo.myobase.football_v0:FootballEnvV0',
#     max_episode_steps=1000, 
#     kwargs={
#         'model_path': "/Users/mayankchandak/miniconda3/envs/MyoSuite/lib/python3.8/site-packages/myosuite/simhive/myo_sim/leg/myolegs_football.xml", 
#     }
# )

import time

env = gym.make("Football-v0")
env.reset()
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.mj_render()
    #time.sleep(0.01)
    if done:
        env.reset()

print("Observation shape:", obs.shape) 
# Observation shape: (425,) # same as MyoLegWalk-v0(403) + football(qpos+qvel=13)+football_pos(3) and football_vel(6)

print("Action shape:", action.shape)
# Action shape: (80,) # same as MyoLegWalk-v0

print("Observation Keys:", env.get_obs_dict(env.sim).keys())
# Observation Keys: dict_keys(['t', 'time', 'qpos_without_xy', 'qvel', 'com_vel', 'torso_angle', 'feet_heights', 'height', 'feet_rel_positions', 'phase_var', 'muscle_length', 'muscle_velocity', 'muscle_force', 'football_pos', 'football_vel', 'act'])

print("Reward Keys:", env.get_reward_dict(env.get_obs_dict(env.sim)).keys())
# Reward Keys: odict_keys(['vel_reward', 'cyclic_hip', 'ref_rot', 'joint_angle_rew', 'act_mag', 'sparse', 'solved', 'done', 'in_goal', 'kick', 'kick_force', 'ball_vel_reward', 'dense'])

env.close()
