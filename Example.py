from myosuite.utils import gym

# Add the following lines to /Users/mayankchandak/miniconda3/envs/MyoSuite/lib/python3.8/site-packages/myosuite/envs/myo/myobase/__init__.py file to register env

# register(id='Football-v0',
#     entry_point='myosuite.envs.myo.myobase.football_v0:FootballEnvV0',
#     max_episode_steps=2000, 
#     kwargs={
#         'model_path': "/Users/mayankchandak/miniconda3/envs/MyoSuite/lib/python3.8/site-packages/myosuite/simhive/myo_sim/leg/myolegs_football.xml", 
#         'normalize_act': True,
#         'frame_skip': 5,
#     }
# )

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
# Observation shape: (426,) # same as MyoLegWalk-v0(403) + football(qpos+qvel=13)+football_pos(3) and football_vel(6) + "kicked"

print("Action shape:", action.shape)
# Action shape: (80,) # same as MyoLegWalk-v0

print("Observation Keys:", env.get_obs_dict(env.sim).keys())
# Observation Keys: dict_keys(['t', 'time', 'qpos_without_xy', 'qvel', 'com_vel', 'torso_angle', 'feet_heights', 'height', 'feet_rel_positions', 'phase_var', 'muscle_length', 'muscle_velocity', 'muscle_force', 'football_pos', 'football_vel', 'act'])

print("Reward Keys:", env.rwd_keys_wt.items())
# Reward Keys: dict_items([('done', 100.0), ('yaw_align', 10.0), ('posture_rew', 25.0), ('joint_angle_rew', 5.0), ('in_goal', 400.0), ('kick', 4.0), ('kick_force', 10.0), ('ball_vel_reward', 15.0), ('dist_from_ball', 8.0), ('post_kick_time', 1.0), ('ball_to_goal_reward', 20.0), ('torso_pelvis_vel', 15.0), ('feet_contact_rew', 5.0), ('root_change', 2.5), ('com_height', 20.0)])

env.close()
