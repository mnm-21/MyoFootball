# MyoFootball
This repository contains a **custom football environment** built using **MyoSuite** and **MuJoCo**. It simulates a **football-playing musculoskeletal agent**, encouraging sports-like behaviors such as locomotion, ball control, kicking, and goal-scoring.

## **Environment Overview**
Consists of a Musculoskeletal from Myosuite(myolegs.xml), a football and a goalpost. A custom reward function consists of multiple terms that guide the agent’s learning:

- **vel_reward (5.0)** – Encourages the agent to move efficiently towards the ball.  
- **done (-100.0)** – Penalizes the agent when an episode ends unsuccessfully due to falling.  
- **cyclic_hip (-8.0)** – Penalizes unnatural gait .  
- **ref_rot (10.0)** – Rewards the agent for maintaining orientation in the ball direction.  
- **joint_angle_rew (5.0)** – Encourages maintaining optimal joint angles for stable movement.  
- **in_goal (200.0)** – Large reward for successfully getting the ball into the goal.  
- **kick (20.0)** – Rewards the agent for making successful contact with the ball.  
- **kick_force (120.0)** – Higher reward for applying greater force in the right direction.  
- **ball_vel_reward (10.0)** – Rewards faster ball movement towards the goal (e.g., a 30 m/s shot perfectly aligned gives 120 reward).
  
## License and Acknowledgments

This project uses **MuJoCo** and **MyoSuite**, which are licensed under the following terms:

### **MuJoCo**
MuJoCo (Multi-Joint dynamics with Contact) is a physics engine for model-based control.  
It is licensed under the **Apache License 2.0**.  
- **Website:** [https://mujoco.org](https://mujoco.org)  
- **License:** [MuJoCo License](https://github.com/google-deepmind/mujoco/blob/main/LICENSE)  

### **MyoSuite**
MyoSuite is a collection of musculoskeletal environments for reinforcement learning, developed by Meta AI.  
It is licensed under the **Apache License 2.0**.  
- **GitHub:** [https://github.com/facebookresearch/myosuite](https://github.com/facebookresearch/myosuite)  
- **License:** [MyoSuite License](https://github.com/facebookresearch/myosuite/blob/main/LICENSE)  

The use of MuJoCo and MyoSuite in this project does not imply any affiliation with or endorsement by their creators.

---
