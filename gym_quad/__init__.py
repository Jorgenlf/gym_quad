from gymnasium.envs.registration import register
import numpy as np

lv_vae_config = {
    "step_size"             : 0.01, #Step size of the simulation
    "max_t_steps"           : 40000, #Maximum number of timesteps in the simulation before it is terminated
    "min_reward"            : -200, #Minimum reward before the simulation is terminated
    "sensor_span"           : (85, 58), # the horizontal and vertical span of the sensors
    "sensor_suite"          : (15, 15), # the number of sensors covering the horizontal and vertical span
    "sensor_input_size"     : (8, 8), # the shape of FLS data passed to the neural network. Max pooling from raw data is used
    "sensor_frequency"      : 2, # Frequency of the sensors
    "sonar_range"           : 10, # Range of the sonar sensor
    "la_dist"               : 5, # Look ahead distance aka distance to the point on path to be followed
    "accept_rad"            : 1, # Acceptance radius for the quadcopter to consider the end as reached
    "n_waypoints"           : 7, # Number of waypoints to be generated
    "s_max"                 : 2.5, # Maximum speed of the quadcopter 
    "i_max"                 : np.pi/2, # Maximum inclination angle of commanded velocity wrt x-axis
    "r_max"                 : 0.5, # Maximum commanded yaw rate
    'PA_band_edge'          : 4, # edge of Path adherence band
    'PA_scale'              : 2,  # scale of Path adherence reward
    'PP_vel_scale'          : 1, # scale of velocity reward
    'PP_rew_max'            : 2.5, # maximum reward for path progression
    'PP_rew_min'            : -1, # minimum reward for path progression
    'rew_collision'         : -50, # reward for collision
    'rew_reach_end'         : 30, # reward for reaching the end of the path
    'existence_reward'      : -0.0001, # reward for existing increments with the value defined here per timestep
    'danger_range'          : 150, # Range between quadcopter and obstacle within which the quadcopter is in danger
    'danger_angle'          : 20, # Angle between quadcopter and obstacle within which the quadcopter is in danger
    'abs_inv_CA_min_rew'    : 1/8, #1/2 -> -2 is min reward per CA fcn range and angle --> rangefcn + anglefcn = -4
}

register(
    id='LV_VAE-v0',
    entry_point='gym_quad.envs:LV_VAE',
    kwargs={'env_config': lv_vae_config}
)







#OLD vv
waypoint_planner_config = {
    "step_size": 0.01,
    "max_t_steps": 60000,
    "min_reward": -20000,
    "n_obs_states": 6,
    "cruise_speed": 2.5,
    "lambda_reward": 0.5,
    "reward_path_following_c": 5,
    "reward_path_following_max": 1,
    "sensor_span": (360, 180), # the horizontal and vertical span of the sensors
    "sensor_suite": (15, 15), # the number of sensors covering the horizontal and vertical span
    "sensor_input_size": (8, 8), # the shape of FLS data passed to the neural network. Max pooling from raw data is used
    "sensor_frequency": 2,
    "sonar_range": 25,
    "n_obs_errors": 8,
    "n_obs_inputs": 0,
    "la_dist": 5,
    "accept_rad": 1,
    "n_waypoints": 7,
    "n_int_obstacles": 5,
    "n_pro_obstacles": 3,
    "n_adv_obstacles": 8,
    "n_fictive_waypoints": 1,
    "fictive_waypoint_span": 10,
    "n_generated_waypoints": 1,
    "simulation_frequency": 10, # How many timesteps to simulate the quadcopter for each new path
    "rl_mode": "path_planning" # [path_planning, desired_acc]
}

register(
    id='WaypointPlanner-v0',
    entry_point='gym_quad.envs:WaypointPlanner',
    kwargs={'env_config': waypoint_planner_config}
)