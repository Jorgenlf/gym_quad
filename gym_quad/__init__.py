import numpy as np

from gym.envs.registration import register


pid_pathcolav_config = {
    "step_size": 0.01,#0.05,#0.10,
    "max_t_steps": 100000,#10000,#4000,
    "min_reward": -20000,#-2000,#-2000,#1000,
    "n_obs_states": 9,#12
    "cruise_speed": 1.5,
    "lambda_reward": 0.6,
    "reward_roll": -1,
    "reward_rollrate": -1,
    "reward_control_derivative": [-0.005, -0.005],
    "reward_heading_error": -1,
    "reward_crosstrack_error": -0.0001,
    "reward_pitch_error": -1,
    "reward_verticaltrack_error": -0.0001,
    "reward_collision": 0,
    "sensor_span": (360,180),#(140,140), # the horizontal and vertical span of the sensors
    "sensor_suite": (15, 15), # the number of sensors covering the horizontal and vertical span
    "sensor_input_size": (8,8), # the shape of FLS data passed to the neural network. Max pooling from raw data is used
    "sensor_frequency": 2,#1,
    "sonar_range": 25,
    "n_obs_errors": 8,#4,
    "n_obs_inputs": 0,
    "n_actuators": 4,
    "la_dist": 3,
    "accept_rad": 1,
    "n_waypoints": 7,
    "n_int_obstacles": 1,
    "n_pro_obstacles": 3,
    "n_adv_obstacles": 8
}

register(
    id='PathColav3d-v0',
    entry_point='gym_quad.envs:PathColav3d',
    kwargs={'env_config': pid_pathcolav_config}
)

waypoint_planner_config = {
    "step_size": 0.01,
    "max_t_steps": 100000,
    "min_reward": -20000,
    "n_obs_states": 9,
    "cruise_speed": 2.5,
    "lambda_reward": 0.6,
    "reward_rollrate": -1,
    "reward_control_derivative": [-0.005, -0.005],
    "reward_heading_error": -1,
    "reward_crosstrack_error": -0.0001,
    "reward_pitch_error": -1,
    "reward_verticaltrack_error": -0.0001,
    "reward_collision": 0,
    "sensor_span": (360,180), # the horizontal and vertical span of the sensors
    "sensor_suite": (15, 15), # the number of sensors covering the horizontal and vertical span
    "sensor_input_size": (8,8), # the shape of FLS data passed to the neural network. Max pooling from raw data is used
    "sensor_frequency": 2,
    "sonar_range": 25,
    "n_obs_errors": 8,
    "n_obs_inputs": 0,
    "n_actuators": 4,
    "la_dist": 3,
    "accept_rad": 1,
    "n_waypoints": 7,
    "n_int_obstacles": 1,
    "n_pro_obstacles": 3,
    "n_adv_obstacles": 8
}

register(
    id='WaypointPlanner-v0',
    entry_point='gym_quad.envs:WaypointPlanner',
    kwargs={'env_config': waypoint_planner_config}
)
