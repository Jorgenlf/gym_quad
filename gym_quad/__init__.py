import numpy as np

from gym.envs.registration import register

waypoint_planner_config = {
    "step_size": 0.01,
    "max_t_steps": 100000,
    "min_reward": -20000,
    "n_obs_states": 6,
    "cruise_speed": 2.5,
    "lambda_reward": 0.6,
    "reward_path_following": 4,
    "sensor_span": (90, 90), # the horizontal and vertical span of the sensors
    "sensor_suite": (15, 15), # the number of sensors covering the horizontal and vertical span
    "sensor_input_size": (8, 8), # the shape of FLS data passed to the neural network. Max pooling from raw data is used
    "sensor_frequency": 2,
    "sonar_range": 25,
    "n_obs_errors": 8,
    "n_obs_inputs": 0,
    "la_dist": 3,
    "accept_rad": 1,
    "n_waypoints": 7,
    "n_int_obstacles": 1,
    "n_pro_obstacles": 3,
    "n_adv_obstacles": 8,
    "n_fictive_waypoints": 4,
    "distance_fictive_waypoints": 3.0,
    "n_generated_waypoints": 1,
    "simulation_frequency": 100 # How many timesteps to simulate the quadcopter for each new path
}

register(
    id='WaypointPlanner-v0',
    entry_point='gym_quad.envs:WaypointPlanner',
    kwargs={'env_config': waypoint_planner_config}
)
