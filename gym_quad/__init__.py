from gymnasium.envs.registration import register
import numpy as np

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

lv_vae_config = {
#General parameters    
    "step_size"                 : 0.01,         # Step size of the simulation
    "max_t_steps"               : 30000,        # Maximum number of timesteps in the simulation before it is terminated
    "mesh_path"                 : "./gym_quad/meshes/sphere.obj", # Path to the mesh of the sphere obstacle
    "enclose_scene"             : True,        # Enclose the scene with a box thats scaled to the scene size
    "padding"                   : 8,           # Padding of the box that encloses the scene [m]
    "drone_radius_for_collision": 0.3,         # Radius of the drone for collision detection [m]
#Depth camera parameters    
    "FOV_vertical"              : 75,           # Vertical field of view of the depth camera
    "FOV_horizontal"            : 62,           # Horizontal field of view of the depth camera
    "depth_map_size"            : (240, 320),   # Size of the depth map Earlier sensor suite
    "max_depth"                 : 10,           # Maximum depth of the depth camera
    "camera_FPS"                : 15,           # Frequency of the sensors
#VAE parameters    
    "compressed_depth_map_size" : 224,          # Size of depth map after compression
    "latent_dim"                : 32,           # Dimension of the latent space
#Path planner parameters
    "la_dist"                   : 20,            # Look ahead distance aka distance to the point on path to be followed
    "accept_rad"                : 5,            # Acceptance radius for the quadcopter to consider the end as reached
    "n_waypoints"               : 4,            # Number of waypoints to be generated
#Drone controller parameters
    "s_max"                     : 2.5,          # Maximum speed of the quadcopter m/s
    "i_max"                     : deg2rad(80/2),      # Maximum inclination angle of commanded velocity wrt x-axis #TODO decide this now set it to half of vertical sensor span
    "r_max"                     : deg2rad(30),          # Maximum commanded yaw rate rad/s
    "kv"                        : 2.5,          # Velocity gain             Been for long time: 2.5
    "kangvel"                   : 0.8,          # Angular velocity gain     Been for long time: 0.8
    "kR"                        : 0.8,          # Attitude gain             Been for long time: 0.8
#Reward parameters
    "min_reward"                : -1e4,         # Minimum reward before the simulation is terminated
    'PA_band_edge'              : 10,            # edge of Path adherence band
    'PA_scale'                  : 3,            # scale of Path adherence reward [-PA_scale, PA_scale]
    'PP_vel_scale'              : 0.7,          # scaling of velocity reward e.g. 1-> make 2.5m/s
    'PP_rew_max'                : 2.5,          # maximum reward for path progression
    'PP_rew_min'                : -1,           # minimum reward for path progression
    'rew_collision'             : -50,          # reward for collision
    'rew_reach_end'             : 30,           # reward for reaching the end of the path
    'existence_reward'          : -0.005,       # reward for existing
    'danger_range'              : 10,          # Range between quadcopter and obstacle within which the quadcopter is in danger #TODO change this to the max_depth?
    'danger_angle'              : 20,           # Angle between quadcopter and obstacle within which the quadcopter is in danger #TODO this is outdated for mesh env unless we get global distances somehow
    'abs_inv_CA_min_rew'        : 1/16,          #1/x -> -x is min reward per CA fcn range and angle --> rangefcn + anglefcn = -2*x 
    #TODO Make it take in 16 instead of 1/16 and invert it in the function also no longer angle involved so remove that
}

register(
    id='LV_VAE-v0',
    entry_point='gym_quad.envs:LV_VAE',
    kwargs={'env_config': lv_vae_config}
)

register(
    id='LV_VAE_MESH-v0',
    entry_point='gym_quad.envs:LV_VAE_MESH',
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