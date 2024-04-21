import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import gym_quad.utils.geomutils as geom
import gym_quad.utils.state_space as ss
from gym_quad.objects.quad import Quad
from gym_quad.objects.IMU import IMU
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints
from gym_quad.objects.depth_camera import *

#TODO Set up curriculum learning
#TODO add stochasticity to make sim2real robust

class LV_VAE(gym.Env):
    '''Creates an environment where the actionspace consists of Linear velocity and yaw rate which will be passed to a PD or PID controller,
    while the observationspace uses a Varial AutoEncoder "plus more" for observations of environment.'''

    def __init__(self, env_config, scenario="line"):
        # np.random.seed(0) #Uncomment to make the environment deterministic

        # Set all the parameters from GYM_QUAD/qym_quad/__init__.py as attributes of the class
        for key in env_config:
            setattr(self, key, env_config[key])

        #Actionspace mapped to speed, inclination of velocity vector wrt x-axis and yaw rate
        self.action_space = gym.spaces.Box(
            low = np.array([-1,-1,-1], dtype=np.float32),
            high = np.array([1, 1, 1], dtype=np.float32),
            dtype = np.float32
        )

        #Observationspace
        #Depth camera observation space
        self.perception_space = gym.spaces.Box( #TODO 2x check the shape and type as this is a tensor want it to be a nice tensor for quick processing
            low = 0,
            high = 1,
            shape = (1, self.compressed_depth_map_size, self.compressed_depth_map_size),
            dtype = np.float32
        )

        # IMU observation space
        self.IMU_space = gym.spaces.Box(
            low = -1,
            high = 1,
            shape = (6,),
            dtype = np.float32
        )

        #Domain observation space (Angles, distances and coordinates in body frame)
        self.domain_space = gym.spaces.Box(
            low = -1,
            high = 1,
            shape = (16,),
            dtype = np.float32
        )

        self.observation_space = gym.spaces.Dict({
        'perception': self.perception_space,
        'IMU': self.IMU_space,
        'domain': self.domain_space
        })

        #Scenario set up
        self.scenario = scenario
        self.obstacles = [] #Filled in the scenario functions
        self.scenario_switch = {
            # Training scenarios, all functions defined at the bottom of this file
            "line": self.scenario_line,
            "line_new": self.scenario_line_new,
            "horizontal": self.scenario_horizontal,
            "horizontal_new": self.scenario_horizontal_new,
            "3d": self.scenario_3d,
            "3d_new": self.scenario_3d_new,
            "helix": self.scenario_helix,
            "intermediate": self.scenario_intermediate,
            "proficient": self.scenario_proficient,
            # "advanced": self.scenario_advanced,
            "expert": self.scenario_expert,
            # Testing scenarios
            "test_path": self.scenario_test_path,
            "test": self.scenario_test,
            "horizontal": self.scenario_horizontal_test,
            "vertical": self.scenario_vertical_test,
            "deadend": self.scenario_deadend_test
        }

        #New init values for sensor using depth camera, mesh and pt3d
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #Attempt to use GPU if available

        #Reset environment to init state
        self.reset()


    def reset(self,**kwargs):
        """
        Resets environment to initial state.
        """
        seed = kwargs.get('seed', None)
        super().reset(seed=seed)
        print("PRINTING SEED WHEN RESETTING:", seed) 
        
        self.quadcopter = None
        self.path = None
        self.path_generated = None
        self.e = None
        self.h = None
        self.chi_error = None
        self.upsilon_error = None
        self.waypoint_index = 0
        self.prog = 0
        self.success = False
        self.done = False
        self.LA_at_end = False
        self.cumulative_reward = 0

        #Obstacle variables
        self.nearby_obstacles = []
        self.obstacles = []
        self.collided = False

        self.depth_map = torch.zeros((1, self.depth_map_size[0], self.depth_map_size[1]), dtype=torch.float32, device=self.device)
        

        self.prev_position_error = [0, 0, 0]
        self.total_position_error = [0, 0, 0]

        self.passed_waypoints = np.zeros((1, 3), dtype=np.float32)
        self.total_t_steps = 0

        ### Path and obstacle generation based on scenario
        scenario = self.scenario_switch.get(self.scenario, lambda: print("Invalid scenario"))
        init_state = scenario() #Called such that the obstacles are generated


        ## Regenerate camera, scene and renderer
        camera = FoVPerspectiveCameras(device = self.device,fov=self.FOV_vertical)
        raster_settings = RasterizationSettings(
                image_size=self.depth_map_size, 
                blur_radius=0.0, 
                faces_per_pixel=1, # Keep at 1, dont change
                perspective_correct=True, # Doesn't do anything(??), but seems to improve speed
                cull_backfaces=True # Do not render backfaces. MAKE SURE THIS IS OK WITH THE GIVEN MESH.
            )
        
        if self.obstacles!=[]:
            scene = SphereScene(device = self.device, sphere_obstacles=self.obstacles)
        else:
            scene = SphereScene(device = self.device)
        
        self.renderer = DepthMapRenderer(device=self.device, 
                                         raster_settings=raster_settings, 
                                         camera=camera, 
                                         scene=scene, 
                                         MAX_MEASURABLE_DEPTH=self.max_depth, 
                                         img_size=self.depth_map_size)



        # Generate Quadcopter
        self.quadcopter = Quad(self.step_size, init_state)
        
        
        ###
        self.info = {}
        self.imu = IMU()
        self.update_errors()
        self.observation = self.observe() 
        return (self.observation,self.info)


    def observe(self): 
        """
        Returns observations of the environment.
        """
        #For saving and plotting
        pure_obs = [] 

        #IMU measurements
        imu_measurement = np.zeros((6,), dtype=np.float32) 
        # I suppose this can be turned into a self variable and inited once in the init or reset fcn
        #What is the best convention have many self variables to init once or have them in the fcn and init them every time? #TODO
        if self.total_t_steps > 0:
            imu_measurement = self.imu.measure(self.quadcopter)

        #The linear acceleration is not in [-1,1] clipping it using the max speed of the quadcopter
        imu_measurement[0:3] = self.m1to1(imu_measurement[0:3], -self.s_max*2, self.s_max*2)
        #The angular velocity is not in [-1,1] clipping it using the max yaw rate of the quadcopter
        imu_measurement[3:6] = self.m1to1(imu_measurement[3:6], -self.r_max*2, self.r_max*2)
        # print(np.round(imu_measurement,2))

        #Ensure dtype is float32
        imu_measurement = imu_measurement.astype(np.float32)

        pure_obs.extend(imu_measurement)

        #Depth camera observation
        if self.total_t_steps % (int((1/self.camera_FPS)/self.step_size)) == 0: #Only update the depth map at the camera FPS
            pos = self.quadcopter.position
            orientation = self.quadcopter.attitude
            Rcam,Tcam = self.renderer.camera_R_T_from_quad_pos_orient(pos, orientation)
            self.renderer.update_R(Rcam)
            self.renderer.update_T(Tcam)
            self.depth_map = self.renderer.render_depth_map()
            temp_depth_map = self.depth_map

            # temp_path = "gym_quad/envs/temp_depth_test_img" #Uncomment to save depth map images for testing
            # self.renderer.save_depth_map(f"{temp_path}/depth_map_{self.total_t_steps}", self.depth_map)
            # print("\nTemp depth map type:", type(temp_depth_map), "  shape:", temp_depth_map.shape, "  dtype:", temp_depth_map.dtype)
        else:
            temp_depth_map = self.depth_map #Use the previous depthmap 
        
        normalized_depth_map = temp_depth_map / self.max_depth
        
        #New
        normalized_depth_map_PIL = transforms.ToPILImage()(normalized_depth_map)

        resize_transform = transforms.Compose([
            transforms.Resize((self.compressed_depth_map_size, self.compressed_depth_map_size)),
            transforms.ToTensor(),  # Convert back to tensor
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
        ])        

        resized_depth_map = resize_transform(normalized_depth_map_PIL)

        # sensor_readings = resized_depth_map #Might rename sensorreadings to comp_normed_depth_map or VAE_ready_depth_map
        #TODO throws warnings about wanting a np.array instead of a tensor
        #Decide if we let the box change to np.array or if we change the tensor to a np.array here.
        #Migh be unfortunate to change the tensor to np.array here as it will be done every time the observation is called
        #Having to move the tensor to the cpu....
        #Per now we cast to np.array here
        sensor_readings = resized_depth_map.detach().cpu().numpy()
        if max(sensor_readings.flatten()) > 1 or min(sensor_readings.flatten()) < 0:
            print("\nMAX VALUE IN SENSORREADINGS:",max(sensor_readings.flatten()),"\nMIN VALUE IN SENSORREADINGS:", min(sensor_readings.flatten()))


        # print("Sensor readings type:", type(sensor_readings), "  shape:", sensor_readings.shape, "  dtype:", sensor_readings.dtype)

        #Domain observation
        domain_obs = np.zeros(16, dtype=np.float32)
        # Heading angle error wrt. the path
        domain_obs[0] = np.sin(self.chi_error)
        domain_obs[1] = np.cos(self.chi_error)
        # Elevation angle error wrt. the path
        domain_obs[2] = np.sin(self.upsilon_error)
        domain_obs[3] = np.cos(self.upsilon_error)
         
        pure_obs.extend([self.chi_error, self.upsilon_error])

        # # Angle to velocity vector from body frame #PROBS NOT NEEDED :) #TODO
        # domain_obs[4] = np.sin(self.quadcopter.aoa) #angle of attack
        # domain_obs[5] = np.cos(self.quadcopter.aoa)
        # domain_obs[6] = np.sin(self.quadcopter.beta) #sideslip angle
        # domain_obs[7] = np.cos(self.quadcopter.beta)
        
        # x y z of closest point on path in body frame
        relevant_distance = 20 #For this value and lower the observation will be changing i.e. giving info if above or below its clipped to -1 or 1 
        #TODO make this a hypervariable or make it dependent on e.g. the scene
        x,y,z = self.quadcopter.position
        closest_point = self.path.get_closest_position([x,y,z], self.waypoint_index)
        closest_point_body = np.transpose(geom.Rzyx(*self.quadcopter.attitude)).dot(closest_point - self.quadcopter.position)
        pure_obs.extend(closest_point_body) 
        domain_obs[4] = self.m1to1(closest_point_body[0], -relevant_distance,relevant_distance) 
        domain_obs[5] = self.m1to1(closest_point_body[1], -relevant_distance, relevant_distance) 
        domain_obs[6] = self.m1to1(closest_point_body[2], -relevant_distance,relevant_distance) 
        
        # print("closestppath world", np.round(closest_point),\
        #       "  closestppath body", np.round(closest_point_body),\
        #       "  quad pos", np.round(self.quadcopter.position),\
        #       "  quad att", np.round(self.quadcopter.attitude))

        # Two angles to describe direction of the vector between the drone and the closeset point on path
        x_b_cpp = closest_point_body[0]
        y_b_cpp = closest_point_body[1]
        z_b_cpp = closest_point_body[2]
        ele_closest_p_point_vec = np.arctan2(z_b_cpp, np.sqrt(x_b_cpp**2 + y_b_cpp**2))
        azi_closest_p_point_vec = np.arctan2(y_b_cpp, x_b_cpp)
        pure_obs.extend([ele_closest_p_point_vec, azi_closest_p_point_vec])
        # print(  "elevation angle to CPP", np.round(ele_closest_p_point_vec*180/np.pi,2),\
        #         "  azimuth angle to CPP", np.round(azi_closest_p_point_vec*180/np.pi,2),\
        #         )
        
        domain_obs[7] = np.sin(ele_closest_p_point_vec)
        domain_obs[8] = np.cos(ele_closest_p_point_vec)
        domain_obs[9] = np.sin(azi_closest_p_point_vec)
        domain_obs[10] = np.cos(azi_closest_p_point_vec)

        #euclidean norm of the distance from drone to next waypoint
        relevant_distance = (self.path.length / self.n_waypoints-1)*2 #Should be n-1 waypoints to get m segments
        distance_to_next_wp = 0
        try:
            distance_to_next_wp = np.linalg.norm(self.path.waypoints[self.waypoint_index+1] - self.quadcopter.position)
        except IndexError:
            distance_to_next_wp = np.linalg.norm(self.path.waypoints[-1] - self.quadcopter.position)
        pure_obs.append(distance_to_next_wp)
        domain_obs[11] = self.m1to1(distance_to_next_wp, -relevant_distance, relevant_distance)
        # print("dist_nxt_wp", np.round(distance_to_next_wp),"  normed", np.round(domain_obs[18],2))

        #euclidean norm of the distance from drone to the final waypoint
        distance_to_end = np.linalg.norm(self.path.get_endpoint() - self.quadcopter.position)
        pure_obs.append(distance_to_end)
        domain_obs[12] = self.m1to1(distance_to_end, -self.path.length*2, self.path.length*2)

        #body coordinates of the look ahead point
        lookahead_world = self.path.get_lookahead_point(self.quadcopter.position, self.la_dist, self.waypoint_index)
        #If lookahead point is the end point lock it to the end point
        if not self.LA_at_end and np.abs(lookahead_world[0] - self.path.get_endpoint()[0]) < 1 and np.abs(lookahead_world[1] - self.path.get_endpoint()[1]) < 1 and np.abs(lookahead_world[2] - self.path.get_endpoint()[2]) < 1:
            self.LA_at_end = True
        if self.LA_at_end:
            lookahead_world = self.path.get_endpoint()    

        lookahead_body = np.transpose(geom.Rzyx(*self.quadcopter.attitude)).dot(lookahead_world - self.quadcopter.position)
        pure_obs.extend(lookahead_body)
        relevant_distance = self.la_dist*2 #TODO decide this value
        domain_obs[13] = self.m1to1(lookahead_body[0], -relevant_distance,relevant_distance)
        domain_obs[14] = self.m1to1(lookahead_body[1], -relevant_distance, relevant_distance)
        domain_obs[15] = self.m1to1(lookahead_body[2], -relevant_distance,relevant_distance)

        #velocity vector in body frame PROBS NOT NEEDED/WORKS AGAINST IE CONFUSES THE AGENT
        # velocity_body = self.quadcopter.velocity
        # pure_obs.extend(velocity_body)
        # domain_obs[13] = self.m1to1(velocity_body[0], -self.s_max, self.s_max)
        # domain_obs[14] = self.m1to1(velocity_body[1], -self.s_max, self.s_max)
        # domain_obs[15] = self.m1to1(velocity_body[2], -self.s_max, self.s_max)
        
        self.info['pure_obs'] = pure_obs
        # print(np.round(domain_obs,2))

        # if self.total_t_steps % 1000 == 0:
        #     print("\nsensor_readings type:", type(sensor_readings), "  shape:", sensor_readings.shape, "  dtype:", sensor_readings.dtype)
        #     print("IMU type:", type(imu_measurement), "  shape:", imu_measurement.shape, "  dtype:", imu_measurement.dtype)
        #     print("domain_obs type:", type(domain_obs), "  shape:", domain_obs.shape, "  dtype:", domain_obs.dtype)
        

        return {'perception':sensor_readings,
                'IMU':imu_measurement,
                'domain':domain_obs}


    def step(self, action):
        """
        Simulates the environment one time-step.
        """
        self.update_errors()

        F = self.geom_ctrlv2(action)
        #TODO maybe need some translation between input u and thrust F i.e translate u to propeller speed? 
        #We currently skip this step for simplicity
        self.quadcopter.step(F)

        if self.path:
            self.prog = self.path.get_closest_u(self.quadcopter.position, self.waypoint_index)

            # Check if a waypoint is passed
            k = self.path.get_u_index(self.prog)
            if k > self.waypoint_index:
                print("Passed waypoint {:d}".format(k+1), self.path.waypoints[k], "\tquad position:", self.quadcopter.position)
                self.passed_waypoints = np.vstack((self.passed_waypoints, self.path.waypoints[k]))
                self.waypoint_index = k

        # Check collision
        self.update_nearby_obstacles()
        for obstacle in self.nearby_obstacles:
            quad_pos_torch = torch.tensor(self.quadcopter.position, dtype=torch.float32, device=self.device)
            if torch.norm(obstacle.position - quad_pos_torch) <= obstacle.radius + self.quadcopter.safety_radius:
                self.collided = True

        end_cond_1 = np.linalg.norm(self.path.get_endpoint() - self.quadcopter.position) < self.accept_rad and self.waypoint_index == self.n_waypoints-2
        end_cond_2 = abs(self.prog - self.path.length) <= self.accept_rad/2.0
        end_cond_3 = self.total_t_steps >= self.max_t_steps
        end_cond_4 = self.cumulative_reward < self.min_reward
        # end_cond_4 = False
        if end_cond_1 or end_cond_2 or end_cond_3 or self.collided or end_cond_4:
            if end_cond_1:
                print("Quadcopter reached target!")
                self.success = True
            elif self.collided:
                print("Quadcopter collided!")
                self.success = False
            elif end_cond_2: #I think this Should be removed such that the quadcopter can fly past the endpoint and come back #TODO
                print("Passed endpoint without hitting")
            elif end_cond_3:
                print("Exceeded time limit") #Should maybe set done = truncated = True here? #TODO
            elif end_cond_4:
                print("Acumulated reward less than", self.min_reward)
            self.done = True

        # Save sim time info
        self.total_t_steps += 1
        
        #Save interesting info
        self.info['env_steps'] = self.total_t_steps
        self.info['time'] = self.total_t_steps*self.step_size
        self.info['progression'] = self.prog/self.path.length
        self.info['state'] = np.copy(self.quadcopter.state)
        self.info['errors'] = np.array([self.e, self.h])
        self.info['cmd_thrust'] = self.quadcopter.input
        self.info['action'] = action

        # Calculate reward
        step_reward = self.reward()
        self.cumulative_reward += step_reward

        # Make next observation
        self.observation = self.observe()
        domain_obs = self.observation['domain']
        self.info['domain_obs'] = domain_obs

        #TODO See stack overflow QnA or Sb3 documentation for how to use truncated
        truncated = False
        return self.observation, step_reward, self.done, truncated, self.info


    def reward(self):
        """
        Calculates the reward function for one time step. 
        """
        tot_reward = 0
        lambda_PA = 1
        lambda_CA = 1

        #Path adherence reward
        dist_from_path = np.linalg.norm(self.path(self.prog) - self.quadcopter.position)
        # reward_path_adherence = np.clip(- np.log(dist_from_path), - np.inf, - np.log(0.1)) / (- np.log(0.1)) #OLD
        reward_path_adherence = -(2*(np.clip(dist_from_path, 0, self.PA_band_edge) / self.PA_band_edge) - 1)*self.PA_scale 
        # print("reward_path_adherence", np.round(reward_path_adherence,2),\
        #       "  dist_from_path", np.round(dist_from_path,2))
              

        #Path progression reward 
        reward_path_progression = 0
        reward_path_progression1 = np.cos(self.chi_error)*np.linalg.norm(self.quadcopter.velocity)*self.PP_vel_scale
        reward_path_progression2 = np.cos(self.upsilon_error)*np.linalg.norm(self.quadcopter.velocity)*self.PP_vel_scale
        reward_path_progression = reward_path_progression1/2 + reward_path_progression2/2
        reward_path_progression = np.clip(reward_path_progression, self.PP_rew_min, self.PP_rew_max)
        # print(  "chi error [deg]", np.round(self.chi_error*180/np.pi),\
        #         "  upsilon error [deg]", np.round(self.upsilon_error*180/np.pi),\
        #         "  yaw", np.round(self.quadcopter.attitude[2]*180/np.pi),\
        #         "  vel", np.round(np.linalg.norm(self.quadcopter.velocity),2),\
        #         "  rew_PA", np.round(reward_path_adherence,2),\
        #         "  rew_PP", np.round(reward_path_progression,2),\
        #         "  rew_PP_Chi", np.round(reward_path_progression1,2),\
        #         "  rew_PP_Ups", np.round(reward_path_progression2,2))


        ####Collision avoidance reward####
        #Find the closest obstacle
        reward_collision_avoidance = 0
        if self.nearby_obstacles != []: #If there are no obstacles, no need to calculate the reward

            inv_abs_min_rew = self.abs_inv_CA_min_rew 
            danger_range = self.danger_range
            danger_angle = self.danger_angle            
            quad_pos_torch = torch.tensor(self.quadcopter.position, dtype=torch.float32, device=self.device)
            drone_closest_obs_dist = torch.norm(self.nearby_obstacles[0].position - quad_pos_torch).item()
            #Determine lambda reward for path following and path adherence based on the distance to the closest obstacle
            if (drone_closest_obs_dist < danger_range):
                lambda_PA = (drone_closest_obs_dist/danger_range)/2
                if lambda_PA < 0.10 : lambda_PA = 0.10
                lambda_CA = 1-lambda_PA
            
            #Determine the angle difference between the velocity vector and the vector to the closest obstacle
            velocity_vec_torch = torch.tensor(self.quadcopter.velocity, dtype=torch.float32, device=self.device)
            drone_to_obstacle_vec = self.nearby_obstacles[0].position - quad_pos_torch
            #No worries to use this in simulation to do reward calculations as long as the observations allow for correlation between this reward and the actual world
            angle_diff = torch.arccos(torch.dot(drone_to_obstacle_vec, velocity_vec_torch)/(torch.norm(drone_to_obstacle_vec)*torch.norm(velocity_vec_torch))).item()

            reward_collision_avoidance = 0
            if (drone_closest_obs_dist < danger_range) and (angle_diff < danger_angle):
                range_rew = -(((danger_range+inv_abs_min_rew*danger_range)/(drone_closest_obs_dist+inv_abs_min_rew*danger_range)) -1) #same fcn in if and elif, but need this structure to color red and orange correctly
                angle_rew = -(((danger_angle+inv_abs_min_rew*danger_angle)/(angle_diff+inv_abs_min_rew*danger_angle)) -1)
                if angle_rew > 0: angle_rew = 0 
                if range_rew > 0: range_rew = 0
                reward_collision_avoidance = range_rew + angle_rew

                self.draw_red_velocity = True
                self.draw_orange_obst_vec = True
            elif drone_closest_obs_dist <danger_range:
                range_rew = -(((danger_range+inv_abs_min_rew*danger_range)/(drone_closest_obs_dist+inv_abs_min_rew*danger_range)) - 1)
                angle_rew = -(((danger_angle+inv_abs_min_rew*danger_angle)/(angle_diff+inv_abs_min_rew*danger_angle)) - 1)
                if angle_rew > 0: angle_rew = 0 #In this case the angle reward may become positive as anglediff may < danger_angle
                if range_rew > 0: range_rew = 0
                reward_collision_avoidance = range_rew + angle_rew
                
                self.draw_red_velocity = False
                self.draw_orange_obst_vec = True
            else:
                reward_collision_avoidance = 0
                self.draw_red_velocity = False
                self.draw_orange_obst_vec = False
            ####Collision avoidance reward done####

        #Collision reward
        reward_collision = 0
        if self.collided:
            reward_collision = self.rew_collision
            print("Collision Reward:", reward_collision)

        #Reach end reward
        reach_end_reward = 0
        if self.success:
            reach_end_reward = self.rew_reach_end

        #Existential reward (penalty for being alive to encourage the quadcopter to reach the end of the path quickly)
        ex_reward = self.existence_reward 

        tot_reward = reward_path_adherence*lambda_PA + reward_collision_avoidance*lambda_CA + reward_collision + reward_path_progression + reach_end_reward + ex_reward

        self.info['reward'] = tot_reward
        self.info['collision_avoidance_reward'] = reward_collision_avoidance*lambda_CA
        self.info['path_adherence'] = reward_path_adherence*lambda_PA
        self.info["path_progression"] = reward_path_progression
        self.info['collision_reward'] = reward_collision
        self.info['reach_end_reward'] = reach_end_reward
        self.info['existence_reward'] = ex_reward
        
        return tot_reward


    def geom_ctrlv2(self, action):
        #Translate the action to the desired velocity and yaw rate
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1]*self.i_max)
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1]*self.i_max)
        cmd_r = self.r_max * action[2]
        self.cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r]) #For plotting

        #Gains, z-axis-basis=e3 and rotation matrix #TODO add stochasticity to make sim2real robust
        kv = 2.5
        kR = 0.8
        kangvel = 0.8

        e3 = np.array([0, 0, 1]) #z-axis basis
        R = geom.Rzyx(*self.quadcopter.attitude)

        #Essentially three different velocities that one can choose to track:
        #Think body or world frame velocity control is the best choice
        ###---###
        ##Vehicle frame velocity control i.e. intermediary frame between body and world frame
        # vehicleR = geom.Rzyx(0, 0, self.quadcopter.attitude[2])
        # vehicle_vels = vehicleR.T @ self.quadcopter.velocity
        # ev = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - vehicle_vels
        ###---###
        #World frame velocity control
        # ev = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - self.quadcopter.position_dot

        #Body frame velocity control
        ev = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - self.quadcopter.velocity
        #Which one to use? vehicle_vels or self.quadcopter.velocity

        #Thrust command (along body z axis)
        f = kv*ev + ss.m*ss.g*e3 + ss.d_w*self.quadcopter.heave*e3
        thrust_command = np.dot(f, R[2])

        #Rd calculation as in Kulkarni aerial gym (works fairly well)
        c_phi_s_theta = f[0]
        s_phi = -f[1]
        c_phi_c_theta = f[2]

        pitch_setpoint = np.arctan2(c_phi_s_theta, c_phi_c_theta)
        roll_setpoint = np.arctan2(s_phi, np.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
        yaw_setpoint = self.quadcopter.attitude[2]
        Rd = geom.Rzyx(roll_setpoint, pitch_setpoint, yaw_setpoint)
        self.att_des = np.array([roll_setpoint, pitch_setpoint, yaw_setpoint]) # Save the desired attitude for plotting

        eR = 1/2*(Rd.T @ R - R.T @ Rd)
        eatt = geom.vee_map(eR)
        eatt = np.reshape(eatt, (3,))

        des_angvel = np.array([0.0, 0.0, cmd_r])

        #Kulkarni approach desired angular rate in body frame:
        s_pitch = np.sin(self.quadcopter.attitude[1])
        c_pitch = np.cos(self.quadcopter.attitude[1])
        s_roll = np.sin(self.quadcopter.attitude[0])
        c_roll = np.cos(self.quadcopter.attitude[0])
        R_euler_to_body = np.array([[1, 0, -s_pitch],
                                    [0, c_roll, s_roll*c_pitch],
                                    [0, -s_roll, c_roll*c_pitch]]) #Uncertain about how this came to be

        des_angvel_body = R_euler_to_body @ des_angvel

        eangvel = self.quadcopter.angular_velocity - R.T @ (Rd @ des_angvel_body) #Kulkarni approach

        torque = -kR*eatt - kangvel*eangvel + np.cross(self.quadcopter.angular_velocity,ss.Ig@self.quadcopter.angular_velocity)

        u = np.zeros(4)
        u[0] = thrust_command
        u[1:] = torque

        F = np.linalg.inv(ss.B()[2:]).dot(u)
        F = np.clip(F, ss.thrust_min, ss.thrust_max)
        return F


    #### UTILS ####
    def m1to1(self,value, min, max): 
        '''
        Normalizes a value from the range [min,max] to the range [-1,1]
        If value is outside the range, it will be clipped to the min or max value
        '''
        value_normalized = 2.0*(value-min)/(max-min) - 1
        return np.clip(value_normalized, -1, 1)

    def invm1to1(self, value, min, max):
        '''
        Inverse normalizes a value from the range [-1,1] to the range [min,max]
        If value that got normalized was outside the min max range it may only be inverted to the min or max value
        '''
        return (value+1)*(max-min)/2.0 + min


    #### UPDATE FUNCTIONS ####
    def update_errors(self): #TODO these dont need to be self.variables and should rather be returned
        '''Updates the cross track and vertical track errors, as well as the course and elevation errors.'''
        self.e = 0.0 #Cross track error
        self.h = 0.0 #Vertical track error
        self.chi_error = 0.0 #Course angle error xy-plane
        self.upsilon_error = 0.0 #Elevation angle error between z and xy-plane

        s = self.prog

        chi_p, upsilon_p = self.path.get_direction_angles(s) #Path direction angles also denoted by pi in some literature
        # Calculate tracking errors Serret Frenet frame
        SF_rotation = geom.Rzyx(0, upsilon_p, chi_p)

        epsilon = np.transpose(SF_rotation).dot(self.quadcopter.position - self.path(s))
        self.e = epsilon[1] #Cross track error
        self.h = epsilon[2] #Vertical track error

        # Calculate course and elevation errors from tracking errors
        chi_r = np.arctan2(self.e, self.la_dist) 
        upsilon_r = np.arctan2(self.h, np.sqrt(self.e**2 + self.la_dist**2))
        
        #Desired course and elevation angles
        chi_d = chi_p - chi_r 
        upsilon_d = upsilon_p - upsilon_r 

        self.chi_error = geom.ssa(chi_d - self.quadcopter.chi) #Course angle error xy-plane 
        self.upsilon_error = geom.ssa(upsilon_d - self.quadcopter.upsilon) #Elevation angle error between z and xy-plane

        # print("upsilon_d", np.round(upsilon_d*180/np.pi), "upsilon_quad", np.round(self.quadcopter.upsilon*180/np.pi), "upsilon_error", np.round(self.upsilon_error*180/np.pi),\
        #       "\n\nchi_d", np.round(chi_d*180/np.pi), "chi_quad", np.round(self.quadcopter.chi*180/np.pi), "chi_error", np.round(self.chi_error*180/np.pi))

    # Numpy version     
    # def update_nearby_obstacles(self): #Keep as long as obstacles are spheres TODO may remove/redo when obstacles are more complex
    #     """
    #     Updates the nearby_obstacles array.
    #     """
    #     self.nearby_obstacles = []
    #     for obstacle in self.obstacles:
    #         distance_vec_world = obstacle.position - self.quadcopter.position
    #         distance = np.linalg.norm(distance_vec_world)
    #         distance_vec_BODY = np.transpose(geom.Rzyx(*self.quadcopter.attitude)).dot(distance_vec_world)
    #         heading_angle_BODY = np.arctan2(distance_vec_BODY[1], distance_vec_BODY[0])
    #         pitch_angle_BODY = np.arctan2(distance_vec_BODY[2], np.sqrt(distance_vec_BODY[0]**2 + distance_vec_BODY[1]**2))

    #         # check if the obstacle is inside the sonar window
    #         if distance - self.quadcopter.safety_radius - obstacle.radius <= self.max_depth and abs(heading_angle_BODY) <= self.FOV_horizontal*np.pi/180 \
    #         and abs(pitch_angle_BODY) <= self.FOV_vertical*np.pi/180:
    #             self.nearby_obstacles.append(obstacle)
    #         elif distance <= obstacle.radius + self.quadcopter.safety_radius:
    #             self.nearby_obstacles.append(obstacle)
    #     # Sort the obstacles such that the closest one is first
    #     self.nearby_obstacles.sort(key=lambda x: np.linalg.norm(x.position - self.quadcopter.position)) 

    #Tensor version
    def update_nearby_obstacles(self): 
        """
        Updates the nearby_obstacles array. 
        The closest obstacle is first in the list.
        """
        self.nearby_obstacles = []
        quad_pos_torch = torch.tensor(self.quadcopter.position, device=self.device).float()
        for obstacle in self.obstacles:
            distance_vec_world = obstacle.position - quad_pos_torch
            distance = torch.norm(distance_vec_world)
            distance_vec_BODY = torch.matmul(torch.tensor(geom.Rzyx(*self.quadcopter.attitude), device=self.device).float(), distance_vec_world)
            heading_angle_BODY = torch.atan2(distance_vec_BODY[1], distance_vec_BODY[0])
            pitch_angle_BODY = torch.atan2(distance_vec_BODY[2], torch.sqrt(distance_vec_BODY[0]**2 + distance_vec_BODY[1]**2))

            # check if the obstacle is inside the sonar window
            if distance - self.quadcopter.safety_radius - obstacle.radius <= self.max_depth \
            and abs(heading_angle_BODY) <= self.FOV_horizontal*np.pi/180 \
            and abs(pitch_angle_BODY) <= self.FOV_vertical*np.pi/180:
                self.nearby_obstacles.append(obstacle)
            # elif distance <= obstacle.radius + self.quadcopter.safety_radius: #This essentially gives 360 degree FOV which we do not have.
            #     self.nearby_obstacles.append(obstacle)
        # Sort the obstacles such that the closest one is first
        self.nearby_obstacles.sort(key=lambda x: torch.norm(x.position - quad_pos_torch))

    #### PLOTTING ####
    def axis_equal3d(self, ax):
        """
        Shifts axis in 3D plots to be equal. Especially useful when plotting obstacles, so they appear spherical.

        Parameters:
        ----------
        ax : matplotlib.axes
            The axes to be shifted.
        """
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        # plt.show()
        return ax

    def plot3D(self, wps_on=True):
        """
        Returns 3D plot of path and obstacles.
        """
        ax = self.path.plot_path(wps_on)
        for obstacle in self.obstacles:
            ax.plot_surface(*obstacle.return_plot_variables(), color='r', zorder=1)

        return self.axis_equal3d(ax)

    def plot_section3d(self):
        """
        Returns 3D plot of path, obstacles and quadcopter.
        """
        plt.rc('lines', linewidth=3)
        ax = self.plot3D(wps_on=False)
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([-50, 0, 50])
        ax.set_zticks([-50, 0, 50])
        ax.view_init(elev=-165, azim=-35)
        ax.scatter3D(*self.quadcopter.position, label="Initial Position", color="y")

        self.axis_equal3d(ax)
        ax.legend(fontsize=14)
        plt.show()


    #### SCENARIOS #### 
        #Utility functions for scenarios
    #Numpy version: want to rewrite to torch to avoid moving data between cpu and gpu    
    # def check_object_overlap(self, new_obstacle): #TODO larger task of rewriting np to torch
    #     """
    #     Checks if a new obstacle is overlapping one that already exists or the target position.
    #     """
    #     overlaps = False
    #     # check if it overlaps target:
    #     if np.linalg.norm(self.path.get_endpoint() - new_obstacle.position.numpy()) < new_obstacle.radius + 5:
    #         return True
    #     # check if it overlaps already placed objects
    #     for obstacle in self.obstacles:
    #         if np.linalg.norm(obstacle.position.numpy() - new_obstacle.position.numpy()) < new_obstacle.radius + obstacle.radius + 5:
    #             overlaps = True
    #     return overlaps
    
    def check_object_overlap(self, new_obstacle): 
        """
        Checks if a new obstacle is overlapping one that already exists or the target position.
        """
        overlaps = False
        # check if it overlaps target:
        endpoint_torch = torch.tensor(self.path.get_endpoint(),device=self.device).float()
        if torch.norm(endpoint_torch - new_obstacle.position) < new_obstacle.radius + 5:
            return True
        # check if it overlaps already placed objects
        for obstacle in self.obstacles:
            if torch.norm(obstacle.position - new_obstacle.position) < new_obstacle.radius + obstacle.radius + 5:
                overlaps = True
        return overlaps

    #No obstacles
    def scenario_line(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'line')
        self.path = QPMI(waypoints)
        # init_pos = [np.random.uniform(0,2)*(-5),0, 0]#np.random.normal(0,1)*5]
        init_pos = [0, 0, 0]#np.random.normal(0,1)*5]
        #init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_line_new(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'line_new')
        self.path = QPMI(waypoints)
        # init_pos = [np.random.uniform(0,2)*(-5),0, 0]#np.random.normal(0,1)*5]
        init_pos = [0, 0, 0]#np.random.normal(0,1)*5]
        #init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_horizontal(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'horizontal')
        self.path = QPMI(waypoints)
        # init_pos = [np.random.uniform(0,2)*(-5), np.random.normal(0,1)*5, 0]#np.random.normal(0,1)*5]
        init_pos = [0, 0, 0]#np.random.normal(0,1)*5]
        #init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_horizontal_new(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'horizontal_new')
        self.path = QPMI(waypoints)
        # init_pos = [np.random.uniform(0,2)*(-5), np.random.normal(0,1)*5, 0]#np.random.normal(0,1)*5]
        init_pos = [0, 0, 0]#np.random.normal(0,1)*5]
        #init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_3d(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'3d')
        self.path = QPMI(waypoints)
        init_pos = [np.random.uniform(0,2)*(-5), np.random.normal(0,1)*5, np.random.normal(0,1)*5]
        #init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        init_attitude=np.array([0, 0, self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_3d_new(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'3d_new')
        self.path = QPMI(waypoints)
        # init_pos=[-10, -10, 0]
        init_pos = [np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-10,10)]
        init_pos=[0, 0, 0]
        init_attitude=np.array([0, 0, self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    #With obstacles
    def scenario_intermediate(self):
        initial_state = self.scenario_3d_new()
        obstacle_radius = np.random.uniform(low=4,high=10)
        obstacle_coords = self.path(self.path.length/2)# + np.random.uniform(low=-obstacle_radius, high=obstacle_radius, size=(1,3))
        
        obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
        self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        return initial_state

    def scenario_proficient(self):
        initial_state = self.scenario_3d_new()
        obstacle_radius = np.random.uniform(low=4,high=10)
        obstacle_coords = self.path(self.path.length/2)# + np.random.uniform(low=-obstacle_radius, high=obstacle_radius, size=(1,3))
        obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze() #go from [[x,y,z]] to [x,y,z]
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
        self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))

        lengths = np.linspace(self.path.length*1/6, self.path.length*5/6, 2)
        for l in lengths:
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(l) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
            obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze() #TODO apply squeeze to all other obstacle_coords that are [[]] and not []
            pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
            obstacle = SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path)
            
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)
        return initial_state


    # def scenario_advanced(self): #IDK WHY COMMENTED OUT
    #     initial_state = self.scenario_proficient()
    #     while len(self.obstacles) < self.n_adv_obstacles: # Place the rest of the obstacles randomly
    #         s = np.random.uniform(self.path.length*1/3, self.path.length*2/3)
    #         obstacle_radius = np.random.uniform(low=4,high=10)
    #         obstacle_coords = self.path(s) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
    #         obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
    #         if self.check_object_overlap(obstacle):
    #             continue
    #         else:
    #             self.obstacles.append(obstacle)
    #     return initial_state

    def scenario_expert(self):
        initial_state = self.scenario_3d_new()
        obstacle_radius = np.random.uniform(low=4,high=10)
        obstacle_coords = self.path(self.path.length/2)# + np.random.uniform(low=-obstacle_radius, high=obstacle_radius, size=(1,3))
        obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
        self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))

        lengths = np.linspace(self.path.length*1.5/6, self.path.length*5/6, 5)
        for l in lengths:
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(l) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
            obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze()
            pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)

            obstacle = SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords[2],device=self.device,path=self.mesh_path)
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)

        return initial_state

    def scenario_test_path(self):
        # test_waypoints = np.array([np.array([0,0,0]), np.array([1,1,0]), np.array([9,9,0]), np.array([10,10,0])])
        # test_waypoints = np.array([np.array([0,0,0]), np.array([5,0,0]), np.array([10,0,0]), np.array([15,0,0])])
        test_waypoints = np.array([np.array([0,0,0]), np.array([10,1,0]), np.array([20,0,0]), np.array([70,0,0])])
        self.n_waypoints = len(test_waypoints)
        self.path = QPMI(test_waypoints)
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])

        obstacle_radius = 10
        obstacle_coords = self.path(20)
        obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
        self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        return initial_state

    def scenario_test(self):
        initial_state = self.scenario_test_path()
        obstacle_radius = 10
        obstacle_coords = self.path(self.path.length/2)
        obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
        self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        return initial_state

    def scenario_horizontal_test(self):
        waypoints = [(0,0,0), (50,0.1,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.obstacles = []
        for i in range(7):
            y = -30+10*i
            obstacle_coords = torch.tensor([50,y,0],device=self.device)
            pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
            self.obstacles.append(SphereMeshObstacle(radius = 5, center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
            
        init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-5, high=5, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        return initial_state

    def scenario_vertical_test(self):
        waypoints = [(0,0,0), (50,0,1), (100,0,0)]
        self.path = QPMI(waypoints)
        self.obstacles = []
        for i in range(7):
            z = -30+10*i
            obstacle_coords = torch.tensor([50,0,z],device=self.device)
            pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
            self.obstacles.append(SphereMeshObstacle(radius = 5,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-5, high=5, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        return initial_state

    def scenario_deadend_test(self):
        waypoints = [(0,0,0), (50,0.5,0), (100,0,0)]
        self.path = QPMI(waypoints)
        radius = 10
        angles = np.linspace(-90, 90, 10)*np.pi/180
        obstacle_radius = (angles[1]-angles[0])*radius/2
        for ang1 in angles:
            for ang2 in angles:
                x = 45 + radius*np.cos(ang1)*np.cos(ang2)
                y = radius*np.cos(ang1)*np.sin(ang2)
                z = -radius*np.sin(ang1)
                
                obstacle_coords = torch.tensor([x,y,z],device=self.device).float().squeeze()
                pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
                self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))

        init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-5, high=5, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        return initial_state

    def scenario_helix(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'helix')
        self.path = QPMI(waypoints)
        # init_pos = helix_param(0)
        init_pos = np.array([110, 0, -26]) + np.random.uniform(low=-5, high=5, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        obstacle_coords = torch.tensor([0,0,0],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords)
        self.obstacles.append(SphereMeshObstacle(radius = 100,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))

        return initial_state
