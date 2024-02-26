import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import gym_quad.utils.geomutils as geom
import gym_quad.utils.state_space as ss
from gym_quad.objects.quad import Quad
from gym_quad.objects.IMU import IMU
from gym_quad.objects.current3d import Current
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints
from gym_quad.objects.obstacle3d import Obstacle


#TODO decide the abstraction level between PPO agent and quadcopter controller and implement new controller
#TODO Implement the reward fcn from the specialization project step by step
#TODO Set up curriculum learning


class LV_VAE(gym.Env):
    '''Creates an environment where the actionspace consists of Linear velocity and yaw rate which will be passed to a PD or PID controller,
    while the observationspace uses a Varial AutoEncoder "plus more" for observations of environment.'''

    def __init__(self, env_config, scenario="line", seed=None):
        np.random.seed(0)

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
        #LIDAR or Depth camera observation space
        self.perception_space = gym.spaces.Box(
            low = 0,
            high = 1,
            shape = (1, self.sensor_suite[0], self.sensor_suite[1]),
            dtype = np.float64
        )

        #IMU observation space
        self.IMU_space = gym.spaces.Box(
            low = -1,
            high = 1,
            shape = (6,),
            dtype = np.float64
        )

        self.observation_space = gym.spaces.Dict({
        'perception': self.perception_space,
        'IMU': self.IMU_space
        })

        #Init values for sensor
        self.n_sensor_readings = self.sensor_suite[0]*self.sensor_suite[1]
        max_horizontal_angle = self.sensor_span[0]/2
        max_vertical_angle = self.sensor_span[1]/2
        self.sectors_horizontal = np.linspace(-max_horizontal_angle*np.pi/180, max_horizontal_angle*np.pi/180, self.sensor_suite[0])
        self.sectors_vertical =  np.linspace(-max_vertical_angle*np.pi/180, max_vertical_angle*np.pi/180, self.sensor_suite[1])

        #Scenario set up
        self.scenario = scenario
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
            "test_current": self.scenario_test_current,
            "horizontal": self.scenario_horizontal_test,
            "vertical": self.scenario_vertical_test,
            "deadend": self.scenario_deadend_test
        }
        #Reset environment to init state
        self.reset()


    def reset(self,**kwargs):
        """
        Resets environment to initial state.
        """
        seed = kwargs.get('seed', None)
        # print("PRINTING SEED WHEN RESETTING:", seed)
        self.quadcopter = None
        self.path = None
        self.path_generated = None
        self.e = None
        self.h = None
        self.chi_error = None
        self.upsilon_error = None
        self.waypoint_index = 0
        self.prog = 0
        # self.path_prog = []
        self.success = False
        self.done = False

        self.obstacles = []
        self.nearby_obstacles = []
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        self.collided = False

        # self.a_des = np.array([0.0, 0.0, 0.0])
        self.a_des = np.array([np.inf, np.inf, np.inf])
        self.prev_position_error = [0, 0, 0]
        self.total_position_error = [0, 0, 0]

        self.passed_waypoints = np.zeros((1, 3), dtype=np.float32)

        self.observation = {
            'perception': np.zeros((1, self.sensor_suite[0], self.sensor_suite[1])),
            'IMU': np.zeros((6,))
        }
        # self.observation = self.observe() #OLD I think the stuff above should be good enough #TODO
        #Or maybe its actually smart to call the observe function here to get the first observation? 

        self.total_t_steps = 0
        self.ex_reward = 0

        ### Path and obstacle generation based on scenario
        scenario = self.scenario_switch.get(self.scenario, lambda: print("Invalid scenario"))
        init_state = scenario()
        # Generate Quadcopter
        self.quadcopter = Quad(self.step_size, init_state)
        ###
        self.update_errors()
        self.info = {}
        return (self.observation,self.info)


    def observe(self):
        """
        Returns observations of the environment.
        """

        imu = IMU()
        imu_measurement = imu.measure(self.quadcopter)

        # Update nearby obstacles and calculate distances PER NOW THESE FCN CALLED ONCE SO DONT NEED TO BE FCNS
        self.update_nearby_obstacles()
        self.update_sensor_readings()

        sensor_readings = self.sensor_readings.reshape(1, self.sensor_suite[0], self.sensor_suite[1])

        return {'perception':sensor_readings,
                'IMU':imu_measurement}


    def step(self, action):
        """
        Simulates the environment one time-step.
        """
        self.update_errors()

        F = self.geom_ctrlv2(action)
        self.quadcopter.step(F)


        if self.path:
            self.prog = self.path.get_closest_u(self.quadcopter.position, self.waypoint_index)
            # self.path_prog.append(self.prog)

            # Check if a waypoint is passed
            k = self.path.get_u_index(self.prog)
            if k > self.waypoint_index:
                print("Passed waypoint {:d}".format(k+1), self.path.waypoints[k], "\tquad position:", self.quadcopter.position)
                self.passed_waypoints = np.vstack((self.passed_waypoints, self.path.waypoints[k]))
                self.waypoint_index = k

        # Check collision
        for obstacle in self.nearby_obstacles:
            if np.linalg.norm(obstacle.position - self.quadcopter.position) <= obstacle.radius + self.quadcopter.safety_radius:
                self.collided = True

        end_cond_1 = np.linalg.norm(self.path.get_endpoint() - self.quadcopter.position) < self.accept_rad and self.waypoint_index == self.n_waypoints-2
        end_cond_2 = abs(self.prog - self.path.length) <= self.accept_rad/2.0
        end_cond_3 = self.total_t_steps >= self.max_t_steps
        # end_cond_4 = self.reward < self.min_reward
        # end_cond_4 = False
        if end_cond_1 or end_cond_2 or end_cond_3 or self.collided: # or end_cond_4:
            if end_cond_1:
                print("Quadcopter reached target!")
                self.success = True
            elif self.collided:
                print("Quadcopter collided!")
                self.success = False
            elif end_cond_2: #I think this Should be removed such that the quadcopter can fly past the endpoint and come back #TODO
                print("Passed endpoint without hitting")
            elif end_cond_3:
                print("Exceeded time limit")
            self.done = True

        # Save sim time info
        self.total_t_steps += 1
        
        #Save interesting info
        self.info['env_steps'] = self.total_t_steps
        #These below were lists before as visible in the get_stats fcn commented out below MIGHT HAVE TO DO SOME TRICKS TO MAKE THIS WORK WITH THE OLD LOGGER
        self.info['time'] = self.total_t_steps*self.step_size
        self.info['progression'] = self.prog/self.path.length
        self.info['state'] = np.copy(self.quadcopter.state)
        self.info['errors'] = np.array([self.chi_error, self.e, self.upsilon_error, self.h])
        self.info['action'] = self.quadcopter.input

        # Calculate reward
        step_reward = self.reward()
        
        # Make next observation
        self.observation = self.observe()

        #dummy truncated for debugging See stack overflow QnA or Sb3 documentation for how to use truncated
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

        #Path progression reward #TODO double check if upsilon should be used here and if yes if cos is correct
        #TODO lock the lookahead point to the end goal when it first reaches the end goal
        reward_path_progression = np.cos(self.chi_error*np.pi)*np.cos(self.upsilon_error*np.pi)*np.linalg.norm(self.quadcopter.velocity)*self.PP_vel_scale
        reward_path_progression = np.clip(reward_path_progression, self.PP_rew_min, self.PP_rew_max)

        ####Collision avoidance reward####
        #Find the closest obstacle
        drone_closest_obs_dist = np.inf
        self.sensor_readings
        for i in range(self.sensor_suite[0]):
            for j in range(self.sensor_suite[1]):
                if self.sensor_readings[j,i] < drone_closest_obs_dist:
                    drone_closest_obs_dist = self.sensor_readings[j,i]
                    print("drone_closest_obs_dist", drone_closest_obs_dist)

        inv_abs_min_rew = self.abs_inv_CA_min_rew 
        danger_range = self.danger_range
        danger_angle = self.danger_angle            
        
        #Determine lambda reward for path following and path adherence based on the distance to the closest obstacle
        if (drone_closest_obs_dist < danger_range):
            lambda_PA = (drone_closest_obs_dist/danger_range)/2
            if lambda_PA < 0.10 : lambda_PA = 0.10
            lambda_CA = 1-lambda_PA
        
        #Determine the angle difference between the velocity vector and the vector to the closest obstacle
        velocity_vec = self.quadcopter.velocity
        drone_to_obstacle_vec = self.nearby_obstacles[0].position - self.quadcopter.position #This wil require state estimation and preferably a GPS too hmm
        angle_diff = np.arccos(np.dot(drone_to_obstacle_vec, velocity_vec)/(np.linalg.norm(drone_to_obstacle_vec)*np.linalg.norm(velocity_vec)))

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
            range_rew = -(((danger_range+inv_abs_min_rew*danger_range)/(drone_closest_obs_dist+inv_abs_min_rew*danger_range)) -1)
            angle_rew = -(((danger_angle+inv_abs_min_rew*danger_angle)/(angle_diff+inv_abs_min_rew*danger_angle)) -1)
            if angle_rew > 0: angle_rew = 0 #In this case the angle reward may become positive as anglediff may !< danger_angle
            if range_rew > 0: range_rew = 0
            reward_collision_avoidance = range_rew + angle_rew
            
            self.draw_red_velocity = False
            self.draw_orange_obst_vec = True
        else:
            reward_collision_avoidance = 0
            self.draw_red_velocity = False
            self.draw_orange_obst_vec = False
        # print('reward_collision_avoidance', reward_collision_avoidance)


        #OLD
        # collision_avoidance_rew = self.penalize_obstacle_closeness()
        # reward_collision_avoidance = - 2 * np.log(1 - collision_avoidance_rew)
        ####Collision avoidance reward done####

        #Collision reward
        if self.collided:
            reward_collision = self.rew_collision
            # print("Reward:", self.reward_collision)

        #Reach end reward
        reach_end_reward = 0
        if self.success:
            reach_end_reward = self.rew_reach_end

        #Existencial reward (penalty for being alive)
        self.ex_reward += self.existence_reward

        tot_reward = reward_path_adherence*lambda_PA + reward_collision_avoidance*lambda_CA + reward_collision + reward_path_progression + reach_end_reward + self.ex_reward

        # self.reward_path_following_sum += self.lambda_reward * reward_path_adherence #OLD logging of info
        # self.reward_collision_avoidance_sum += (1 - self.lambda_reward) * reward_collision_avoidance
        # self.reward += tot_reward

        self.info['reward'] = tot_reward
        self.info['collision_avoidance_reward'] = reward_collision_avoidance*lambda_CA
        self.info['path_adherence'] = reward_path_adherence*lambda_PA
        self.info["path_progression"] = reward_path_progression
        self.info['collision_reward'] = reward_collision
        self.info['reach_end_reward'] = reach_end_reward
        self.info['existence_reward'] = self.ex_reward

        return tot_reward


    def geom_ctrlv2(self, action):
        #Translate the action to the desired velocity and yaw rate
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1])*self.i_max
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1])*self.i_max
        cmd_r = self.r_max * action[2]
        self.cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r]) #For plotting

        #Gains, z-axis-basis=e3 and rotation matrix
        kv = 2.5
        kR = 0.8
        kangvel = 0.8

        e3 = np.array([0, 0, 1])
        R = geom.Rzyx(*self.quadcopter.attitude)

        #Essentially three different velocities that one can choose to track:
        #Think body or wolrd frame velocity control is the best choice
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
    def get_stats(self):
        return self.info
        # return {"reward_path_following":self.reward_path_following_sum, #OLD
        #         "reward_collision_avoidance":self.reward_collision_avoidance_sum,
        #         "reward_collision":self.reward_collision}
        #         #,"obs":self.past_obs,"states":self.past_states,"errors":self.past_errors}

    def calculate_object_distance(self, alpha, beta, obstacle):
        """
        Searches along a sonar ray for an object
        """
        s = 0
        while s < self.sonar_range:
            x = self.quadcopter.position[0] + s*np.cos(alpha)*np.cos(beta)
            y = self.quadcopter.position[1] + s*np.sin(alpha)*np.cos(beta)
            z = self.quadcopter.position[2] + s*np.sin(beta)
            if np.linalg.norm(obstacle.position - [x,y,z]) <= obstacle.radius:
                break
            else:
                s += 1
        closeness = np.clip(1-(s/self.sonar_range), 0, 1)
        return s, closeness


    def penalize_obstacle_closeness(self): #TODO Probs doesnt need to be a fcn as called once
        """
        Calculates the colav reward
        """
        reward_colav = 0
        sensor_suite_correction = 0
        gamma_c = self.sonar_range/2
        epsilon = 0.05
        epsilon_closeness = 0.05

        horizontal_angles = np.linspace(- self.sensor_span[0]/2, self.sensor_span[0]/2, self.sensor_suite[0])
        vertical_angles = np.linspace(- self.sensor_span[1]/2, self.sensor_span[1]/2, self.sensor_suite[1])
        for i, horizontal_angle in enumerate(horizontal_angles):
            horizontal_factor = 1 - abs(horizontal_angle) / horizontal_angles[-1]
            for j, vertical_angle in enumerate(vertical_angles):
                vertical_factor = 1 - abs(vertical_angle) / vertical_angles[-1]
                beta = vertical_factor * horizontal_factor + epsilon
                sensor_suite_correction += beta
                reward_colav += (beta * (1 / (gamma_c * max(1 - self.sensor_readings[j,i], epsilon_closeness)**2)))**2

        return - 20 * reward_colav / sensor_suite_correction


    #### UPDATE FUNCTIONS ####
    def update_errors(self): #TODO these dont need to be self.variables and should rather be returned
        self.e = 0.0
        self.h = 0.0
        self.chi_error = 0.0
        self.upsilon_error = 0.0

        # Get path course and elevation
        # s = self.prog + self.la_dist
        # if s > self.path.us[-1]:
        #     s = self.path.us[-1]
        s = self.prog
        chi_p, upsilon_p = self.path.get_direction_angles(s)

        # Calculate tracking errors
        SF_rotation = geom.Rzyx(0, upsilon_p, chi_p)

        epsilon = np.transpose(SF_rotation).dot(self.quadcopter.position - self.path(s))
        self.e = epsilon[1] #Cross track error
        self.h = epsilon[2] #Vertical track error 

        # Calculate course and elevation errors from tracking errors
        chi_r = np.arctan2(self.e, self.la_dist)
        upsilon_r = np.arctan2(self.h, np.sqrt(self.e**2 + self.la_dist**2))
        chi_d = chi_p + chi_r
        upsilon_d =upsilon_p + upsilon_r
        self.chi_error = np.clip(geom.ssa(self.quadcopter.chi - chi_d)/np.pi, -1, 1) #Course angle error
        self.upsilon_error = np.clip(geom.ssa(self.quadcopter.upsilon - upsilon_d)/np.pi, -1, 1) #Elevation angle error

    def update_nearby_obstacles(self):
        """
        Updates the nearby_obstacles array.
        """
        self.nearby_obstacles = []
        for obstacle in self.obstacles:
            distance_vec_world = obstacle.position - self.quadcopter.position
            distance = np.linalg.norm(distance_vec_world)
            distance_vec_BODY = np.transpose(geom.Rzyx(*self.quadcopter.attitude)).dot(distance_vec_world)
            heading_angle_BODY = np.arctan2(distance_vec_BODY[1], distance_vec_BODY[0])
            pitch_angle_BODY = np.arctan2(distance_vec_BODY[2], np.sqrt(distance_vec_BODY[0]**2 + distance_vec_BODY[1]**2))

            # check if the obstacle is inside the sonar window
            if distance - self.quadcopter.safety_radius - obstacle.radius <= self.sonar_range and abs(heading_angle_BODY) <= self.sensor_span[0]*np.pi/180 \
            and abs(pitch_angle_BODY) <= self.sensor_span[1]*np.pi/180:
                self.nearby_obstacles.append(obstacle)
            elif distance <= obstacle.radius + self.quadcopter.safety_radius:
                self.nearby_obstacles.append(obstacle)
        # Sort the obstacles by lowest to largest distance
        self.nearby_obstacles.sort(key=lambda x: np.linalg.norm(x.position - self.quadcopter.position)) #TODO check if this works

    def update_sensor_readings(self):
        """
        Updates the sonar data closeness array.
        """
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        for obstacle in self.nearby_obstacles:
            for i in range(self.sensor_suite[0]):
                alpha = self.quadcopter.heading + self.sectors_horizontal[i]
                for j in range(self.sensor_suite[1]):
                    beta = self.quadcopter.pitch + self.sectors_vertical[j]
                    _, closeness = self.calculate_object_distance(alpha, beta, obstacle)
                    self.sensor_readings[j,i] = max(closeness, self.sensor_readings[j,i])

    def update_sensor_readings_with_plots(self):
        """
        Updates the sonar data array and renders the simulations as 3D plot. Used for debugging.
        """
        print("Time: {}, Nearby Obstacles: {}".format(self.total_t_steps, len(self.nearby_obstacles)))
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        ax = self.plot3D()
        ax2 = self.plot3D()
        #for obstacle in self.nearby_obstacles:
        for i in range(self.sensor_suite[0]):
            alpha = self.quadcopter.heading + self.sectors_horizontal[i]
            for j in range(self.sensor_suite[1]):
                beta = self.quadcopter.pitch + self.sectors_vertical[j]
                #s, closeness = self.calculate_object_distance(alpha, beta, obstacle)
                s=25
                #self.sensor_readings[j,i] = max(closeness, self.sensor_readings[j,i])
                color = "#05f07a"# if s >= self.sonar_range else "#a61717"
                s = np.linspace(0, s, 100)
                x = self.quadcopter.position[0] + s*np.cos(alpha)*np.cos(beta)
                y = self.quadcopter.position[1] + s*np.sin(alpha)*np.cos(beta)
                z = self.quadcopter.position[2] - s*np.sin(beta)
                ax.plot3D(x, y, z, color=color)
                #if color == "#a61717":
                ax2.plot3D(x, y, z, color=color)
            plt.rc('lines', linewidth=3)
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.scatter3D(*self.quadcopter.position, color="y", s=40, label="AUV")
        print(np.round(self.sensor_readings,3))
        self.axis_equal3d(ax)
        ax.legend(fontsize=14)
        ax2.set_xlabel(xlabel="North [m]", fontsize=14)
        ax2.set_ylabel(ylabel="East [m]", fontsize=14)
        ax2.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax2.xaxis.set_tick_params(labelsize=12)
        ax2.yaxis.set_tick_params(labelsize=12)
        ax2.zaxis.set_tick_params(labelsize=12)
        ax2.scatter3D(*self.quadcopter.position, color="y", s=40, label="AUV")
        self.axis_equal3d(ax2)
        ax2.legend(fontsize=14)
        plt.show()

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
    def check_object_overlap(self, new_obstacle):
        """
        Checks if a new obstacle is overlapping one that already exists or the target position.
        """
        overlaps = False
        # check if it overlaps target:
        if np.linalg.norm(self.path.get_endpoint() - new_obstacle.position) < new_obstacle.radius + 5:
            return True
        # check if it overlaps already placed objects
        for obstacle in self.obstacles:
            if np.linalg.norm(obstacle.position - new_obstacle.position) < new_obstacle.radius + obstacle.radius + 5:
                overlaps = True
        return overlaps


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


    def scenario_intermediate(self):
        initial_state = self.scenario_3d_new()
        obstacle_radius = np.random.uniform(low=4,high=10)
        obstacle_coords = self.path(self.path.length/2)# + np.random.uniform(low=-obstacle_radius, high=obstacle_radius, size=(1,3))
        self.obstacles.append(Obstacle(radius=obstacle_radius, position=obstacle_coords))
        return initial_state


    def scenario_proficient(self):
        initial_state = self.scenario_3d_new()
        obstacle_radius = np.random.uniform(low=4,high=10)
        obstacle_coords = self.path(self.path.length/2)# + np.random.uniform(low=-obstacle_radius, high=obstacle_radius, size=(1,3))
        self.obstacles.append(Obstacle(radius=obstacle_radius, position=obstacle_coords))

        lengths = np.linspace(self.path.length*1/6, self.path.length*5/6, 2)
        for l in lengths:
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(l) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
            # print(self.path(l))
            # print(np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3)))
            # print(obstacle_coords)
            obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)
        return initial_state


    # def scenario_advanced(self):
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
        self.obstacles.append(Obstacle(radius=obstacle_radius, position=obstacle_coords))

        lengths = np.linspace(self.path.length*1.5/6, self.path.length*5/6, 5)
        for l in lengths:
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(l) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
            # print(self.path(l))
            # print(np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3)))
            # print(obstacle_coords)
            obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
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
        self.obstacles.append(Obstacle(radius=10, position=self.path(20)))
        return initial_state


    def scenario_test(self):
        initial_state = self.scenario_test_path()
        points = np.linspace(self.path.length/4, 3*self.path.length/4, 3)
        self.obstacles.append(Obstacle(radius=10, position=self.path(self.path.length/2)))
        return initial_state


    def scenario_test_current(self):
        initial_state = self.scenario_test()
        self.current = Current(mu=0, Vmin=0.75, Vmax=0.75, Vc_init=0.75, alpha_init=np.pi/4, beta_init=np.pi/6, t_step=0) # Constant velocity current (reproducability for report)
        return initial_state


    def scenario_horizontal_test(self):
        waypoints = [(0,0,0), (50,0.1,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        self.obstacles = []
        for i in range(7):
            y = -30+10*i
            self.obstacles.append(Obstacle(radius=5, position=[50,y,0]))
        init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-5, high=5, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        return initial_state


    def scenario_vertical_test(self):
        waypoints = [(0,0,0), (50,0,1), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        self.obstacles = []
        for i in range(7):
            z = -30+10*i
            self.obstacles.append(Obstacle(radius=5, position=[50,0,z]))
        init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-5, high=5, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        return initial_state


    def scenario_deadend_test(self):
        waypoints = [(0,0,0), (50,0.5,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        radius = 10
        angles = np.linspace(-90, 90, 10)*np.pi/180
        obstalce_radius = (angles[1]-angles[0])*radius/2
        for ang1 in angles:
            for ang2 in angles:
                x = 45 + radius*np.cos(ang1)*np.cos(ang2)
                y = radius*np.cos(ang1)*np.sin(ang2)
                z = -radius*np.sin(ang1)
                self.obstacles.append(Obstacle(obstalce_radius, [x, y, z]))
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
        # init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        self.obstacles.append(Obstacle(radius=100, position=[0,0,0]))
        return initial_state