from turtle import up
import numpy as np
import pandas as pd
import gym
import gym_quad.utils.geomutils as geom
import gym_quad.utils.state_space as ss
import matplotlib.pyplot as plt
import skimage.measure
import time

from gym_quad.objects.quad import Quad
from gym_quad.objects.current3d import Current
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints
from gym_quad.objects.obstacle3d import Obstacle
from gym_quad.utils.controllers import PI, PID


class WaypointPlanner(gym.Env):
    """
    Creates an environment with a quadcopter, a path and obstacles.
    """
    def __init__(self, env_config, scenario="line", seed=None):
        np.random.seed(0)
        
        for key in env_config:
            setattr(self, key, env_config[key])
        self.action_space = gym.spaces.Box(
            low = np.array([-10]*2*self.n_fictive_waypoints, dtype=np.float32),
            high = np.array([10]*2*self.n_fictive_waypoints, dtype=np.float32),
            dtype = np.float32
        )

        self.perception_space = gym.spaces.Box(
            low = 0,
            high = 1,
            shape = (1, self.sensor_suite[0], self.sensor_suite[1]),
            dtype = np.float32
        )
        self.navigation_space = gym.spaces.Box(
            low = -np.inf, # Try -1
            high = np.inf, # Try +1
            shape = (1, self.n_obs_states + 3*self.n_fictive_waypoints),
            dtype = np.float32
        )
        self.observation_space = gym.spaces.Dict({
            'perception': self.perception_space,
            'navigation': self.navigation_space
        })
        
        self.n_sensor_readings = self.sensor_suite[0]*self.sensor_suite[1]
        max_horizontal_angle = self.sensor_span[0]/2
        max_vertical_angle = self.sensor_span[1]/2
        self.sectors_horizontal = np.linspace(-max_horizontal_angle*np.pi/180, max_horizontal_angle*np.pi/180, self.sensor_suite[0])
        self.sectors_vertical =  np.linspace(-max_vertical_angle*np.pi/180, max_vertical_angle*np.pi/180, self.sensor_suite[1])
        
        self.scenario = scenario
        self.scenario_switch = {
            # Training scenarios
            "line": self.scenario_line,
            "line_new": self.scenario_line_new,
            "horizontal": self.scenario_horizontal,
            "horizontal_new": self.scenario_horizontal_new,
            "3d": self.scenario_3d,
            "3d_new": self.scenario_3d_new,
            #"helix": self.scenario_helix,
            "intermediate": self.scenario_intermediate,
            "proficient": self.scenario_proficient,
            "advanced": self.scenario_advanced,
            "expert": self.scenario_expert,
            # Testing scenarios
            "test_path": self.scenario_test_path,
            "test": self.scenario_test,
            "test_current": self.scenario_test_current,
            #"horizontal": self.scenario_horizontal_test,
            "vertical": self.scenario_vertical_test,
            "deadend": self.scenario_deadend_test
        }
        self.reset()
    

    def get_stats(self):
        return {"reward_path_following":self.reward_path_following_sum}#,"obs":self.past_obs,"states":self.past_states,"errors":self.past_errors}


    def reset(self):
        """
        Resets environment to initial state. 
        """
        self.quadcopter = None
        self.path = None
        self.path_generated = None
        self.u_error = None
        self.e = None
        self.h = None
        self.chi_error = None
        self.upsilon_error = None
        self.waypoint_index = 0
        self.prog = 0
        self.path_prog = []
        self.success = False

        self.obstacles = []
        self.nearby_obstacles = []
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        self.collided = False
        self.penalize_control = 0.0

        self.prev_position_error = [0, 0, 0]
        self.total_position_error = [0, 0, 0]

        self.passed_waypoints = np.zeros((1, 3), dtype=np.float32)
        self.fictive_waypoint_at_end = [False]*self.n_fictive_waypoints

        self.observation = None
        self.past_states = []
        self.past_actions = []
        self.past_errors = []
        self.past_obs = []
        self.past_prog=[]
        self.time = []
        self.total_t_steps = 0
        self.reward = 0
        self.reward_path_following_sum=0
        self.progression=[]
        self.generate_environment()
        self.update_control_errors()
        self.observation = self.observe()
        return self.observation


    def generate_environment(self):
        """
        Generates environment with a quadcopter, potentially ocean current and a 3D path.
        """     
        # Generate training/test scenario
        scenario = self.scenario_switch.get(self.scenario, lambda: print("Invalid scenario"))
        init_state = scenario()
        
        # Generate Quadcopter
        self.quadcopter = Quad(self.step_size, init_state)


    def step(self, action):
        """
        Simulates the environment one time-step. 
        """
        # Simulate quadcopter dynamics one time-step and save action and state
        self.update_control_errors()

        action = np.array([[0.0, 0.0, 0.0]]*self.n_fictive_waypoints, dtype=np.float32)
        if self.total_t_steps % 100 == 0:
            self.alternative_path = self.generate_alternative_path(action)

        F = self.path_following_controller(self.alternative_path)
        self.quadcopter.step(F)

        self.progression.append(self.prog/self.path.length)
        self.past_states.append(np.copy(self.quadcopter.state))
        # self.past_errors.append(np.array([self.e,self.h]))
        self.past_errors.append(np.array([self.u_error, self.chi_error, self.e, self.upsilon_error, self.h]))
        self.past_actions.append(self.quadcopter.input)

        if self.path:
            self.prog = self.path.get_closest_u(self.quadcopter.position, self.waypoint_index)
            self.path_prog.append(self.prog)
            
            # Check if a waypoint is passed
            k = self.path.get_u_index(self.prog)
            if k > self.waypoint_index:
                print("Passed waypoint {:d}".format(k+1), self.path.waypoints[k], "\tquad position:", self.quadcopter.position)
                self.passed_waypoints = np.vstack((self.passed_waypoints, self.path.waypoints[k]))
                self.waypoint_index = k
        
        # Calculate reward based on observation and actions
        done, step_reward = self.step_reward(self.observation, action)
        info = {}

        # Make next observation
        self.observation = self.observe()
        self.past_obs.append(self.observation['navigation'])
        self.past_prog.append([self.prog/self.path.length])

        # Save sim time info
        self.total_t_steps += 1
        self.time.append(self.total_t_steps*self.step_size)

        return self.observation, step_reward, done, info


    def observe(self):
        """
        Returns observations of the environment. 
        """
        self.fictive_waypoints = self.generate_fictive_waypoints(distance = self.distance_fictive_waypoints)

        obs = np.zeros(self.n_obs_states + 3*self.n_fictive_waypoints)
        obs[0:3] = self.quadcopter.position
        obs[3:6] = self.quadcopter.attitude
        obs[6:6+3*self.n_fictive_waypoints] = self.fictive_waypoints.flatten()

        # Update nearby obstacles and calculate distances
        self.update_nearby_obstacles()
        self.update_sensor_readings()

        return {'perception':[self.sensor_readings], 'navigation': [obs]}


    def step_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode should end. 
        """
        done = False
        step_reward = 0

        dist_from_path = np.linalg.norm(self.path(self.prog) - self.quadcopter.position)
# 
        reward_path_following = - self.reward_path_following * np.log(dist_from_path)
        reward_collision_avoidance = self.penalize_obstacle_closeness()

        self.reward_path_following_sum += self.lambda_reward * reward_path_following
        step_reward = self.lambda_reward * reward_path_following + (1 - self.lambda_reward) * reward_collision_avoidance
        self.reward += step_reward
    
        # Check collision
        for obstacle in self.nearby_obstacles:
            if np.linalg.norm(obstacle.position - self.quadcopter.position) <= obstacle.radius + self.quadcopter.safety_radius:
                self.collided = True
        
        end_cond_1 = self.reward < self.min_reward
        end_cond_2 = self.total_t_steps >= self.max_t_steps
        end_cond_3 = np.linalg.norm(self.path.get_endpoint()-self.quadcopter.position) < self.accept_rad and self.waypoint_index == self.n_waypoints-2
        end_cond_4 = abs(self.prog - self.path.length) <= self.accept_rad/2.0
        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            if end_cond_3:
                print("Quadcopter reached target!")
                self.success = True
            elif self.collided:
                print("Quadcopter collided!")
                self.success = False
            done = True
        return done, step_reward

    
    def path_following_controller(self, alternative_path : QPMI) -> np.array:
        """
        Path following controller.
        Calculate desired acceleration for position control: PID-controller + velocity-control along the tangent + feedforward acceleration of path.
        Attitude control: PD-controller.

        Parameters:
        ----------
        alternative_path : QPMI
            Alternative path for the quadcopter to follow, deviating from the original path to avoid obstacles.

        Returns:
        -------
        F : np.array
            Thrust inputs needed to follow the alternative path.
        """
        prog = alternative_path.get_closest_u(self.quadcopter.position, self.waypoint_index, margin=self.distance_fictive_waypoints)
        self.chi_p, self.upsilon_p = alternative_path.get_direction_angles(prog)

        # Calculate tracking errors
        v = alternative_path.calculate_gradient(prog)
        a = alternative_path.calculate_acceleration(prog)
        e_p = alternative_path(prog) - self.quadcopter.position
        e_p_dot = (e_p - self.prev_position_error) / self.step_size
        self.total_position_error += e_p
        e_v = self.cruise_speed * v / np.linalg.norm(v) - geom.Rzyx(*self.quadcopter.attitude) @ self.quadcopter.velocity
        self.prev_position_error = e_p

        # Define PD control weights for position and attitude
        K_p_pos = np.diag([3, 3, 6])
        K_d_pos = np.diag([0.5, 0.5, 0.5])
        K_i_pos = np.diag([0.01, 0.01, 0.01])
        K_v = np.diag([0.5, 0.5, 0.5])

        omega_n = 9 # Natural frequency
        xi = 1      # Damping ratio, 1 -> critically damped

        K_p_att = np.diag([ss.I_x * omega_n**2, ss.I_y * omega_n**2, ss.I_z * omega_n**2])
        K_d_att = np.diag([2 * ss.I_x * xi * omega_n, 2 * ss.I_y * xi * omega_n, 2 * ss.I_z * xi * omega_n])

        # Calculate desired accelerations and attitudes
        a_des = K_p_pos @ e_p + K_d_pos @ e_p_dot + K_i_pos @ self.total_position_error + K_v @ e_v + a
        b_x = a_des[0] / (a_des[2] + ss.g + (ss.d_w*self.quadcopter.heave - ss.d_u*self.quadcopter.surge)/ss.m)
        b_y = a_des[1] / (a_des[2] + ss.g + (ss.d_w*self.quadcopter.heave - ss.d_v*self.quadcopter.sway)/ss.m)
        phi_des   = geom.ssa(b_x * np.sin(self.chi_p) - b_y * np.cos(self.chi_p))
        theta_des = geom.ssa(b_x * np.cos(self.chi_p) + b_y * np.sin(self.chi_p))
        e_att = geom.ssa(np.array([phi_des, theta_des, self.chi_p]) - self.quadcopter.attitude)
        e_angvel = np.array([0.0, 0.0, 0.0]) - self.quadcopter.angular_velocity

        # Calculate desired inputs
        u_des = np.array([0.0, 0.0, 0.0, 0.0])
        u_des[0] = ss.m * (a_des[2] + ss.g) + ss.d_w*self.quadcopter.heave
        u_des[1:] = K_p_att @ e_att + K_d_att @ e_angvel

        F = np.linalg.inv(ss.B()[2:]).dot(u_des)
        F = np.clip(F, ss.thrust_min, ss.thrust_max)

        return F


    def generate_fictive_waypoints(self, distance : float = 1.0) -> np.array:
        """
        Generates fictive waypoints along the path, with certain distance.

        Parameters:
        ----------
        n_waypoints : int
            Number of waypoints to be generated. The first waypoint is at the point of the path closest to the quadcopter.
        distance : float
            Distance in meters between each of the waypoints.

        Returns:
        -------
        waypoints : np.array
            Generated waypoints.
        """

        waypoints = np.zeros((self.n_fictive_waypoints, 3))

        prog = self.path.get_closest_u(self.quadcopter.position, self.waypoint_index)

        for i in range(self.n_fictive_waypoints):
            dist = prog + (i+1) * distance
            if self.fictive_waypoint_at_end[i]:
                waypoints[i,:] = self.path.get_endpoint()
            elif dist <= self.path.length - distance:
                waypoints[i,:] = self.path(dist)
            else:
                self.fictive_waypoint_at_end[i] = True
                waypoints[i,:] = self.path.get_endpoint()
                print("ENDPOINT")

        return waypoints
    

    def generate_alternative_path(self, action : np.array) -> QPMI:
        """
        Generate a new path for the quadcopter to follow to avoid obstacles. In the case of no obstacles, this should coincide with the original path.

        Parameters:
        ----------
        action : np.array
            Position deviation from the fictive waypoints along the original path.
        
        Returns:
        -------
        alternative_path : QPMI
            New path for the quadcopter to follow.
        """

        # Add result from neural network
        waypoint_deviations = action.reshape(-1, 3)

        path_waypoints = self.path.waypoints
        fictive_waypoints = self.fictive_waypoints + waypoint_deviations

        # Add waypoint indices to all points
        path_waypoints = np.append(path_waypoints, np.array(range(self.n_waypoints))[...,None], axis=1)
        fictive_waypoints = np.append(fictive_waypoints, np.array([self.waypoint_index]*self.n_fictive_waypoints)[...,None], axis=1)

        # When a fictive waypoint is at the end of the path, remove it
        fictive_waypoints_temp = np.copy(fictive_waypoints)
        fictive_waypoints = []
        for waypoint in fictive_waypoints_temp:
            if not np.allclose(waypoint[:3], self.path.get_endpoint()):
                fictive_waypoints.append(waypoint)
        fictive_waypoints = np.array(fictive_waypoints)

        # If all fictive waypoints are at the end of the path, disregard them
        if np.size(fictive_waypoints) == 0:
            waypoints = path_waypoints
            fictive_waypoint_prog = self.path.length
            fictive_waypoint_prog = -np.inf
        else:
            waypoints = np.vstack((path_waypoints, fictive_waypoints))
            fictive_waypoint_prog = self.path.get_closest_u(fictive_waypoints[-1,:3], self.waypoint_index)
        
        # Sort all waypoints so they are in correct order
        df = pd.DataFrame({'X' : [], 'Y' : [], 'Z' : [], 'Index': [], 'Prog' : []})
        for waypoint in waypoints:
            waypoint_prog = self.path.get_closest_u(waypoint[:3], int(waypoint[3]))
            new_row = pd.DataFrame([[waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint_prog]], columns=['X', 'Y', 'Z', 'Index', 'Prog'])
            df = pd.concat([df, new_row]).reset_index(drop=True)
        df = df.sort_values('Prog')
        all_waypoints = df.to_numpy()[:,0:4]
        
        # When there are path-waypoints between the quadcopter and the fictive waypoints, remove them
        waypoints = []
        for waypoint in all_waypoints:
            waypoint_prog = self.path.get_closest_u(waypoint[:3], int(waypoint[3]))
            if waypoint_prog - 1e-5 > self.prog and waypoint_prog < fictive_waypoint_prog and waypoint not in fictive_waypoints:
                continue
            else:
                waypoints.append(waypoint[:3])
        waypoints = np.array(waypoints)
        # print(waypoints)

        alternative_path = QPMI(waypoints)
        return alternative_path
    

    def get_chi_upsilon(self,la_dist):
        chi_r = np.arctan2(self.e, la_dist)
        upsilon_r = np.arctan2(self.h, np.sqrt(self.e**2 + la_dist**2))
        chi_d = self.chi_p + chi_r
        upsilon_d =self.upsilon_p + upsilon_r
        chi_error = np.clip(geom.ssa(self.quadcopter.chi - chi_d)/np.pi, -1, 1)
        # chi_error = geom.ssa(self.quadcopter.chi - chi_d)
        upsilon_error = np.clip(geom.ssa(self.quadcopter.upsilon - upsilon_d)/np.pi, -1, 1)
        # upsilon_error = geom.ssa(self.quadcopter.upsilon - upsilon_d)
        return chi_error,upsilon_error


    def update_control_errors(self):
        # Update cruise speed error
        self.u_error = np.clip((self.cruise_speed - self.quadcopter.velocity[0])/2, -1, 1)
        self.chi_error = 0.0
        self.e = 0.0
        self.upsilon_error = 0.0
        self.h = 0.0

        # Get path course and elevation
        s = self.prog
        self.chi_p, self.upsilon_p = self.path.get_direction_angles(s)

        # Calculate tracking errors
        SF_rotation = geom.Rzyx(0,self.upsilon_p,self.chi_p)

        epsilon = np.transpose(SF_rotation).dot(self.quadcopter.position-self.path(self.prog))
        e = epsilon[1]
        h = epsilon[2]
        # Calculate course and elevation errors from tracking errors
        #self.e = np.clip(e/12, -1, 1)
        self.e = e
        #self.h = np.clip(h/12, -1, 1)
        self.h = h
        self.chi_error,self.upsilon_error=self.get_chi_upsilon(self.la_dist)


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


    def penalize_obstacle_closeness(self):
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
        return - reward_colav / sensor_suite_correction


    def plot_section3(self):
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


    def plot3D(self, wps_on=True):
        """
        Returns 3D plot of path and obstacles.
        """
        ax = self.path.plot_path(wps_on)
        for obstacle in self.obstacles:    
            ax.plot_surface(*obstacle.return_plot_variables(), color='r')
        return self.axis_equal3d(ax)


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
        return ax


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
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    
    def scenario_3d_new(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'3d_new')
        self.path = QPMI(waypoints)
        init_pos=[0, 0, 0]
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    
    """
    def scenario_helix(self):
        initial_state = np.zeros(6)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0) #Current object with zero velocity
        waypoints = generate_random_waypoints(self.n_waypoints,'helix')
        self.path = QPMI(waypoints)
        init_pos = helix_param(0)
        #init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    """

    def scenario_intermediate(self):
        initial_state = self.scenario_3d_new()
        rad = np.random.uniform(4, 10)
        pos = self.path(self.path.length/2)
        self.obstacles.append(Obstacle(radius=rad, position=pos))
        lengths = np.linspace(self.path.length*1/3, self.path.length*2/3, self.n_int_obstacles)
        for l in lengths:
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(l)
            obstacle = Obstacle(obstacle_radius, obstacle_coords)
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)
        return initial_state


    def scenario_proficient(self):
        initial_state = self.scenario_intermediate()
        lengths = np.random.uniform(self.path.length*1/3, self.path.length*2/3, self.n_pro_obstacles)
        n_checks = 0
        while len(self.obstacles) < self.n_pro_obstacles and n_checks < 1000:
            for l in lengths:
                obstacle_radius = np.random.uniform(low=4,high=10)
                obstacle_coords = self.path(l)
                obstacle = Obstacle(obstacle_radius, obstacle_coords)
                if self.check_object_overlap(obstacle):
                    n_checks += 1
                    continue

                else:
                    self.obstacles.append(obstacle)
        return initial_state


    def scenario_advanced(self):
        initial_state = self.scenario_proficient()
        while len(self.obstacles) < self.n_adv_obstacles: # Place the rest of the obstacles randomly
            s = np.random.uniform(self.path.length*1/3, self.path.length*2/3)
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(s) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
            obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)
        return initial_state


    def scenario_expert(self):
        initial_state = self.scenario_advanced()
        self.current = Current(mu=0.2, Vmin=0.5, Vmax=1.0, Vc_init=np.random.uniform(0.5, 1), \
                                    alpha_init=np.random.uniform(-np.pi, np.pi), beta_init=np.random.uniform(-np.pi/4, np.pi/4), t_step=self.step_size)
        self.penalize_control = 1.0
        return initial_state


    def scenario_test_path(self):
        # test_waypoints = np.array([np.array([0,0,0]), np.array([1,1,0]), np.array([9,9,0]), np.array([10,10,0])])
        # test_waypoints = np.array([np.array([0,0,0]), np.array([5,0,0]), np.array([10,0,0]), np.array([15,0,0])])
        test_waypoints = np.array([np.array([0,0,0]), np.array([5,0,0]), np.array([10,0,0]), np.array([15,0,0])])
        self.n_waypoints = len(test_waypoints)
        self.path = QPMI(test_waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
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
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        self.obstacles = []
        for i in range(7):
            y = -30+10*i
            self.obstacles.append(Obstacle(radius=5, position=[50,y,0]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state


    def scenario_vertical_test(self):
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        self.obstacles = []
        for i in range(7):
            z = -30+10*i
            self.obstacles.append(Obstacle(radius=5, position=[50,0,z]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state


    def scenario_deadend_test(self):
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        radius = 25
        angles = np.linspace(-90, 90, 10)*np.pi/180
        obstalce_radius = (angles[1]-angles[0])*radius/2
        for ang1 in angles:
            for ang2 in angles:
                x = 30+radius*np.cos(ang1)*np.cos(ang2)
                y = radius*np.cos(ang1)*np.sin(ang2)
                z = -radius*np.sin(ang1)
                self.obstacles.append(Obstacle(obstalce_radius, [x, y, z]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state