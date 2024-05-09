import numpy as np
import unittest

import sys
import os
#Use os and sys to access the modules inside gym_quad (imports below are from gym_quad)
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent and grandparent directory of the current script
parent_dir = os.path.dirname(script_dir)
grand_parent_dir = os.path.dirname(parent_dir)
# Add the parent directory to the Python path
sys.path.append(grand_parent_dir)

import gym_quad.utils.geomutils as geom
import gym_quad.utils.state_space as ss
from gym_quad.utils.ODE45JIT import j_Rzyx
from gym_quad.objects.quad import Quad
from old.controllers2 import LeeVelocityController

    # "s_max": 0.5, # Maximum speed of the quadcopter
    # "i_max": 0.5, # Maximum inclination angle of commanded velocity wrt x-axis
    # "omega_max": 0.5, # Maximum commanded yaw rate

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi


class TestController(unittest.TestCase):

    def setUp(self):
        self.quadcopter = Quad(0.01,np.zeros(6))
        self.quadcopter.state[2] = 1 # Set the initial height to 1m
        self.s_max = 2
        self.i_max = deg2rad(80/2) 
        self.r_max = deg2rad(30) 
        self.total_t_steps = 0
        self.step_size = 0.01

    def PID_PD_Lin_velocity_controller(self, action):
        """
        The velocity controller for the quadcopter.
        Based on small angle approximation linearizing the quadcopter dynamics about the hover point.

        Parameters:
        ----------
        action : np.array (3,)
        The action input from the RL agent. 
        a1: reference speed in body frame, 
        a2: inclination of velocity vector wrt x-axis of body
        a3: yaw rate in body frame relative to world frame
        
        Returns:
        -------
        F : np.array (4,)
        The thrust inputs required to follow path and avoid obstacles according to the action of the DRL agent.
        """
        #Using hyperparam of s_max, i_max, r_max to clip the action to get the commanded velocity and yaw rate
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1]*self.i_max)#*self.i_max)
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1]*self.i_max)#*self.i_max)
        cmd_r = self.r_max * action[2]

        self.cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r])

        #For the attitude the natural frequency and damping ratio are set to 9 and 1 respectively    
        omega_n = 9 # Natural frequency
        xi = 1      # Damping ratio, 1 -> critically damped

        K_p_att = np.diag([ss.I_x * omega_n**2, ss.I_y * omega_n**2, ss.I_z * omega_n**2])
        K_d_att = np.diag([2 * ss.I_x * xi * omega_n, 2 * ss.I_y * xi * omega_n, 2 * ss.I_z * xi * omega_n])

        #Prop, damp and integral gains for the linear velocity
        #OLD gains get fairish tracking of x and z but not y
        #These make the attitude reach angles of 30 deg where i think the small angle assumption begins to fail.
        # K_p = np.diag([6, 0.8, 4])
        # K_d = np.diag([0.5, 0.2, 0.5])
        # K_i = np.diag([4, 0.3, 1.5])

        K_p = np.diag([1.2, 1.2, 1.2])
        K_d = np.diag([0.1, 0.1, 0.1])
        K_i = np.diag([2, 2, 2])

        #KP part
        vel_body = self.quadcopter.velocity
        # body velocity available by integrating linear accelerations from IMU
        # attitude available from integrating up the angular rates from the IMU after calibration ofc

        # Try vehicle frame velcoity control!
        # euler_angles = self.quadcopter.attitude
        # vehicle_frame_euler = np.zeros_like(euler_angles)
        # vehicle_frame_euler[2] = euler_angles[2]
        # vehicle_vels = self.quadcopter.velocity
        # vehicle_frame_transforms = geom.Rzyx(vehicle_frame_euler[0], vehicle_frame_euler[1], vehicle_frame_euler[2])
        # vehicle_frame_transforms_transpose = vehicle_frame_transforms.T
        # vehicle_frame_velocity = vehicle_frame_transforms_transpose @ vehicle_vels
        #Seemingly no change....        
        v_error = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - vel_body

        #KD part
        # v_error_dot = np.array([0.0, 0.0, 0.0])
        if self.total_t_steps > 0:
            v_error_dot = (v_error - self.prev_v_error) / self.step_size
        else:
            v_error_dot = np.array([0.0, 0.0, 0.0])
        self.prev_v_error = v_error
        
        #KI part
        total_v_error = np.zeros(3)
        if self.total_t_steps > 0:
            total_v_error += v_error * (np.absolute(self.prev_a_des) < 0.5).astype(int) * self.step_size # Anti-wind up

        a_des = K_p @ v_error + K_d @ v_error_dot + K_i @ total_v_error
        self.prev_a_des = a_des # Save the previous a_des for the next iteration to use in anti-wind up above
        #V2desired angles inspired by Carlsen's thesis assumes small angles
        #Assuming desired yaw aka heading is the same as the current yaw as the change of yaw is controlled by the yaw rate command
        yaw_des = self.quadcopter.attitude[2] #Dunno if this is fair: + cmd_r*self.step_size
        term1 = (a_des[0]/(a_des[2] + ss.g + (ss.d_w*self.quadcopter.heave - ss.d_u*self.quadcopter.surge)/ss.m))
        term2 = (a_des[1]/(a_des[2] + ss.g + (ss.d_w*self.quadcopter.heave - ss.d_v*self.quadcopter.sway)/ss.m))
        phi_des = geom.ssa(term1*np.sin(yaw_des) + term2*np.cos(yaw_des))
        theta_des = geom.ssa(term1*np.cos(yaw_des) + term2*np.sin(yaw_des))

        self.att_des = np.array([phi_des, theta_des, yaw_des]) # Save the desired attitude for plotting

        e_att = geom.ssa(np.array([phi_des, theta_des, yaw_des]) - self.quadcopter.attitude)
        
        e_angvel = np.array([0.0, 0.0, cmd_r]) - self.quadcopter.angular_velocity
        # e_angvel = geom.Rzyx(*self.quadcopter.attitude).T @ geom.Rzyx(phi_des, theta_des, yaw_des) @ np.array([0.0, 0.0, cmd_r]) - self.quadcopter.angular_velocity
        #These two e_angvel are equivalent

        torque = K_p_att @ e_att + K_d_att @ e_angvel #+ np.cross(self.quadcopter.angular_velocity,ss.Ig@self.quadcopter.angular_velocity)
        u = np.zeros(4)
        u[0] = ss.m * (a_des[2] + ss.g) + ss.d_w*self.quadcopter.heave #thurst force in body z direction i.e. total thrust
        u[1:] = torque

        F = np.linalg.inv(ss.B()[2:]).dot(u)
        F = np.clip(F, ss.thrust_min, ss.thrust_max)    

        return F

    # THIRD ATTEMPT AT THE GEOMETRIC CONTROLLER ILL LEAVE IT HERE FOR SOME TIME AND USE THE ONE ABOVE PER NOW UNTIL A BETTER LV CONTROLLER IS FOUND    
    def geometric_velocity_controller(self, action):
            """
            The velocity controller for the quadcopter.
            Based on multirotor paper https://ieeexplore.ieee.org/document/6289431 and geometric paper
            Nonlinear control that tackles the quadcopter dynamics directly without linearization.

            Parameters:
            ----------
            action : np.array (3,)
            The action input from the RL agent. 
            a1: reference speed in BODY frame, 
            a2: inclination of velocity vector wrt x-axis of body
            a3: yaw rate of body frame relative to world frame
            
            Returns:
            -------
            F : np.array (4,)
            The thrust inputs required to follow path and avoid obstacles according to the action of the DRL agent.
            """
            #Using hyperparam of s_max, i_max, r_max to clip the action to get the commanded velocity and yaw rate
            cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1]*self.i_max)#*self.i_max)
            cmd_v_y = 0
            cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1]*self.i_max)#*self.i_max)
            cmd_r = self.r_max * action[2]
            self.cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r])

            #For the attitude the natural frequency and damping ratio are set to 9 and 1 respectively    
            # omega_n = 9 # Natural frequency
            # xi = 1      # Damping ratio, 1 -> critically damped
            # K_p_att = np.diag([ss.I_x * omega_n**2, ss.I_y * omega_n**2, ss.I_z * omega_n**2])
            # K_d_att = np.diag([2 * ss.I_x * xi * omega_n, 2 * ss.I_y * xi * omega_n, 2 * ss.I_z * xi * omega_n])
            # K_p_att = np.diag([1, 1, 1])
            # K_d_att = np.diag([0.5, 0.2, 0.5])
            kp_att = 1
            kd_att = 0.5

            #Prop, damp and integral gains for the linear velocity
            K_p = np.diag([1,1,1])
            K_d = np.diag([0.5, 0.2, 0.5])
            K_i = np.diag([4, 0.3, 1.5])

            #KP part
            vel_body = self.quadcopter.velocity #Also given by self.quadcopter.position_dot
            v_error = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - vel_body

            #KD part
            if self.total_t_steps > 0:
                v_error_dot = (v_error - self.prev_v_error) / self.step_size
            else:
                v_error_dot = np.array([0.0, 0.0, 0.0])
            self.prev_v_error = v_error
            
            #KI part
            total_v_error = np.zeros(3)
            if self.total_t_steps > 0:
                total_v_error += v_error * (np.absolute(self.prev_a_des) < 0.5).astype(int) * self.step_size # Anti-wind up

            a_des = K_p @ v_error + K_d @ v_error_dot + K_i @ total_v_error
            self.prev_a_des = a_des 
            
            u = np.zeros(4)
            fdz = ss.d_w*self.quadcopter.heave #Drag force in z direction
            u[0] = (ss.m * (a_des[2] + ss.g) + fdz) #thurst force in body z direction i.e. total thrust

            #Take 1
            # a_3 = np.array([0, 0, 1])
            # yaw_des = geom.ssa(cmd_r*self.step_size + self.quadcopter.attitude[2])
            # e_1 = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
            
            # v_dot = self.quadcopter.state_dot(self.quadcopter.state)[6:9] 
            # b3 = (v_dot - ss.g*a_3- fdz/ss.m) / np.linalg.norm(v_dot - ss.g*a_3 - fdz/ss.m)
            # b2 = (np.cross(b3, e_1))/np.linalg.norm(np.cross(b3, e_1))
            # b1 = np.cross(b2, b3)
            # R_des = np.array([b1, b2, b3])

            forces_command = a_des*ss.m +ss.m*ss.g*np.array([0, 0, 1]) + np.array([0, 0, fdz])
            
            # Calculate euler setpoints
            #Take 2
            c_phi_s_theta = forces_command[0]
            s_phi = -forces_command[1]
            c_phi_c_theta = forces_command[2]
            pitch_setpoint = np.arctan2(c_phi_s_theta, c_phi_c_theta)
            roll_setpoint = np.arctan2(s_phi, np.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
            yaw_setpoint = self.quadcopter.attitude[2]

            self.att_des = np.array([roll_setpoint, pitch_setpoint, yaw_setpoint]) # Save the desired attitude for plotting
            
            R_des = geom.Rzyx(roll_setpoint, pitch_setpoint, yaw_setpoint)
            Rmat = geom.Rzyx(*self.quadcopter.attitude)

            e_Rx = 1/2*(R_des.T @ Rmat - Rmat.T @ R_des)
            e_att = geom.vee_map(e_Rx)
            e_att = np.reshape(e_att, (3,))

            des_angvel = np.array([0.0, 0.0, cmd_r])
            e_angvel = self.quadcopter.angular_velocity - Rmat.T @ R_des @ des_angvel

            torque = ss.Ig @ (kp_att * e_att + kd_att * e_angvel) + np.cross(self.quadcopter.angular_velocity,ss.Ig@self.quadcopter.angular_velocity)
            
            u[1:] = torque

            F = np.linalg.inv(ss.B()[2:]).dot(u)
            F = np.clip(F, ss.thrust_min, ss.thrust_max)    

            return F
    
    def leectrl(self, action):
            """
            Lee velocity controller
            :param robot_state: array of shape (12) with state of the robot
            :param command_actions: array of shape (4) with desired velocity setpoint in vehicle frame and yaw_rate command in vehicle frame
            :return: m*g normalized thrust and inertial normalized torques
            """
            #Using hyperparam of s_max, i_max, r_max to clip the action to get the commanded velocity and yaw rate
            cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1]*self.i_max)#*self.i_max)
            cmd_v_y = 0
            cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1]*self.i_max)#*self.i_max)
            cmd_r = self.r_max * action[2]

            self.cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r])
            # LEE VELOCITY CONTROLLER
            state = self.quadcopter.state
            
            k_vel = 0.5
            k_rot = 0.5
            k_angvel = 0.5

            LeeVelCtrl = LeeVelocityController(k_vel, k_rot, k_angvel)
            
            output_thrust_mass_normalized, output_torques_inertia_normalized = LeeVelCtrl(state, self.cmd)

            self.att_des = LeeVelCtrl.des_att
            fdz = ss.d_w*self.quadcopter.heave #Drag force in z direction
            f = output_thrust_mass_normalized * (ss.W) + fdz
            T = output_torques_inertia_normalized #* ss.Ig.trace()
            #* ss.W * ss.l

            u_des = np.array([f, T[0][0], T[1][0], T[2][0]])

            F = np.linalg.inv(ss.B()[2:]).dot(u_des)
            F = np.clip(F, ss.thrust_min, ss.thrust_max)

            return F
            ##
    
    def geom_ctrlv2(self, action):
        #Translate the action to the desired velocity and yaw rate
        #Try without limiting for easier debugging
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1]*self.i_max)#*self.i_max
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1]*self.i_max)#*self.i_max
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

        #Body frame velocity control
        ev = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - self.quadcopter.velocity 
        #Which one to use? vehicle_vels or self.quadcopter.velocity

        #World frame velocity control
        # ev = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - self.quadcopter.position_dot
        
        #Thrust command (along body z axis)  
        f = kv*ev + ss.m*ss.g*e3 + ss.d_w*self.quadcopter.heave*e3
        thrust_command = np.dot(f, R[2])

        ###---###
        #Rd calculation as in Kulkarni aerial gym (works fairly well)
        c_phi_s_theta = f[0]
        s_phi = -f[1]
        c_phi_c_theta = f[2]

        pitch_setpoint = np.arctan2(c_phi_s_theta, c_phi_c_theta)
        roll_setpoint = np.arctan2(s_phi, np.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
        yaw_setpoint = self.quadcopter.attitude[2]
        Rd = geom.Rzyx(roll_setpoint, pitch_setpoint, yaw_setpoint)
        self.att_des = np.array([roll_setpoint, pitch_setpoint, yaw_setpoint]) # Save the desired attitude for plotting
        ###---###

        ##-#-#
        ##Calculating Rd from papers(Does not work per now): 
        ##OG source:https://arxiv.org/abs/1003.2005v1 
        ##I prefer https://arxiv.org/pdf/2306.09651.pdf 
        # r3 = f/np.linalg.norm(f)
        # yawdes = self.quadcopter.attitude[2]
        # headingdes = np.array([np.cos(yawdes), np.sin(yawdes), 0])
        # r2 = np.cross(r3, headingdes)/np.linalg.norm(np.cross(r3, headingdes))
        # r1 = np.cross(r2, r3)
        # Rd = np.array([r1,r2,r3])
        # self.att_des = geom.R2Euler(Rd)
        # self.att_des = np.array([self.att_des[0], self.att_des[1], self.att_des[2]]) #potentially -roll and -pitch
        ##-#-#


        eR = 1/2*(Rd.T @ R - R.T @ Rd)
        eatt = geom.vee_map(eR)
        eatt = np.reshape(eatt, (3,))

        des_angvel = np.array([0.0, 0.0, cmd_r])

        ###---###
        #Kulkarni approach desired angular rate in body frame:
        s_pitch = np.sin(self.quadcopter.attitude[1])
        c_pitch = np.cos(self.quadcopter.attitude[1])
        s_roll = np.sin(self.quadcopter.attitude[0])
        c_roll = np.cos(self.quadcopter.attitude[0])
        R_euler_to_body = np.array([[1, 0, -s_pitch],
                                    [0, c_roll, s_roll*c_pitch],
                                    [0, -s_roll, c_roll*c_pitch]]) #Uncertain about how this came to be
        
        des_angvel_body = R_euler_to_body @ des_angvel
        
        #There is a difference in desired angular velocity between Kulkarnis and my direct approach
        # if self.total_t_steps < 500: 
        #     print("timestep: ", self.total_t_steps)
        #     print(des_angvel_body,des_angvel,"\n")

        eangvel = self.quadcopter.angular_velocity - R.T @ (Rd @ des_angvel_body) #Kulkarni approach
        ###---###

        # eangvel = self.quadcopter.angular_velocity - R.T @ Rd @ des_angvel #My approach use des_angvel directly

        torque = -kR*eatt - kangvel*eangvel + np.cross(self.quadcopter.angular_velocity,ss.Ig@self.quadcopter.angular_velocity)

        u = np.zeros(4)
        u[0] = thrust_command
        u[1:] = torque

        F = np.linalg.inv(ss.B()[2:]).dot(u)
        F = np.clip(F, ss.thrust_min, ss.thrust_max)
        return F

    def active_geom_ctrlv2(self, action): 
        #Translate the action to the desired velocity and yaw rate
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1]*self.i_max)
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1]*self.i_max)
        cmd_r = self.r_max * action[2]
        self.cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r]) #For plotting

        kv = 2.5
        kR = 0.8
        kangvel = 0.8

        e3 = np.array([0, 0, 1]) #z-axis basis

        # R = geom.Rzyx(*self.quadcopter.attitude) #OLD
        R = j_Rzyx(*self.quadcopter.attitude)  #NEW using jit version

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
        Rd = j_Rzyx(roll_setpoint, pitch_setpoint, yaw_setpoint) #NEW using jit version
        self.att_des = np.array([-roll_setpoint, pitch_setpoint, yaw_setpoint]) # Save the desired attitude for plotting during testing
        #TODO there is something funky about the roll setpoint, it is inverted for some reason

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
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    Test = TestController()
    Test.setUp()

    #Velocity and yaw rate reference
    vel_ref = []
    incline_ref = []
    yaw_rate_ref = []
    tot_time = 1500 # 1500*0.01 = 150 seconds
    # referencetype = "z_velocity" # "hover", "x_velocity", "z_velocity", "yaw_rate", "yaw_rate_xvel", "velocity_step", "incline_step", "velx_yaw_rate_velx"
    # referencetype = "yaw_rate_xvel"
    # referencetype = "velx_yaw_rate_velx"
    # referencetype = "complex"
    referencetype = "x_velocity"
    referencetype = "manual_input" #Mimics manual input

    save_forces = True #If flipped saves the forces to file so can be used in e.g. IMU

    if referencetype == "hover": # Creates reference for hovering at 1m height check initial state of the quadcopter in setUp
        for t in range(tot_time):
            vel_ref.append(-1)
            yaw_rate_ref.append(0)
            incline_ref.append(0)
    elif referencetype == "x_velocity": # Creates reference for moving in the x direction then stopping
        for t in range(tot_time):
            if t>0 and t<500:
                vel_ref.append(0.2)
            else:
                vel_ref.append(-1)
            yaw_rate_ref.append(0)
            incline_ref.append(0)
    elif  referencetype == "z_velocity": #Creates reference for moving in the z direction
        for t in range(tot_time):
            vel_ref.append(1)
            incline_ref.append(np.pi/2)
            yaw_rate_ref.append(0)        
    elif referencetype == "yaw_rate": #Creates reference for a step in yaw rate
        for t in range(tot_time):
            vel_ref.append(-1)
            incline_ref.append(0)
            if t > 200 and t < 400:
                yaw_rate_ref.append(np.pi/12) #15 degrees per second
            else:
                yaw_rate_ref.append(0)
    elif referencetype == "yaw_rate_xvel": #Creates reference for moving in the x direction with a yaw rate
        for t in range(tot_time):
            vel_ref.append(0.5)
            incline_ref.append(0)
            if t > 200 and t < 220:
                yaw_rate_ref.append(np.pi/6) #30 degrees per second
            else:
                yaw_rate_ref.append(0)
    elif referencetype == "velocity_step": #Creates reference for moving in the x and z direction with a step in velocity
        for t in range(tot_time):
            if t < 200:
                vel_ref.append(0.25)
                incline_ref.append(np.pi/4)
                yaw_rate_ref.append(0)
            else:
                vel_ref.append(1)
                incline_ref.append(np.pi/4)
                yaw_rate_ref.append(0)
    elif referencetype == "incline_step": #Creates reference for moving in the x and z direction with a step in incline
        for t in range(tot_time):
            if t < 200:
                vel_ref.append(0.5)
                incline_ref.append(np.pi/8)
                yaw_rate_ref.append(0)
            else:
                vel_ref.append(0.5)
                incline_ref.append(np.pi/4)
                yaw_rate_ref.append(0)  
    elif referencetype == "velx_yaw_rate_velx": #Creates reference for moving in the x direction then stop to yaw then move in the x direction
        for t in range(tot_time):
            if t < 200:
                vel_ref.append(0.5)
                incline_ref.append(0)
                yaw_rate_ref.append(0)
            elif t > 200 and t < 240:
                vel_ref.append(0)
                incline_ref.append(0)
                yaw_rate_ref.append(np.pi/6)
            else:
                vel_ref.append(0.5)
                incline_ref.append(0)
                yaw_rate_ref.append(0)
    elif referencetype == "sin_xvel_sin_zvel":
        for t in range(tot_time):
            vel_ref.append(0.1*np.sin(0.1*t))
            incline_ref.append(0)
            yaw_rate_ref.append(0)
    elif referencetype == "complex":
        for t in range(tot_time):
            vel_ref.append(0.1*np.sin(0.1*t))
            incline_ref.append(0.1*np.sin(0.1*t))
            yaw_rate_ref.append(0.1*np.sin(0.1*t))
    
    elif referencetype == "manual_input":
        for t in range(tot_time):
            if t < 100:
                #Move forward along xaxis
                vel_ref.append(1)
                incline_ref.append(0.3)
                yaw_rate_ref.append(0)
            elif t > 100 and t < 200:
                #Move up along zaxis
                vel_ref.append(1)
                incline_ref.append(1)
                yaw_rate_ref.append(0)
            elif t > 200 and t < 300:
                #Move rotate right
                vel_ref.append(-1)
                incline_ref.append(0)
                yaw_rate_ref.append(-1)
            elif t > 300 and t < 400:
                #move forward along xaxis
                vel_ref.append(1)
                incline_ref.append(0.3)
                yaw_rate_ref.append(0)
            elif t > 400 and t < 500:
                #Move down along zaxis
                vel_ref.append(1)
                incline_ref.append(-1)
                yaw_rate_ref.append(0)
            elif t > 500 and t < 600:
                #Rotate left
                vel_ref.append(-1)
                incline_ref.append(0)
                yaw_rate_ref.append(1)
            elif t > 600 and t < 700:
                #Move forward along xaxis
                vel_ref.append(1)
                incline_ref.append(0.3)
                yaw_rate_ref.append(0)
            else:
                #Hvover
                vel_ref.append(-1)
                incline_ref.append(0)
                yaw_rate_ref.append(0)

           
    actual_vel_world = []
    actual_vel_body = []
    actual_yaw_rate = []
    actual_incl_angle_world = []
    actual_incl_angle_body = []
    aoa = []
    x_vel_cmd = []
    y_vel_cmd = []
    z_vel_cmd = []
    yaw_rate_cmd = []
    forces = []
    timesteps = []
    statelog = []
    att_des = []
    for t in range(tot_time): 
        action = vel_ref[t], incline_ref[t], yaw_rate_ref[t]
        # F = Test.PID_PD_Lin_velocity_controller(action)
        # F = Test.geometric_velocity_controller(action)
        # F = Test.leectrl(action)
        # F = Test.geom_ctrlv2(action)
        F = Test.active_geom_ctrlv2(action)
        Test.quadcopter.step(F)
        Test.total_t_steps += 1

        #save forces 
        timesteps.append(t)
        forces.append(F)

        #save the states and commands
        x_vel_cmd.append(Test.cmd[0])
        y_vel_cmd.append(Test.cmd[1])
        z_vel_cmd.append(Test.cmd[2])
        yaw_rate_cmd.append(Test.cmd[3])
        att_des.append(Test.att_des)
        actual_vel_world.append(Test.quadcopter.position_dot)
        actual_vel_body.append(Test.quadcopter.velocity)
        actual_yaw_rate.append(Test.quadcopter.angular_velocity[2])
        # actual_incl_angle_world.append(Test.quadcopter.upsilon)
        actual_incl_angle_body.append(geom.ssa(Test.quadcopter.upsilon-Test.quadcopter.pitch))
        aoa.append(Test.quadcopter.aoa)
        statelog.append(Test.quadcopter.state)
    
    #save forces to a npy file
    if save_forces:
        np.save(f"forces_{referencetype}.npy", np.array(forces))

    # integrate the commanded velocity and commanded yaw rate commands to get the reference path (is this ish correct)
    pos_ref = np.zeros((tot_time, 3))
    pos_ref[0][2] = 1
    for t in range(tot_time):
        yaw = yaw_rate_cmd[t]
        if t > 0:
            pos_ref[t] = pos_ref[t-1] + np.array([x_vel_cmd[t]*np.cos(yaw), y_vel_cmd[t]*np.sin(yaw), z_vel_cmd[t]])*0.01
        
    #Scale timestep to be in seconds
    timesteps = np.array(timesteps)*0.01

    #Subplot the velocity commanded and the actual velocity
    actual_vel_world = np.array(actual_vel_world)    
    actual_vel_body = np.array(actual_vel_body)
    velxlim = (min(0, min(x_vel_cmd),np.min(actual_vel_world[:,0]), np.min(actual_vel_body[:,0]))-0.1, max(max(x_vel_cmd), np.max(actual_vel_world[:,0]), np.max(actual_vel_body[:,0])) +0.1)
    velylim = (min(0, min(y_vel_cmd),np.min(actual_vel_world[:,1]),np.min(actual_vel_body[:,1]))-0.1, max(max(y_vel_cmd),np.max(actual_vel_world[:,1]),np.max(actual_vel_body[:,1]))+0.1)
    velzlim = (min(0, min(z_vel_cmd),np.min(actual_vel_world[:,2]), np.min(actual_vel_body[:,2]))-0.1, max(max(z_vel_cmd),np.max(actual_vel_world[:,2]), np.max(actual_vel_body[:,2]))+0.1)

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    #Commanded vs actual velocity and yaw rate
    plt.figure(1)
    plt.suptitle(referencetype)
    plt.subplot(2, 2, 1)
    plt.plot(timesteps,x_vel_cmd, label='x velocity command')
    plt.plot(timesteps,[v[0] for v in actual_vel_world], label='actual x velocity in world',linestyle='dashed')
    plt.plot(timesteps,[v[0] for v in actual_vel_body], label='actual x velocity in body')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Timesteps[s]')
    plt.ylim(velxlim[0], velxlim[1])
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(timesteps,y_vel_cmd, label='y velocity command')
    plt.plot(timesteps,[v[1] for v in actual_vel_world], label='actual y velocity in world',linestyle='dashed')
    plt.plot(timesteps,[v[1] for v in actual_vel_body], label='actual y velocity in body')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Timesteps[s]')
    plt.ylim(velylim[0], velylim[1])
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(timesteps,z_vel_cmd, label='z velocity command')
    plt.plot(timesteps,[v[2] for v in actual_vel_world], label='actual z velocity in world',linestyle='dashed')
    plt.plot(timesteps,[v[2] for v in actual_vel_body], label='actual z velocity in body')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Timesteps[s]')
    plt.ylim(velzlim[0], velzlim[1])
    plt.legend()
    plt.subplot(2, 2, 4)
    yaw_rate_cmd = np.array(yaw_rate_cmd)
    actual_yaw_rate = np.array(actual_yaw_rate)
    plt.plot(timesteps,yaw_rate_cmd*180/np.pi, label='yaw rate command')
    plt.plot(timesteps,actual_yaw_rate*180/np.pi, label='yaw rate actual')
    plt.ylim(-90, 90)
    plt.ylabel('Yaw rate (deg/s)')
    plt.xlabel('Timesteps[s]')
    plt.legend()

    #Plot the force of each motor in subplot 
    plt.figure(2)
    plt.suptitle(referencetype)
    plt.subplot(2, 2, 1)
    plt.plot(timesteps, [f[0] for f in forces], label='motor 1')
    plt.ylabel('Force (N)')
    plt.xlabel('Timesteps[s]')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, [f[1] for f in forces], label='motor 2')
    plt.ylabel('Force (N)')
    plt.xlabel('Timesteps[s]')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(timesteps, [f[2] for f in forces], label='motor 3')
    plt.ylabel('Force (N)')
    plt.xlabel('Timesteps[s]')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(timesteps, [f[3] for f in forces], label='motor 4')
    plt.ylabel('Force (N)')
    plt.xlabel('Timesteps[s]')
    plt.legend()

    #Plot the position of the quadcopter in 3D
    x = [s[0] for s in statelog]
    y = [s[1] for s in statelog]
    z = [s[2] for s in statelog]
    plt.figure(3)
    plt.suptitle(referencetype)
    ax = plt.axes(projection='3d')
    # ax.plot3D(pos_ref[:,0], pos_ref[:,1], pos_ref[:,2], 'tab:blue', label='reference path') #THIS PATH IS ONLY SOMEWHAT CORRECT AS IT INTEGRATES THE VELOCITY COMMANDS BUT DOES NOT ACCOUNT FOR DYNAMICS OR THE YAW
    ax.plot3D(x, y, z, 'gray', label='actual path')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #place a red dot at the start position and a blue dot at the end position add a legend
    ax.scatter3D(x[0], y[0], z[0], color='r', label='start')
    ax.scatter3D(x[-1], y[-1], z[-1], color='b',label='end')
    ax.legend()
    #scale the axes such that theyre in the same scale
    max_range = np.array([x, y, z]).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])

    # #Plot the actual incline angle and the reference incline angle if upsilon is the inclination angle
    # plt.figure(4)
    # # actual_incl_angle_world=np.array(actual_incl_angle_world)*180/np.pi
    # actual_incl_angle_body=np.array(actual_incl_angle_body)*180/np.pi
    # aoa=np.array(aoa)*180/np.pi
    # incline_ref=np.array(incline_ref)*180/np.pi
    # # plt.plot(timesteps, actual_incl_angle_world, label='actual incline angle world')
    # plt.suptitle(referencetype)
    # # plt.plot(timesteps, actual_incl_angle_body, label='actual incline angle body')
    # plt.plot(timesteps,aoa, label='angle of attack')
    # plt.plot(timesteps, incline_ref, label='incline ref')
    # plt.ylabel('Incline angle (degrees)')
    # plt.xlabel('Timesteps[s]')
    # plt.legend()

    #Plot the attitude of the quadcopter in subplot
    roll = np.array([s[3] for s in statelog])*180/np.pi
    pitch = np.array([s[4] for s in statelog])*180/np.pi
    yaw = np.array([s[5] for s in statelog])*180/np.pi
    des_roll = np.array([a[0] for a in att_des])*180/np.pi
    des_pitch = np.array([a[1] for a in att_des])*180/np.pi
    des_yaw = np.array([a[2] for a in att_des])*180/np.pi
    plt.figure(5)
    plt.suptitle(referencetype)
    plt.subplot(2, 2, 1)
    plt.plot(timesteps, roll, label='roll actual')
    plt.plot(timesteps, des_roll, label='roll desired')
    plt.ylabel('Attitude (degrees)')
    plt.xlabel('Timesteps[s]')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, pitch, label='pitch actual')
    plt.plot(timesteps, des_pitch, label='pitch desired')
    plt.ylabel('Attitude (degrees)')
    plt.xlabel('Timesteps[s]')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(timesteps, yaw, label='yaw actual')
    plt.plot(timesteps, des_yaw, label='yaw desired')
    plt.ylabel('Attitude (degrees)')
    plt.xlabel('Timesteps[s]')
    plt.legend()

    plt.show()

#OLD code that I want to keep around for a little while
# Lee velocity controller inspired by the dorne lab ntnu
# ##DRONE LAB NTNU LINEAR VELOCITY CONTROLLER
        
# # Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# # All rights reserved.

# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# copter = Quad(0.01,np.zeros(6))
# state = copter.state

# class LeeVelocityController:
#     def __init__(self, k_vel, k_rot, k_angvel):
#         self.k_vel_ = k_vel
#         self.k_rot_ = k_rot
#         self.k_angvel = k_angvel

#     def __call__(self, robot_state, command_actions):
#         """
#         Lee velocity controller
#         :param robot_state: array of shape (12) with state of the robot
#         :param command_actions: array of shape (4) with desired velocity setpoint in vehicle frame and yaw_rate command in vehicle frame
#         :return: m*g normalized thrust and inertial normalized torques
#         """
#         # Perform calculation for transformation matrices
#         rotation_matrix = geom.Rzyx(robot_state[3], robot_state[4], robot_state[5])
#         rotation_matrix_transpose = rotation_matrix.T
#         euler_angles = robot_state[3:6]

#         # Convert to vehicle frame
#         vehicle_frame_euler = np.zeros_like(euler_angles)
#         vehicle_frame_euler[2] = euler_angles[2]
#         vehicle_vels = robot_state[6:9]
        
#         vehicle_frame_transforms = geom.Rzyx(vehicle_frame_euler[0], vehicle_frame_euler[1], vehicle_frame_euler[2])
#         vehicle_frame_transforms_transpose = vehicle_frame_transforms.T

#         vehicle_frame_velocity = vehicle_frame_transforms_transpose @ vehicle_vels

#         desired_vehicle_velocity = command_actions[:3]

#         # Compute desired accelerations
#         vel_error = desired_vehicle_velocity - vehicle_frame_velocity
#         accel_command = self.k_vel_ * vel_error
#         accel_command[2] += 1 

#         #From paper Thrust_cmd = -(-kx*ex - kv*ev - m*g*e_3 + m*x_dot_dot_desired) dot R.T @ e_3
#         # e_3 is [0,0,1] R@e_3 is the third column of R
#         # Our z axis points upwards while the paper's z axis points downwards, explains the flip of signs.
#         # We dont have a position controller, so we dont have the -kx*ex term.
#         # The m*x_dot_dot_desired term is the acceleration command but we dont have it as we dont have xd(?)
#         forces_command = accel_command
#         thrust_command = np.dot(forces_command, rotation_matrix[2])

#         #OLD
#         c_phi_s_theta = forces_command[0]
#         s_phi = -forces_command[1]
#         c_phi_c_theta = forces_command[2]

#         # Calculate euler setpoints
#         pitch_setpoint = np.arctan2(c_phi_s_theta, c_phi_c_theta)
#         roll_setpoint = np.arctan2(s_phi, np.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
#         yaw_setpoint = euler_angles[2]
#         #OLD

#         #NEW maybe working
#         # roll_setpoint = np.arctan2(forces_command[1], forces_command[2])
#         # pitch_setpoint = np.arctan2(-forces_command[0], np.sqrt(forces_command[1]**2 + forces_command[2]**2))
#         # yaw_setpoint = euler_angles[2]

#         euler_setpoints = np.zeros_like(euler_angles)
#         euler_setpoints[0] = roll_setpoint
#         euler_setpoints[1] = pitch_setpoint
#         euler_setpoints[2] = yaw_setpoint

#         # perform computation on calculated values
#         rotation_matrix_desired = geom.Rzyx(euler_setpoints[0], euler_setpoints[1], euler_setpoints[2])
#         rotation_matrix_desired_transpose = rotation_matrix_desired.T

#         rot_err_mat = np.matmul(rotation_matrix_desired_transpose, rotation_matrix) - np.matmul(rotation_matrix_transpose, rotation_matrix_desired)
#         rot_err = 0.5 * geom.vee_map(rot_err_mat)

#         rotmat_euler_to_body_rates = np.zeros_like(rotation_matrix)

#         s_pitch = np.sin(euler_angles[1])
#         c_pitch = np.cos(euler_angles[1])

#         s_roll = np.sin(euler_angles[0])
#         c_roll = np.cos(euler_angles[0])

#         rotmat_euler_to_body_rates[0, 0] = 1.0
#         rotmat_euler_to_body_rates[1, 1] = c_roll
#         rotmat_euler_to_body_rates[0, 2] = -s_pitch
#         rotmat_euler_to_body_rates[2, 1] = -s_roll
#         rotmat_euler_to_body_rates[1, 2] = s_roll * c_pitch
#         rotmat_euler_to_body_rates[2, 2] = c_roll * c_pitch

#         euler_angle_rates = np.zeros_like(euler_angles)
#         euler_angle_rates[2] = command_actions[3]

#         omega_desired_body = np.matmul(rotmat_euler_to_body_rates, euler_angle_rates)

#         # omega_des_body = [0, 0, yaw_rate] ## approximated body_rate as yaw_rate
#         # omega_body = R_t @ omega_world
#         # angvel_err = omega_body - R_t @ R_des @ omega_des_body
#         # Refer to Lee et. al. (2010) for details (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717652)
        
#         desired_angvel_err = np.matmul(rotation_matrix_transpose, np.matmul(rotation_matrix_desired, omega_desired_body[:, np.newaxis]))

#         actual_angvel_err = np.matmul(rotation_matrix_transpose, robot_state[9:12, np.newaxis])

#         angvel_err = actual_angvel_err - desired_angvel_err

#         torque = - self.k_rot_ * rot_err - self.k_angvel * angvel_err + np.reshape(np.cross(robot_state[9:12],ss.Ig@robot_state[9:12]),(3,1))
        
#         return thrust_command, torque            
    

# Example call to the LeeVelocityController done inside the veloitcy_controller function
        ## LEE VELOCITY CONTROLLER
        # state = self.quadcopter.state
        
        # k_vel = 0.5
        # k_rot = 0.5
        # k_angvel = 0.5

        # LeeVelCtrl = LeeVelocityController(k_vel, k_rot, k_angvel)
        
        # output_thrust_mass_normalized, output_torques_inertia_normalized = LeeVelCtrl(state,self.quadcopter.chi, self.cmd)

        # f = output_thrust_mass_normalized * (ss.W)
        # T = output_torques_inertia_normalized #* ss.Ig.trace()
        # #* ss.W * ss.l

        # u_des = np.array([f, T[0][0], T[1][0], T[2][0]])

        # F = np.linalg.inv(ss.B()[2:]).dot(u_des)
        # F = np.clip(F, ss.thrust_min, ss.thrust_max)

        # return F
        ###

        ## My own PID controller for velocity control and PD for attitude control
        # Calculate the thrust inputs using the commanded velocity + PID control
        # and yaw rate + PD control for attitude   