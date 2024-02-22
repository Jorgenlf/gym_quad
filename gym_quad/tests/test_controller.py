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
from gym_quad.objects.quad import Quad


    # "s_max": 0.5, # Maximum speed of the quadcopter
    # "i_max": 0.5, # Maximum inclination angle of commanded velocity wrt x-axis
    # "omega_max": 0.5, # Maximum commanded yaw rate

class TestController(unittest.TestCase):

    def setUp(self):
        self.quadcopter = Quad(0.01,np.zeros(6))
        self.quadcopter.state[2] = 1 # Set the initial height to 1m
        self.s_max = 2.5
        self.i_max = np.pi/2 #90 degrees
        self.r_max = np.pi/3 #60 degrees per second
        self.total_t_steps = 0
        self.step_size = 0.01

    def velocity_controller(self, action):
        """
        The velocity controller for the quadcopter.
        Based on Kulkarni and Kostas paper. And Ørjans thesis controller.

        Parameters:
        ----------
        action : np.array (3,)
        The action input from the RL agent. 
        a1: reference speed in world frame, 
        a2: inclination of velocity vector wrt x-axis of body
        a3: yaw rate of body frame relative to world frame
        
        Returns:
        -------
        F : np.array (4,)
        The thrust inputs required to follow path and avoid obstacles according to the action of the DRL agent.
        #TODO add an additional loop that converts desired thrust per rotor to angular velocity of the rotorblades.
        """
        #Using hyperparam of s_max, i_max, r_max to clip the action to get the commanded velocity and yaw rate
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1])#*self.i_max)
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1])#*self.i_max)
        cmd_r = self.r_max * action[2]

        self.cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r])

        #For the attitude the natural frequency and damping ratio are set to 9 and 1 respectively    
        omega_n = 9 # Natural frequency
        xi = 1      # Damping ratio, 1 -> critically damped

        K_p_att = np.diag([ss.I_x * omega_n**2, ss.I_y * omega_n**2, ss.I_z * omega_n**2])
        K_d_att = np.diag([2 * ss.I_x * xi * omega_n, 2 * ss.I_y * xi * omega_n, 2 * ss.I_z * xi * omega_n])

        #Prop, damp and integral gains for the linear velocity
        #OLD gains get fair tracking of x and z but not y
        # K_p = np.diag([6, 0.8, 4])
        # K_d = np.diag([0.5, 0.2, 0.5])
        # K_i = np.diag([4, 0.3, 1.5])

        # #NEW gains to get better tracking of x and z and y #TODO
        K_p = np.diag([1.2, 1.2, 1.2])
        K_d = np.diag([0.1, 0.1, 0.1])
        K_i = np.diag([2, 2, 2])

        #KP part
        vel_world = geom.Rzyx(*self.quadcopter.attitude) @ self.quadcopter.velocity #Also given by self.quadcopter.position_dot
        #The world frame velocity of the quadcopter may not be available when using an actual quadcopter with IMU meas?
        #- no worries can get the world velocity by rotating the body velocity available by integrating linear accelerations from IMU
        # with the attitude from integrating up the angular rates from the IMU
        # Or try to do state estimation with a kalman filter or something as integrating up the IMU data is not very accurate #TODO
        v_error = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - vel_world

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
        yaw_des = self.quadcopter.attitude[2]
        term1 = (a_des[0]/(a_des[2] + ss.g + (ss.d_w*self.quadcopter.heave - ss.d_u*self.quadcopter.surge)/ss.m))
        term2 = (a_des[1]/(a_des[2] + ss.g + (ss.d_w*self.quadcopter.heave - ss.d_v*self.quadcopter.sway)/ss.m))
        phi_des = geom.ssa(term1*np.sin(yaw_des) + term2*np.cos(yaw_des))
        theta_des = geom.ssa(term1*np.cos(yaw_des) + term2*np.sin(yaw_des))

        e_att = geom.ssa(np.array([phi_des, theta_des, yaw_des]) - self.quadcopter.attitude)
        
        e_angvel = np.array([0.0, 0.0, cmd_r]) - self.quadcopter.angular_velocity

        torque = K_p_att @ e_att + K_d_att @ e_angvel #+ np.cross(self.quadcopter.angular_velocity,ss.Ig@self.quadcopter.angular_velocity)
        u = np.zeros(4)
        u[0] = ss.m * (a_des[2] + ss.g) + ss.d_w*self.quadcopter.heave #thurst force in body z direction i.e. total thrust
        u[1:] = torque

        F = np.linalg.inv(ss.B()[2:]).dot(u)
        F = np.clip(F, ss.thrust_min, ss.thrust_max)    

        return F
    
    def test_velocity_controller(self):

        action = np.array([0.5, 0.5, 0.5])

        F = self.velocity_controller(action)
        
        self.assertEqual(F.shape, (4,))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    Test = TestController()
    Test.setUp()

    #Velocity and yaw rate reference
    vel_ref = []
    incline_ref = []
    yaw_rate_ref = []
    tot_time = 900

    # Creates reference for hovering at 1m height check initial state of the quadcopter in setUp
    # for t in range(tot_time):
    #     vel_ref.append(-1)
    #     yaw_rate_ref.append(0)
    #     incline_ref.append(0)
    
    # Creates reference for moving in the x direction
    # for t in range(tot_time):
        # vel_ref.append(1)
        # yaw_rate_ref.append(0)
        # incline_ref.append(0)

    #Creates reference for moving in the z direction
    # for t in range(tot_time):
    #     vel_ref.append(1)
    #     incline_ref.append(np.pi/2)
    #     yaw_rate_ref.append(0)        

    #Creates reference for moving in the x direction with a yaw rate
    for t in range(tot_time):
        vel_ref.append(0.5)
        incline_ref.append(0)
        if t > 200 and t < 400:
            yaw_rate_ref.append(np.pi/12) #15 degrees per second
        else:
            yaw_rate_ref.append(0)
        
    #Creates reference for moving in the x and z direction with a step in velocity
    # for t in range(tot_time):
    #     if t < 200:
    #         vel_ref.append(0.25)
    #         incline_ref.append(np.pi/4)
    #         yaw_rate_ref.append(0)
    #     else:
    #         vel_ref.append(1)
    #         incline_ref.append(np.pi/4)
    #         yaw_rate_ref.append(0)

    #Creates reference for moving in the x and z direction with a step in incline
    # for t in range(tot_time):
    #     if t < 200:
    #         vel_ref.append(0.5)
    #         incline_ref.append(np.pi/8)
    #         yaw_rate_ref.append(0)
    #     else:
    #         vel_ref.append(0.5)
    #         incline_ref.append(np.pi/4)
    #         yaw_rate_ref.append(0)

    actual_vel_world = []
    actual_vel_body = []
    actual_yaw_rate = []
    actual_incl_angle = []
    x_vel_cmd = []
    y_vel_cmd = []
    z_vel_cmd = []
    yaw_rate_cmd = []
    forces = []
    timesteps = []
    statelog = []
    for t in range(tot_time):
        action = vel_ref[t], incline_ref[t], yaw_rate_ref[t]
        F = Test.velocity_controller(action)
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
        actual_vel_world.append(Test.quadcopter.position_dot)
        actual_vel_body.append(Test.quadcopter.velocity)
        actual_yaw_rate.append(Test.quadcopter.angular_velocity[2])
        actual_incl_angle.append(Test.quadcopter.upsilon)
        statelog.append(Test.quadcopter.state)

    #Subplot the velocity commanded and the actual velocity
    actual_vel_world = np.array(actual_vel_world)    
    actual_vel_body = np.array(actual_vel_body)
    velxlim = (min(0, min(x_vel_cmd),np.min(actual_vel_world[:,0]), np.min(actual_vel_body[:,0]))-0.1, max(max(x_vel_cmd), np.max(actual_vel_world[:,0]), np.max(actual_vel_body[:,0])) +0.1)
    velylim = (min(0, min(y_vel_cmd),np.min(actual_vel_world[:,1]),np.min(actual_vel_body[:,1]))-0.1, max(max(y_vel_cmd),np.max(actual_vel_world[:,1]),np.max(actual_vel_body[:,1]))+0.1)
    velzlim = (min(0, min(z_vel_cmd),np.min(actual_vel_world[:,2]), np.min(actual_vel_body[:,2]))-0.1, max(max(z_vel_cmd),np.max(actual_vel_world[:,2]), np.max(actual_vel_body[:,2]))+0.1)
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(x_vel_cmd, label='x velocity command')
    plt.plot([v[0] for v in actual_vel_world], label='actual x velocity in world')
    plt.plot([v[0] for v in actual_vel_body], label='actual x velocity in body', linestyle='dashed')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Timesteps')
    plt.ylim(velxlim[0], velxlim[1])
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(y_vel_cmd, label='y velocity command')
    plt.plot([v[1] for v in actual_vel_world], label='actual y velocity in world')
    plt.plot([v[1] for v in actual_vel_body], label='actual y velocity in body', linestyle='dashed')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Timesteps')
    plt.ylim(velylim[0], velylim[1])
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(z_vel_cmd, label='z velocity command')
    plt.plot([v[2] for v in actual_vel_world], label='actual z velocity in world ')
    plt.plot([v[2] for v in actual_vel_body], label='actual z velocity in body', linestyle='dashed')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Timesteps')
    plt.ylim(velzlim[0], velzlim[1])
    plt.legend()
    plt.subplot(2, 2, 4)
    yaw_rate_cmd = np.array(yaw_rate_cmd)
    actual_yaw_rate = np.array(actual_yaw_rate)
    plt.plot(yaw_rate_cmd*180/np.pi, label='yaw rate command')
    plt.plot(actual_yaw_rate*180/np.pi, label='yaw rate actual')
    #scale so y axis displays 180 degrees
    plt.ylim(-90, 90)
    plt.ylabel('Yaw rate (deg/s)')
    plt.xlabel('Timesteps')
    plt.legend()

    #Plot the force of each motor in subplot 
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.plot(timesteps, [f[0] for f in forces], label='motor 1')
    plt.ylabel('Force (N)')
    plt.xlabel('Timesteps')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, [f[1] for f in forces], label='motor 2')
    plt.ylabel('Force (N)')
    plt.xlabel('Timesteps')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(timesteps, [f[2] for f in forces], label='motor 3')
    plt.ylabel('Force (N)')
    plt.xlabel('Timesteps')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(timesteps, [f[3] for f in forces], label='motor 4')
    plt.ylabel('Force (N)')
    plt.xlabel('Timesteps')
    plt.legend()

    #Plot the position of the quadcopter in 3D
    x = [s[0] for s in statelog]
    y = [s[1] for s in statelog]
    z = [s[2] for s in statelog]
    plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'gray', label='path')
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

    #Plot the actual incline angle and the reference incline angle if upsilon is the inclination angle
    plt.figure(4)
    actual_incl_angle=np.array(actual_incl_angle)*180/np.pi
    incline_ref=np.array(incline_ref)*180/np.pi
    plt.plot(timesteps, actual_incl_angle, label='actual incline angle')
    plt.plot(timesteps, incline_ref, label='incline ref')
    plt.ylabel('Incline angle (degrees)')
    plt.xlabel('Timesteps')
    plt.legend()

    #Plot the attitude of the quadcopter in subplot
    roll = np.array([s[3] for s in statelog])*180/np.pi
    pitch = np.array([s[4] for s in statelog])*180/np.pi
    yaw = np.array([s[5] for s in statelog])*180/np.pi
    plt.figure(5)
    plt.subplot(2, 2, 1)
    plt.plot(timesteps, roll, label='roll')
    plt.ylabel('Attitude (degrees)')
    plt.xlabel('Timesteps')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, pitch, label='pitch')
    plt.ylabel('Attitude (degrees)')
    plt.xlabel('Timesteps')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(timesteps, yaw, label='yaw')
    plt.ylabel('Attitude (degrees)')
    plt.xlabel('Timesteps')
    plt.legend()

    plt.show()
    # Run the tests
    unittest.main()


#OLD code that I want to keep around for a little while
# Lee velocity controller inspired by the dorne lab ntnu
# ##DRONE LAB NTNU LINEAR VELOCITY CONTROLLER
        
# # Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# # All rights reserved.

# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.
# import torch
# from torch import Tensor
# # def compute_vee_map(skew_matrix):
# #     # type: (Tensor) -> Tensor
# #     # return vee map of skew matrix
# #     vee_map = torch.stack(
# #         [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
# #     return vee_map

# def compute_vee_map(skew_matrix):
#     # skew_matrix is assumed to be a numpy array
#     # Extract the elements of the skew matrix
#     m12 = skew_matrix[0, 2]
#     m20 = skew_matrix[1, 0]
#     m01 = skew_matrix[2, 1]
#     # Compute the vee map
#     vee_map = np.array([[-m12, m20, -m01]])
#     vee_map = np.reshape(vee_map, (3, 1))
    
#     return vee_map    

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
#         accel_command[2] += 1 # Copilot thinks: compensate for gravity as normalized g = 1. I think: IS THIS CORRECT?

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

#         #NEW NEW using quad and statespace:
#         # b_x = accel_command[0] / (accel_command[2] + (ss.d_w*robot_state[8] - ss.d_u*robot_state[6]))
#         # b_y = accel_command[1] / (accel_command[2] + (ss.d_w*robot_state[8] - ss.d_v*robot_state[7]))
#         # roll_setpoint   = geom.ssa(b_x * np.sin(chi_p) - b_y * np.cos(chi_p))
#         # pitch_setpoint = geom.ssa(b_x * np.cos(chi_p) + b_y * np.sin(chi_p))
#         ##

#         euler_setpoints = np.zeros_like(euler_angles)
#         euler_setpoints[0] = roll_setpoint
#         euler_setpoints[1] = pitch_setpoint
#         euler_setpoints[2] = yaw_setpoint

#         # perform computation on calculated values
#         rotation_matrix_desired = geom.Rzyx(euler_setpoints[0], euler_setpoints[1], euler_setpoints[2])
#         rotation_matrix_desired_transpose = rotation_matrix_desired.T

#         rot_err_mat = np.matmul(rotation_matrix_desired_transpose, rotation_matrix) - np.matmul(rotation_matrix_transpose, rotation_matrix_desired)
#         rot_err = 0.5 * compute_vee_map(rot_err_mat)

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