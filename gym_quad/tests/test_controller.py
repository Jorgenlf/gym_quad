
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



##DRONE LAB NTNU LINEAR VELOCITY CONTROLLER
        
# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# import torch
# from torch import Tensor
# @torch.jit.script
# def compute_vee_map(skew_matrix):
#     # type: (Tensor) -> Tensor
#     # return vee map of skew matrix
#     vee_map = torch.stack(
#         [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
#     return vee_map

def compute_vee_map(skew_matrix):
    # skew_matrix is assumed to be a numpy array
    # Extract the elements of the skew matrix
    m12 = skew_matrix[0, 2]
    m20 = skew_matrix[1, 0]
    m01 = skew_matrix[2, 1]
    # Compute the vee map
    vee_map = np.array([[-m12, m20, -m01]])
    vee_map = np.reshape(vee_map, (3, 1))
    
    return vee_map    

copter = Quad(0.01,np.zeros(6))
state = copter.state

class LeeVelocityController:
    def __init__(self, k_vel, k_rot, k_angvel):
        self.k_vel_ = k_vel
        self.k_rot_ = k_rot
        self.k_angvel = k_angvel

    def __call__(self, robot_state, command_actions):
        """
        Lee velocity controller
        :param robot_state: tensor of shape (12) with state of the robot
        :param command_actions: tensor of shape (4) with desired velocity setpoint in vehicle frame and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        # # perform calculation for transformation matrices

        # OLD rotation_matrices = p3d_transforms.quaternion_to_matrix(robot_state[:, [6, 3, 4, 5]])
        rotation_matrix = geom.Rzyx(robot_state[3], robot_state[4], robot_state[5])
        
        # OLD rotation_matrix_transpose = torch.transpose(rotation_matrices, 1, 2)
        rotation_matrix_transpose = rotation_matrix.T

        # OLD euler_angles = p3d_transforms.matrix_to_euler_angles(rotation_matrices, "ZYX")[:, [2, 1, 0]]
        euler_angles = robot_state[3:6]

        # Convert to vehicle frame
        vehicle_frame_euler = np.zeros_like(euler_angles)
        vehicle_frame_euler[2] = euler_angles[2]
        vehicle_vels = robot_state[6:9]
        
        #OLD vehicle_frame_transforms  = p3dtransforms.euler_angles_to_matrix(vehicle_frame_euler[:, [2, 1, 0]], "ZYX")
        vehicle_frame_transforms = geom.Rzyx(vehicle_frame_euler[0], vehicle_frame_euler[1], vehicle_frame_euler[2])
        #OLD vehicle_frame_transforms_transpose = torch.transpose(vehicle_frame_transforms, 1, 2)
        vehicle_frame_transforms_transpose = vehicle_frame_transforms.T

        vehicle_frame_velocity = vehicle_frame_transforms_transpose @ vehicle_vels

        desired_vehicle_velocity = command_actions[:3]

        # Compute desired accelerations
        vel_error = desired_vehicle_velocity - vehicle_frame_velocity
        accel_command = self.k_vel_ * vel_error
        accel_command[2] += 1

        forces_command = accel_command
        # OLD thrust_command = torch.sum(forces_command * rotation_matrices[:, :, 2], dim=1)
        thrust_command = np.dot(forces_command, rotation_matrix[2])

        c_phi_s_theta = forces_command[0]
        s_phi = -forces_command[1]
        c_phi_c_theta = forces_command[2]

        # Calculate euler setpoints
        pitch_setpoint = np.arctan2(c_phi_s_theta, c_phi_c_theta)
        roll_setpoint = np.arctan2(s_phi, np.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
        yaw_setpoint = euler_angles[2]

        euler_setpoints = np.zeros_like(euler_angles)
        euler_setpoints[0] = roll_setpoint
        euler_setpoints[1] = pitch_setpoint
        euler_setpoints[2] = yaw_setpoint

        # perform computation on calculated values
        # rotation_matrix_desired = p3d_transforms.euler_angles_to_matrix(euler_setpoints[:, [2, 1, 0]], "ZYX")
        rotation_matrix_desired = geom.Rzyx(euler_setpoints[0], euler_setpoints[1], euler_setpoints[2])
        rotation_matrix_desired_transpose = rotation_matrix_desired.T
        
        # OLD rot_err_mat = torch.bmm(rotation_matrix_desired_transpose, rotation_matrices) - \
        #     torch.bmm(rotation_matrix_transpose, rotation_matrix_desired)

        rot_err_mat = np.matmul(rotation_matrix_desired_transpose, rotation_matrix) - \
              np.matmul(rotation_matrix_transpose, rotation_matrix_desired)
        rot_err = 0.5 * compute_vee_map(rot_err_mat)

        rotmat_euler_to_body_rates = np.zeros_like(rotation_matrix)

        s_pitch = np.sin(euler_angles[1])
        c_pitch = np.cos(euler_angles[1])

        s_roll = np.sin(euler_angles[0])
        c_roll = np.cos(euler_angles[0])

        rotmat_euler_to_body_rates[0, 0] = 1.0
        rotmat_euler_to_body_rates[1, 1] = c_roll
        rotmat_euler_to_body_rates[0, 2] = -s_pitch
        rotmat_euler_to_body_rates[2, 1] = -s_roll
        rotmat_euler_to_body_rates[1, 2] = s_roll * c_pitch
        rotmat_euler_to_body_rates[2, 2] = c_roll * c_pitch

        euler_angle_rates = np.zeros_like(euler_angles)
        euler_angle_rates[2] = command_actions[3]

        omega_desired_body = np.matmul(rotmat_euler_to_body_rates, euler_angle_rates)

        # omega_des_body = [0, 0, yaw_rate] ## approximated body_rate as yaw_rate
        # omega_body = R_t @ omega_world
        # angvel_err = omega_body - R_t @ R_des @ omega_des_body
        # Refer to Lee et. al. (2010) for details (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717652)

        # OLD desired_angvel_err = torch.bmm(rotation_matrix_transpose, torch.bmm(
        #     rotation_matrix_desired, omega_desired_body.unsqueeze(2))).squeeze(2)
        
        desired_angvel_err = np.matmul(rotation_matrix_transpose, np.matmul(rotation_matrix_desired, omega_desired_body[:, np.newaxis]))


        #OLD actual_angvel_err = torch.bmm(rotation_matrix_transpose, robot_state[:, 10:13].unsqueeze(2)).squeeze(2)
        actual_angvel_err = np.matmul(rotation_matrix_transpose, robot_state[9:12, np.newaxis])

        angvel_err = actual_angvel_err - desired_angvel_err

        torque = - self.k_rot_ * rot_err - self.k_angvel * angvel_err #+ np.cross(robot_state[9:12],robot_state[9:12]) #TODO check if this makes sense turns it to 3x3 matrix and cross by itself is wrong?
        
        return thrust_command, torque



class TestController(unittest.TestCase):

    def setUp(self):
        self.quadcopter = Quad(0.01,np.zeros(6))
        self.s_max = 0.5
        self.i_max = 0.5
        self.r_max = 0.5
        self.total_t_steps = 0
        self.step_size = 0.01

    def velocity_controller(self, action):
        """
        PD Controller mapping linear velocity and yaw rate to forces. 
        Based on Kulkarni and Kostas paper.
        
        Parameters:
        ----------
        action : np.array
        The action input from the RL agent.
        Speed, inclination of velocity vector wrt x-axis and yaw rate
        
        Returns:
        -------
        F : np.array
        The thrust inputs requiered to follow path and avoid obstacles according to the action of the DRL agent.
        """
        #Using hyperparam of s_max, i_max, r_max to clip the action to get the commanded velocity and yaw rate
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(self.i_max * action[1])
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(self.i_max * action[1])
        cmd_r = self.r_max * action[2]

        cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r])
        state = self.quadcopter.state
        
        k_vel = 0.5
        k_rot = 0.5
        k_angvel = 0.5

        LeeVelCtrl = LeeVelocityController(k_vel, k_rot, k_angvel)
        f, T = LeeVelCtrl(state, cmd)

        u_des = np.array([f, T[0][0], T[1][0], T[2][0]])

        F = np.linalg.inv(ss.B()[2:]).dot(u_des)
        F = np.clip(F, ss.thrust_min, ss.thrust_max)

        return F

        # #Calculate the thrust inputs using the commanded velocity and yaw rate and a PD controller
        # v_error = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - self.quadcopter.velocity
        # r_error = cmd_r - self.quadcopter.angular_velocity[2]

        # #For the yaw rate, the natural frequency and damping ratio are set to 9 and 1 respectively    
        # omega_n = 9 # Natural frequency
        # xi = 1      # Damping ratio, 1 -> critically damped

        # #Prop and damp gains for the yaw rate
        # k_p_r = ss.I_z * omega_n**2
        # k_d_r =  2 * ss.I_z * xi * omega_n

        # #Prop and damp gains for the linear velocity
        # K_p = np.diag([0.5, 0.5, 0.5])
        # K_d = np.diag([0.5, 0.5, 0.5])

        # #Calculate the derivative of the error for the yaw rate
        # if self.total_t_steps > 0:
        #     r_error_dot = (r_error - self.prev_r_error) / self.step_size
        # else:
        #     r_error_dot = 0
        # self.prev_r_error = r_error

        # u = np.zeros(4)
        # u[0] = r_error*k_p_r + r_error_dot*k_d_r

        # desired_acc = np.array([0, 0, 0]) # Want to move at a constant velocity or hover.
        # lin_accleration = self.quadcopter.state_dot(self.quadcopter.state)[6:9]
        # u[1:] = np.dot(K_p, v_error) + np.dot(K_d,  lin_accleration-desired_acc)


        #TODO get this working before copying into LV_VAE  
        # B = ????? #TODO
        # u = np.dot(B, u) #TODO
        # F = u

        return F
    
    def test_velocity_controller(self):

        action = np.array([0.5, 0.5, 0.5])

        F = self.velocity_controller(action)
        
        self.assertEqual(F.shape, (4,))



if __name__ == '__main__':

    Test = TestController()
    Test.setUp()
    #Create a set of actions between -1 and 1 to test the velocity controller
    for i in range(100):
        action = np.random.uniform(-1, 1, 3)
        F = Test.velocity_controller(action)
        Test.quadcopter.step(F)
        Test.total_t_steps += 1
        print("FORCE: ",F)
        print("ACTION: ",action)
        print("")

    #plot linear and angular velocities and control inputs


    # Run the tests
    unittest.main()