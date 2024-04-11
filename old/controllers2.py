import numpy as np
from numpy import pi
from numpy import sin, cos, tan, sqrt
from numpy.linalg import norm
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


# ##DRONE LAB NTNU LINEAR VELOCITY CONTROLLER
        
# # Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# # All rights reserved.

# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

import torch
# import pytorch3d.transforms as p3d_transforms


# from torch import Tensor
# @torch.jit.script
# def compute_vee_map(skew_matrix):
#     # type: (Tensor) -> Tensor

#     # return vee map of skew matrix
#     vee_map = torch.stack(
#         [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
#     return vee_map

# class LeeVelocityController:
#     def __init__(self, K_vel_tensor, K_rot_tensor, K_angvel_tensor):
#         self.K_vel_tensor = K_vel_tensor
#         self.K_rot_tensor = K_rot_tensor
#         self.K_angvel_tensor = K_angvel_tensor

#     def __call__(self, robot_state, command_actions):
#         """
#         Lee velocity controller
#         :param robot_state: tensor of shape (num_envs, 13) with state of the robot
#         :param command_actions: tensor of shape (num_envs, 4) with desired velocity setpoint in vehicle frame and yaw_rate command in vehicle frame
#         :return: m*g normalized thrust and interial normalized torques
#         """
#         # perform calculation for transformation matrices
#         rotation_matrices = p3d_transforms.quaternion_to_matrix(
#             robot_state[:, [6, 3, 4, 5]])
#         rotation_matrix_transpose = torch.transpose(rotation_matrices, 1, 2)
#         euler_angles = p3d_transforms.matrix_to_euler_angles(
#             rotation_matrices, "ZYX")[:, [2, 1, 0]]


#         # Convert to vehicle frame
#         vehicle_frame_euler = torch.zeros_like(euler_angles)
#         vehicle_frame_euler[:, 2] = euler_angles[:, 2]
#         vehicle_vels = robot_state[:, 7:10].unsqueeze(2)
        
#         vehicle_frame_transforms = p3d_transforms.euler_angles_to_matrix(
#             vehicle_frame_euler[:, [2, 1, 0]], "ZYX")
#         vehicle_frame_transforms_transpose = torch.transpose(vehicle_frame_transforms, 1, 2)

#         vehicle_frame_velocity = (
#             vehicle_frame_transforms_transpose @ vehicle_vels).squeeze(2)

#         desired_vehicle_velocity = command_actions[:, :3]

#         # Compute desired accelerations
#         vel_error = desired_vehicle_velocity - vehicle_frame_velocity
#         accel_command = self.K_vel_tensor * vel_error
#         accel_command[:, 2] += 1

#         forces_command = accel_command
#         thrust_command = torch.sum(forces_command * rotation_matrices[:, :, 2], dim=1)

#         c_phi_s_theta = forces_command[:, 0]
#         s_phi = -forces_command[:, 1]
#         c_phi_c_theta = forces_command[:, 2]

#         # Calculate euler setpoints
#         pitch_setpoint = torch.atan2(c_phi_s_theta, c_phi_c_theta)
#         roll_setpoint = torch.atan2(s_phi, torch.sqrt(
#             c_phi_c_theta**2 + c_phi_s_theta**2))
#         yaw_setpoint = euler_angles[:, 2]


#         euler_setpoints = torch.zeros_like(euler_angles)
#         euler_setpoints[:, 0] = roll_setpoint
#         euler_setpoints[:, 1] = pitch_setpoint
#         euler_setpoints[:, 2] = yaw_setpoint

#         # perform computation on calculated values
#         rotation_matrix_desired = p3d_transforms.euler_angles_to_matrix(
#             euler_setpoints[:, [2, 1, 0]], "ZYX")
#         rotation_matrix_desired_transpose = torch.transpose(
#             rotation_matrix_desired, 1, 2)
        
#         rot_err_mat = torch.bmm(rotation_matrix_desired_transpose, rotation_matrices) - \
#             torch.bmm(rotation_matrix_transpose, rotation_matrix_desired)
#         rot_err = 0.5 * compute_vee_map(rot_err_mat)

#         rotmat_euler_to_body_rates = torch.zeros_like(rotation_matrices)

#         s_pitch = torch.sin(euler_angles[:, 1])
#         c_pitch = torch.cos(euler_angles[:, 1])

#         s_roll = torch.sin(euler_angles[:, 0])
#         c_roll = torch.cos(euler_angles[:, 0])

#         rotmat_euler_to_body_rates[:, 0, 0] = 1.0
#         rotmat_euler_to_body_rates[:, 1, 1] = c_roll
#         rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
#         rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
#         rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
#         rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch

#         euler_angle_rates = torch.zeros_like(euler_angles)
#         euler_angle_rates[:, 2] = command_actions[:, 3]

#         omega_desired_body = torch.bmm(rotmat_euler_to_body_rates, euler_angle_rates.unsqueeze(2)).squeeze(2)

#         # omega_des_body = [0, 0, yaw_rate] ## approximated body_rate as yaw_rate
#         # omega_body = R_t @ omega_world
#         # angvel_err = omega_body - R_t @ R_des @ omega_des_body
#         # Refer to Lee et. al. (2010) for details (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717652)

#         desired_angvel_err = torch.bmm(rotation_matrix_transpose, torch.bmm(
#             rotation_matrix_desired, omega_desired_body.unsqueeze(2))).squeeze(2)

#         actual_angvel_err = torch.bmm(
#             rotation_matrix_transpose, robot_state[:, 10:13].unsqueeze(2)).squeeze(2)
#         angvel_err = actual_angvel_err - desired_angvel_err

#         torque = - self.K_rot_tensor * rot_err - self.K_angvel_tensor * angvel_err + torch.cross(robot_state[:, 10:13],robot_state[:, 10:13], dim=1)

#         return thrust_command, torque
    
# # Example usage of the lee velocity controller from aerial gym environment
#         # clear actions for reset envs
#         self.forces[:] = 0.0
#         self.torques[:, :] = 0.0

#         output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, self.action_input)
#         self.forces[:, 0, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized
#         self.torques[:, 0] = output_torques_inertia_normalized
#         self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)

#         # apply actions
#         self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
#             self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)


#ATTEMPTED TRANSLATION OF THE LEE VELOCITY CONTROLLER FROM PYTORCH TO NUMPY
class LeeVelocityController:
    def __init__(self, k_vel, k_rot, k_angvel):
        self.k_vel_ = k_vel
        self.k_rot_ = k_rot
        self.k_angvel = k_angvel

    def __call__(self, robot_state, command_actions):
        """
        Lee velocity controller
        :param robot_state: array of shape (12) with state of the robot
        :param command_actions: array of shape (4) with desired velocity setpoint in vehicle frame and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and inertial normalized torques
        """
        # Perform calculation for transformation matrices
        rotation_matrix = geom.Rzyx(robot_state[3], robot_state[4], robot_state[5])
        rotation_matrix_transpose = rotation_matrix.T
        euler_angles = robot_state[3:6]

        # Convert to vehicle frame
        vehicle_frame_euler = np.zeros_like(euler_angles)
        vehicle_frame_euler[2] = euler_angles[2]
        vehicle_vels = robot_state[6:9]
        
        vehicle_frame_transforms = geom.Rzyx(vehicle_frame_euler[0], vehicle_frame_euler[1], vehicle_frame_euler[2])
        vehicle_frame_transforms_transpose = vehicle_frame_transforms.T

        vehicle_frame_velocity = vehicle_frame_transforms_transpose @ vehicle_vels

        desired_vehicle_velocity = command_actions[:3]

        # Compute desired accelerations
        vel_error = desired_vehicle_velocity - vehicle_frame_velocity
        accel_command = self.k_vel_ * vel_error
        accel_command[2] += 1 
        forces_command = accel_command
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
        self.des_att = euler_setpoints
        # perform computation on calculated values
        rotation_matrix_desired = geom.Rzyx(euler_setpoints[0], euler_setpoints[1], euler_setpoints[2])
        rotation_matrix_desired_transpose = rotation_matrix_desired.T

        rot_err_mat = np.matmul(rotation_matrix_desired_transpose, rotation_matrix) - np.matmul(rotation_matrix_transpose, rotation_matrix_desired)
        rot_err = 0.5 * geom.vee_map(rot_err_mat)

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
        
        desired_angvel_err = np.matmul(rotation_matrix_transpose, np.matmul(rotation_matrix_desired, omega_desired_body[:, np.newaxis]))

        actual_angvel_err = np.matmul(rotation_matrix_transpose, robot_state[9:12, np.newaxis])

        angvel_err = actual_angvel_err - desired_angvel_err

        torque = - self.k_rot_ * rot_err - self.k_angvel * angvel_err + np.reshape(np.cross(robot_state[9:12],ss.Ig@robot_state[9:12]),(3,1))
        
        return thrust_command, torque         

if __name__ == "__main__":
    # Define the controller gains
    K_vel = torch.tensor([1.0, 1.0, 1.0])
    K_rot = torch.tensor([1.0, 1.0, 1.0])
    K_angvel = torch.tensor([1.0, 1.0, 1.0])

    # Create the controller
    controller = LeeVelocityController(K_vel, K_rot, K_angvel)

    # Create the robot state and the command actions
    robot_state = torch.rand(1, 13)
    command_actions = torch.rand(1, 4)

    # Get the thrust and torque
    thrust, torque = controller(robot_state, command_actions)
    print(thrust, torque)