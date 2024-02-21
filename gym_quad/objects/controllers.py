# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

# Position and Velocity Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/PositionControl.cpp
# Desired Thrust to Desired Attitude based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/Utility/ControlMath.cpp
# Attitude Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp
# and https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf
# Rate Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/mc_att_control_main.cpp

import numpy as np
from numpy import pi
from numpy import sin, cos, tan, sqrt
from numpy.linalg import norm

# Select Orientation of Quadcopter and Reference Frame
# ---------------------------
# "NED" for front-right-down (frd) and North-East-Down
# "ENU" for front-left-up (flu) and East-North-Up
orient = "ENU"

# Select whether to use gyroscopic precession of the rotors in the quadcopter dynamics
# ---------------------------
# Set to False if rotor inertia isn't known (gyro precession has negigeable effect on drone dynamics)
usePrecession = bool(False)
#--------------------------------

# Quaternion Functions ----------
# Normalize quaternion, or any vector
def vectNormalize(q):
    return q/norm(q)


# Quaternion multiplication
def quatMultiply(q, p):
    Q = np.array([[q[0], -q[1], -q[2], -q[3]],
                  [q[1],  q[0], -q[3],  q[2]],
                  [q[2],  q[3],  q[0], -q[1]],
                  [q[3], -q[2],  q[1],  q[0]]])
    return Q@p


# Inverse quaternion
def inverse(q):
    qinv = np.array([q[0], -q[1], -q[2], -q[3]])/norm(q)
    return qinv
#--------------------------------


# Mixer Function ---------------
# Convert desired thrust and torques to motor commands i.e desired angular velocities of the motors
def mixerFM(quad, thr, moment):
    t = np.array([thr, moment[0], moment[1], moment[2]])
    w_cmd = np.sqrt(np.clip(np.dot(quad.params["mixerFMinv"], t), quad.params["minWmotor"]**2, quad.params["maxWmotor"]**2))

    return w_cmd
#--------------------------------

# Rot conversion Functions ---------------
def quat2Dcm(q):
    dcm = np.zeros([3,3])

    dcm[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    dcm[0,1] = 2.0*(q[1]*q[2] - q[0]*q[3])
    dcm[0,2] = 2.0*(q[1]*q[3] + q[0]*q[2])
    dcm[1,0] = 2.0*(q[1]*q[2] + q[0]*q[3])
    dcm[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    dcm[1,2] = 2.0*(q[2]*q[3] - q[0]*q[1])
    dcm[2,0] = 2.0*(q[1]*q[3] - q[0]*q[2])
    dcm[2,1] = 2.0*(q[2]*q[3] + q[0]*q[1])
    dcm[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    return dcm

def RotToQuat(R):
    
    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]
    # From page 68 of MotionGenesis book
    tr = R11 + R22 + R33

    if tr > R11 and tr > R22 and tr > R33:
        e0 = 0.5 * np.sqrt(1 + tr)
        r = 0.25 / e0
        e1 = (R32 - R23) * r
        e2 = (R13 - R31) * r
        e3 = (R21 - R12) * r
    elif R11 > R22 and R11 > R33:
        e1 = 0.5 * np.sqrt(1 - tr + 2*R11)
        r = 0.25 / e1
        e0 = (R32 - R23) * r
        e2 = (R12 + R21) * r
        e3 = (R13 + R31) * r
    elif R22 > R33:
        e2 = 0.5 * np.sqrt(1 - tr + 2*R22)
        r = 0.25 / e2
        e0 = (R13 - R31) * r
        e1 = (R12 + R21) * r
        e3 = (R23 + R32) * r
    else:
        e3 = 0.5 * np.sqrt(1 - tr + 2*R33)
        r = 0.25 / e3
        e0 = (R21 - R12) * r
        e1 = (R13 + R31) * r
        e2 = (R23 + R32) * r

    # e0,e1,e2,e3 = qw,qx,qy,qz
    q = np.array([e0,e1,e2,e3])
    q = q*np.sign(e0)
    
    q = q/np.sqrt(np.sum(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2))
    
    return q
#--------------------------------


# Constants
rad2deg = 180.0/pi
deg2rad = pi/180.0

# Set PID Gains and Max Values
# ---------------------------

# Position P gains
Py    = 1.0
Px    = Py
Pz    = 1.0

pos_P_gain = np.array([Px, Py, Pz])

# Velocity P-D gains
Pxdot = 5.0
Dxdot = 0.5
Ixdot = 5.0

Pydot = Pxdot
Dydot = Dxdot
Iydot = Ixdot

Pzdot = 4.0
Dzdot = 0.5
Izdot = 5.0

vel_P_gain = np.array([Pxdot, Pydot, Pzdot])
vel_D_gain = np.array([Dxdot, Dydot, Dzdot])
vel_I_gain = np.array([Ixdot, Iydot, Izdot])

# Attitude P gains
Pphi = 8.0
Ptheta = Pphi
Ppsi = 1.5
PpsiStrong = 8

att_P_gain = np.array([Pphi, Ptheta, Ppsi])

# Rate P-D gains
Pp = 1.5
Dp = 0.04

Pq = Pp
Dq = Dp 

Pr = 1.0
Dr = 0.1

rate_P_gain = np.array([Pp, Pq, Pr])
rate_D_gain = np.array([Dp, Dq, Dr])

# Max Velocities
uMax = 5.0
vMax = 5.0
wMax = 5.0

velMax = np.array([uMax, vMax, wMax])
velMaxAll = 5.0

saturateVel_separetely = False

# Max tilt
tiltMax = 50.0*deg2rad

# Max Rate
pMax = 200.0*deg2rad
qMax = 200.0*deg2rad
rMax = 150.0*deg2rad

rateMax = np.array([pMax, qMax, rMax])

class Control:
    
    def __init__(self, quad, yawType):
        self.sDesCalc = np.zeros(16)
        self.w_cmd = np.ones(4)*quad.params["w_hover"]
        self.thr_int = np.zeros(3)
        if (yawType == 0):
            att_P_gain[2] = 0
        self.setYawWeight()
        self.pos_sp    = np.zeros(3)
        self.vel_sp    = np.zeros(3)
        self.acc_sp    = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp    = np.zeros(3)
        self.pqr_sp    = np.zeros(3)
        self.yawFF     = np.zeros(3)

    
    def controller(self, traj, quad, sDes, Ts):

        # Desired State (Create a copy, hence the [:]) sp=SetPoint
        # ---------------------------
        self.pos_sp[:]    = traj.sDes[0:3]
        self.vel_sp[:]    = traj.sDes[3:6]
        self.acc_sp[:]    = traj.sDes[6:9]
        self.thrust_sp[:] = traj.sDes[9:12]
        self.eul_sp[:]    = traj.sDes[12:15]
        self.pqr_sp[:]    = traj.sDes[15:18]
        self.yawFF[:]     = traj.sDes[18]
        
        # Select Controller
        # ---------------------------
        if (traj.ctrlType == "xyz_vel"):
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)
        elif (traj.ctrlType == "xy_vel_z_pos"):
            self.z_pos_control(quad, Ts)
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)
        elif (traj.ctrlType == "xyz_pos"):
            self.z_pos_control(quad, Ts)
            self.xy_pos_control(quad, Ts)
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)

        # Mixer
        # --------------------------- 
        self.w_cmd = mixerFM(quad, norm(self.thrust_sp), self.rateCtrl)
        
        # Add calculated Desired States
        # ---------------------------         
        self.sDesCalc[0:3] = self.pos_sp
        self.sDesCalc[3:6] = self.vel_sp
        self.sDesCalc[6:9] = self.thrust_sp
        self.sDesCalc[9:13] = self.qd
        self.sDesCalc[13:16] = self.rate_sp


    def z_pos_control(self, quad, Ts):
       
        # Z Position Control
        # --------------------------- 
        pos_z_error = self.pos_sp[2] - quad.pos[2]
        self.vel_sp[2] += pos_P_gain[2]*pos_z_error
        
    
    def xy_pos_control(self, quad, Ts):

        # XY Position Control
        # --------------------------- 
        pos_xy_error = (self.pos_sp[0:2] - quad.pos[0:2])
        self.vel_sp[0:2] += pos_P_gain[0:2]*pos_xy_error
        
        
    def saturateVel(self):

        # Saturate Velocity Setpoint
        # --------------------------- 
        # Either saturate each velocity axis separately, or total velocity (prefered)
        if (saturateVel_separetely):
            self.vel_sp = np.clip(self.vel_sp, -velMax, velMax)
        else:
            totalVel_sp = norm(self.vel_sp)
            if (totalVel_sp > velMaxAll):
                self.vel_sp = self.vel_sp/totalVel_sp*velMaxAll


    def z_vel_control(self, quad, Ts):
        
        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to 
        # allow hover when the position and velocity error are nul
        vel_z_error = self.vel_sp[2] - quad.vel[2]
        if (orient == "NED"):
            thrust_z_sp = vel_P_gain[2]*vel_z_error - vel_D_gain[2]*quad.vel_dot[2] + quad.params["mB"]*(self.acc_sp[2] - quad.params["g"]) + self.thr_int[2]
        elif (orient == "ENU"):
            thrust_z_sp = vel_P_gain[2]*vel_z_error - vel_D_gain[2]*quad.vel_dot[2] + quad.params["mB"]*(self.acc_sp[2] + quad.params["g"]) + self.thr_int[2]
        
        # Get thrust limits
        if (orient == "NED"):
            # The Thrust limits are negated and swapped due to NED-frame
            uMax = -quad.params["minThr"]
            uMin = -quad.params["maxThr"]
        elif (orient == "ENU"):
            uMax = quad.params["maxThr"]
            uMin = quad.params["minThr"]

        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[2] += vel_I_gain[2]*vel_z_error*Ts * quad.params["useIntergral"]
            # Limit thrust integral
            self.thr_int[2] = min(abs(self.thr_int[2]), quad.params["maxThr"])*np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    
    def xy_vel_control(self, quad, Ts):
        
        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------
        vel_xy_error = self.vel_sp[0:2] - quad.vel[0:2]
        thrust_xy_sp = vel_P_gain[0:2]*vel_xy_error - vel_D_gain[0:2]*quad.vel_dot[0:2] + quad.params["mB"]*(self.acc_sp[0:2]) + self.thr_int[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(self.thrust_sp[2])*np.tan(tiltMax)
        thrust_max_xy = sqrt(quad.params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if (np.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy**2):
            mag = norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp/mag*thrust_max_xy
        
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0/vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2])*arw_gain
        self.thr_int[0:2] += vel_I_gain[0:2]*vel_err_lim*Ts * quad.params["useIntergral"]
    
    def thrustToAttitude(self, quad, Ts):
        
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = self.eul_sp[2]

        # Desired body_z axis direction
        body_z = -vectNormalize(self.thrust_sp)
        if (orient == "ENU"):
            body_z = -body_z
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = np.array([-sin(yaw_sp), cos(yaw_sp), 0.0])
        
        # Desired body_x axis direction
        body_x = np.cross(y_C, body_z)
        body_x = vectNormalize(body_x)
        
        # Desired body_y axis direction
        body_y = np.cross(body_z, body_x)

        # Desired rotation matrix
        R_sp = np.array([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        self.qd_full = RotToQuat(R_sp)
        
        
    def attitude_control(self, quad, Ts):

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = quad.dcm[:,2]
        e_z_d = -vectNormalize(self.thrust_sp)
        if (orient == "ENU"):
            e_z_d = -e_z_d

        # Quaternion error between the 2 vectors
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + sqrt(norm(e_z)**2 * norm(e_z_d)**2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = vectNormalize(qe_red)
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        self.qd_red = quatMultiply(qe_red, quad.quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = quatMultiply(inverse(self.qd_red), self.qd_full)
        q_mix = q_mix*np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        self.qd = quatMultiply(self.qd_red, np.array([cos(self.yaw_w*np.arccos(q_mix[0])), 0, 0, sin(self.yaw_w*np.arcsin(q_mix[3]))]))

        # Resulting error quaternion
        self.qe = quatMultiply(inverse(quad.quat), self.qd)

        # Create rate setpoint from quaternion error
        self.rate_sp = (2.0*np.sign(self.qe[0])*self.qe[1:4])*att_P_gain
        
        # Limit yawFF
        self.yawFF = np.clip(self.yawFF, -rateMax[2], rateMax[2])

        # Add Yaw rate feed-forward
        self.rate_sp += quat2Dcm(inverse(quad.quat))[:,2]*self.yawFF

        # Limit rate setpoint
        self.rate_sp = np.clip(self.rate_sp, -rateMax, rateMax)


    def rate_control(self, quad, Ts):
        
        # Rate Control
        # ---------------------------
        rate_error = self.rate_sp - quad.omega
        self.rateCtrl = rate_P_gain*rate_error - rate_D_gain*quad.omega_dot     # Be sure it is right sign for the D part
        

    def setYawWeight(self):
        
        # Calculate weight of the Yaw control gain
        roll_pitch_gain = 0.5*(att_P_gain[0] + att_P_gain[1])
        self.yaw_w = np.clip(att_P_gain[2]/roll_pitch_gain, 0.0, 1.0)

        att_P_gain[2] = roll_pitch_gain




##DRONE LAB NTNU LINEAR VELOCITY CONTROLLER
        
# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytorch3d.transforms as p3d_transforms


from torch import Tensor
@torch.jit.script
def compute_vee_map(skew_matrix):
    # type: (Tensor) -> Tensor

    # return vee map of skew matrix
    vee_map = torch.stack(
        [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
    return vee_map

class LeeVelocityController:
    def __init__(self, K_vel_tensor, K_rot_tensor, K_angvel_tensor):
        self.K_vel_tensor = K_vel_tensor
        self.K_rot_tensor = K_rot_tensor
        self.K_angvel_tensor = K_angvel_tensor

    def __call__(self, robot_state, command_actions):
        """
        Lee velocity controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired velocity setpoint in vehicle frame and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        # perform calculation for transformation matrices
        rotation_matrices = p3d_transforms.quaternion_to_matrix(
            robot_state[:, [6, 3, 4, 5]])
        rotation_matrix_transpose = torch.transpose(rotation_matrices, 1, 2)
        euler_angles = p3d_transforms.matrix_to_euler_angles(
            rotation_matrices, "ZYX")[:, [2, 1, 0]]


        # Convert to vehicle frame
        vehicle_frame_euler = torch.zeros_like(euler_angles)
        vehicle_frame_euler[:, 2] = euler_angles[:, 2]
        vehicle_vels = robot_state[:, 7:10].unsqueeze(2)
        
        vehicle_frame_transforms = p3d_transforms.euler_angles_to_matrix(
            vehicle_frame_euler[:, [2, 1, 0]], "ZYX")
        vehicle_frame_transforms_transpose = torch.transpose(vehicle_frame_transforms, 1, 2)

        vehicle_frame_velocity = (
            vehicle_frame_transforms_transpose @ vehicle_vels).squeeze(2)

        desired_vehicle_velocity = command_actions[:, :3]

        # Compute desired accelerations
        vel_error = desired_vehicle_velocity - vehicle_frame_velocity
        accel_command = self.K_vel_tensor * vel_error
        accel_command[:, 2] += 1

        forces_command = accel_command
        thrust_command = torch.sum(forces_command * rotation_matrices[:, :, 2], dim=1)

        c_phi_s_theta = forces_command[:, 0]
        s_phi = -forces_command[:, 1]
        c_phi_c_theta = forces_command[:, 2]

        # Calculate euler setpoints
        pitch_setpoint = torch.atan2(c_phi_s_theta, c_phi_c_theta)
        roll_setpoint = torch.atan2(s_phi, torch.sqrt(
            c_phi_c_theta**2 + c_phi_s_theta**2))
        yaw_setpoint = euler_angles[:, 2]


        euler_setpoints = torch.zeros_like(euler_angles)
        euler_setpoints[:, 0] = roll_setpoint
        euler_setpoints[:, 1] = pitch_setpoint
        euler_setpoints[:, 2] = yaw_setpoint

        # perform computation on calculated values
        rotation_matrix_desired = p3d_transforms.euler_angles_to_matrix(
            euler_setpoints[:, [2, 1, 0]], "ZYX")
        rotation_matrix_desired_transpose = torch.transpose(
            rotation_matrix_desired, 1, 2)
        
        rot_err_mat = torch.bmm(rotation_matrix_desired_transpose, rotation_matrices) - \
            torch.bmm(rotation_matrix_transpose, rotation_matrix_desired)
        rot_err = 0.5 * compute_vee_map(rot_err_mat)

        rotmat_euler_to_body_rates = torch.zeros_like(rotation_matrices)

        s_pitch = torch.sin(euler_angles[:, 1])
        c_pitch = torch.cos(euler_angles[:, 1])

        s_roll = torch.sin(euler_angles[:, 0])
        c_roll = torch.cos(euler_angles[:, 0])

        rotmat_euler_to_body_rates[:, 0, 0] = 1.0
        rotmat_euler_to_body_rates[:, 1, 1] = c_roll
        rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
        rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
        rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
        rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch

        euler_angle_rates = torch.zeros_like(euler_angles)
        euler_angle_rates[:, 2] = command_actions[:, 3]

        omega_desired_body = torch.bmm(rotmat_euler_to_body_rates, euler_angle_rates.unsqueeze(2)).squeeze(2)

        # omega_des_body = [0, 0, yaw_rate] ## approximated body_rate as yaw_rate
        # omega_body = R_t @ omega_world
        # angvel_err = omega_body - R_t @ R_des @ omega_des_body
        # Refer to Lee et. al. (2010) for details (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717652)

        desired_angvel_err = torch.bmm(rotation_matrix_transpose, torch.bmm(
            rotation_matrix_desired, omega_desired_body.unsqueeze(2))).squeeze(2)

        actual_angvel_err = torch.bmm(
            rotation_matrix_transpose, robot_state[:, 10:13].unsqueeze(2)).squeeze(2)
        angvel_err = actual_angvel_err - desired_angvel_err

        torque = - self.K_rot_tensor * rot_err - self.K_angvel_tensor * angvel_err + torch.cross(robot_state[:, 10:13],robot_state[:, 10:13], dim=1)

        return thrust_command, torque
    
# Example usage of the lee velocity controller from aerial gym environment
        # clear actions for reset envs
        self.forces[:] = 0.0
        self.torques[:, :] = 0.0

        output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, self.action_input)
        self.forces[:, 0, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized
        self.torques[:, 0] = output_torques_inertia_normalized
        self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
            self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)


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