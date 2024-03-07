#This is done to make the import below work However should work by itself #TODO low priority
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


import numpy as np
import gym_quad.utils.state_space as ss
import gym_quad.utils.geomutils as geom
from gym_quad.objects.quad import Quad

class IMU():
    """
    Simple Inertial Measurement Unit (IMU) class. Might make it more complex later.
    measurement = [ax, ay, az, p, q, r].T where a is acceleration and p, q, r are angular velocities.
    """
    def __init__(self,  noiseAngAcc=0.0, noiseLinAcc=0.0):
        self.noiseAngAcc = noiseAngAcc
        self.noiseLinAcc = noiseLinAcc
        self.measurement = np.zeros((6,))

    def measure(self, quad:Quad):
        """
        Measure the state of the quadcopter.
        """
        # Add noise to the state #TODO implement later
        bodyaccl = quad.state_dot(quad.state)[6:9] #Speed derivative of the body frame velocity
        # R_b_to_w = geom.Rzyx(*quad.attitude)

        self.measurement[0:3] = bodyaccl #@ R_b_to_w.T + np.random.normal(0, self.noiseLinAcc, 3) #linear acceleration
        self.measurement[3:6] = quad.angular_velocity # + np.random.normal(0, self.noiseAngAcc, 3) #angular velocity aka angular rate

        return self.measurement

    def reset(self):
        self.measurement = np.zeros((6,))   
    
    def set_noise(self, noiseAngAcc, noiseLinAcc):
        self.noiseAngAcc = noiseAngAcc
        self.noiseLinAcc = noiseLinAcc
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    imu = IMU()
    # imu.set_noise(0.01,0.1)
    quad = Quad(0.01,np.zeros(6))
    quad.state[2] = 1

    totT = 800
    force_ref = []

    for t in range(totT):
        if t > 100 and t <110:
            force_ref.append([1.2, 1, 1.2, 1])
        else: 
            force_ref.append([0, 0, 0, 0])

    timesteps = []
    imu_meas = np.zeros((6, totT))
    state_log = np.zeros((12, totT))
    axazayActual_w = np.zeros((3, totT))
    axazayActual_b = np.zeros((3, totT))
    for t in range(totT):
        timesteps.append(t)
        quad.step(force_ref[t])
        imu_meas[:,t] = imu.measure(quad)
        state_log[:,t] = quad.state
        #save ax az and ay 
        axazayActual_w[:,t] = quad.state_dot(quad.state)[0:3]  #World frame acceleration Should mabye be in body frame
        R_b_to_w = geom.Rzyx(*quad.attitude)
        axazayActual_b[:,t] = quad.state_dot(quad.state)[0:3] @ R_b_to_w.T #Body frame acceleration Should mabye be in body frame

    timesteps = np.array(timesteps)*0.01

    plt.figure(1)
    plt.plot(timesteps, imu_meas[0,:], label='ax measured')
    plt.plot(timesteps, imu_meas[1,:], label='ay measured')
    plt.plot(timesteps, imu_meas[2,:], label='az measured')
    # plt.plot(timesteps, axazayActual_w[0,:], label='ax actual w')
    # plt.plot(timesteps, axazayActual_w[1,:], label='ay actual w')
    # plt.plot(timesteps, axazayActual_w[2,:], label='az actual w')
    plt.plot(timesteps, axazayActual_b[0,:], label='ax actual b')
    plt.plot(timesteps, axazayActual_b[1,:], label='ay actual b')
    plt.plot(timesteps, axazayActual_b[2,:], label='az actual b')
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.figure(2)
    plt.plot(timesteps, imu_meas[3,:], label='p measured')
    plt.plot(timesteps, imu_meas[4,:], label='q measured')
    plt.plot(timesteps, imu_meas[5,:], label='r measured')
    plt.plot(timesteps, state_log[9,:], label='p actual')
    plt.plot(timesteps, state_log[10,:], label='q actual')
    plt.plot(timesteps, state_log[11,:], label='r actual')
    plt.ylabel('Angular velocity [rad/s]')
    plt.xlabel('Time [s]')
    plt.legend()
    
    plt.show()
        