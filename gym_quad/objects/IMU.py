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
        self.stdAngAcc = noiseAngAcc
        self.stdLinAcc = noiseLinAcc
        self.measurement = np.zeros((6,))
        self.lin_noise = np.zeros((3,))
        self.ang_noise = np.zeros((3,))

    def measure(self, quad:Quad):
        """
        Measure the state of the quadcopter.
        """
        bodyaccl = quad.state_dot(quad.state)[6:9] #Speed derivative of the body frame velocity

        # R_b_to_w = geom.Rzyx(*quad.attitude)
        # gravity_effect = R_b_to_w.T @ np.array([0, 0, ss.g])
        
        self.lin_noise = np.random.normal(0, self.stdLinAcc, 3)
        #linear acceleration
        self.measurement[0:3] = bodyaccl + self.lin_noise #- gravity_effect  #Commented out gravity effect since it is already in the state_dot function
        
        self.ang_noise = np.random.normal(0, self.stdAngAcc, 3)
        #angular velocity aka angular rate
        self.measurement[3:6] = quad.angular_velocity  + self.ang_noise

        return self.measurement

    def reset(self):
        self.measurement = np.zeros((6,))   
    
    def set_std(self, noiseAngAcc, noiseLinAcc):
        self.stdAngAcc = noiseAngAcc
        self.stdLinAcc = noiseLinAcc
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    imu = IMU()
    imu.set_noise(0.01,0.1)
    quad = Quad(0.01,np.zeros(6))
    quad.state[2] = 1

    reference_type = "offset_force_then_no_force"
    reference_type = "no_force"
    reference_type = "constant_force"

    read_force_from_file = True

    if read_force_from_file:
        force_ref = np.load("gym_quad/tests/trajectories/forces_manual_input.npy")
        totT = len(force_ref)
    else:
        totT = 800
        force_ref = []
        for t in range(totT):
            if reference_type == "no_force":
                force_ref.append([0, 0, 0, 0])
            elif reference_type == "constant_force":
                force_ref.append([1.2, 1.2, 1.2, 1.2])
            elif reference_type == "offset_force_then_no_force":
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
        quad.step(np.array(force_ref[t]))
        imu_meas[:,t] = imu.measure(quad)
        state_log[:,t] = quad.state
        
        #save ax az and ay 
        #Body frame acceleration 
        axazayActual_b[:,t] = quad.state_dot(quad.state)[6:9]  
        
        #World frame acceleration 
        #Do not have access to world frame acceleration as this would be the double derivative of the position which is not in the state
        #The state is [x, y, z, phi, theta, psi, u, v, w, p, q, r] where u, v, w are velocities in the body frame
        # R_b_to_w = geom.Rzyx(*quad.attitude)
        # axazayActual_b[:,t] = quad.state_dot(quad.state)[6:9] @ R_b_to_w.T 

    timesteps = np.array(timesteps)*0.01

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(timesteps, imu_meas[0,:], label='ax measured')
    plt.plot(timesteps, axazayActual_b[0,:], label='ax actual b')
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(timesteps, imu_meas[1,:], label='ay measured')
    plt.plot(timesteps, axazayActual_b[1,:], label='ay actual b')
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [s]')
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(timesteps, imu_meas[2,:], label='az measured')
    plt.plot(timesteps, axazayActual_b[2,:], label='az actual b')
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [s]')
    plt.legend()


    plt.figure(2)
    plt.subplot(3,1,1)
    plt.plot(timesteps, imu_meas[3,:], label='p measured')
    plt.plot(timesteps, state_log[9,:], label='p actual')
    plt.ylabel('Angular velocity [rad/s]')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(timesteps, imu_meas[4,:], label='q measured')
    plt.plot(timesteps, state_log[10,:], label='q actual')
    plt.ylabel('Angular velocity [rad/s]')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(timesteps, imu_meas[5,:], label='r measured')
    plt.plot(timesteps, state_log[11,:], label='r actual')
    plt.ylabel('Angular velocity [rad/s]')
    plt.xlabel('Time [s]')
    plt.legend()
    
    plt.show()
        