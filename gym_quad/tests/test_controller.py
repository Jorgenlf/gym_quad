
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
        self.s_max = 0.5
        self.i_max = 0.5
        self.omega_max = 0.5

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
        #Using hyperparam of s_max, i_max, omega_max to clip the action to get the commanded velocity and yaw rate
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(self.i_max * action[1])
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(self.i_max * action[1])
        cmd_omega_z = self.omega_max * action[2]

        #Calculate the thrust inputs using the commanded velocity and yaw rate and a PD controller
        v_error = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - self.quadcopter.velocity
        omega_z_error = cmd_omega_z - self.quadcopter.angular_velocity[2]

        #For the yaw rate, the natural frequency and damping ratio are set to 9 and 1 respectively    
        omega_n = 9 # Natural frequency
        xi = 1      # Damping ratio, 1 -> critically damped

        k_p_omega_z = ss.I_z * omega_n**2
        k_d_omega_z =  2 * ss.I_z * xi * omega_n

        K_p = np.diag([1.0, 1.0, 1.0])
        K_d = np.diag([0.5, 0.5, 0.5])

        u = np.zeros(4) 
        u[0] = omega_z_error*k_p_omega_z + omega_z_error*k_d_omega_z

        desired_acc = np.array([0, 0, 0])#????????
        u[1:] = np.dot(K_p, v_error) + np.dot(K_d, self.quadcopter.acceleration-desired_acc)     #TODO get this working before copying into LV_VAE  
        
        # B = ????? #TODO



        return F
    
    def test_velocity_controller(self):

        action = np.array([0.5, 0.5, 0.5])

        F = self.velocity_controller(action)
        
        self.assertEqual(F.shape, (4,))



if __name__ == '__main__':

    Test = TestController()
    Test.setUp()
    #Create a set of actions between -1 and 1 to test the velocity controller
    for i in range(10):
        action = np.random.uniform(-1, 1, 3)
        F = Test.velocity_controller(action)
        print(F)
        print(action)
        print("")

    #plot linear and angular velocities and control inputs


    # Run the tests
    unittest.main()