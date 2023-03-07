import unittest
import numpy as np

import sys;
sys.path.append("/cluster/work/orjanic/gym_quad")

import gym_quad.objects.quad as quad
import gym_quad.utils.state_space as ss
import gym_quad.utils.geomutils as geom

h = 0.01
g = ss.g
m = ss.m
l = ss.l
lamb = ss.lamb
Ig = ss.Ig
thrust_max = ss.thrust_max

k_lin = np.array([ss.k_u, ss.k_v, ss.k_w])
k_rot = np.array([ss.k_p, ss.k_q, ss.k_r])

# Rotation from body to world
def R_wb(x):
    phi = x[0]
    theta = x[1]
    psi = x[2]
    return np.array([
        [np.cos(theta)*np.cos(psi), -np.cos(phi)*np.sin(psi)+np.sin(phi)*np.sin(theta)*np.cos(psi), np.sin(phi)*np.sin(psi)+np.cos(phi)*np.sin(theta)*np.cos(psi)],
        [np.cos(theta)*np.sin(psi), np.cos(phi)*np.cos(psi)+np.sin(phi)*np.sin(theta)*np.sin(psi), -np.sin(phi)*np.cos(psi)+np.cos(phi)*np.sin(theta)*np.sin(psi)],
        [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]
    ])


# Rotation from world to body
def R_bw(x):
    return np.transpose(R_wb(x))


# Transformation from body to world
def T_wb(x):
    phi = x[0]
    theta = x[1]
    return np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])


# Transformation from world to body
def T_bw(x):
    phi = x[0]
    theta = x[1]
    return np.array([
        [1, 0, -np.sin(theta)],
        [0, np.cos(phi), np.sin(phi)*np.cos(theta)],
        [0, -np.sin(phi), np.cos(phi)*np.cos(theta)]
    ])


class TestStateSpace(unittest.TestCase):
    def run_dynamics(self, init_eta, action):
        # Make sure inputs are on correct format
        init_eta[3] = geom.ssa(init_eta[3])
        init_eta[4] = geom.ssa(init_eta[4])
        init_eta[5] = geom.ssa(init_eta[5])
        for i in range(len(action)):
            action[i] = np.clip(action[i], -1, 1)

        # Initialize quadcopter
        quadcopter = quad.Quad(step_size=h, init_eta=init_eta, safety_radius=1)

        # One timestep, initial velocities are zero.
        quadcopter.step(action)
        position_sim    = quadcopter.state[0:3]
        orientation_sim = quadcopter.state[3:6]
        velocity_sim    = quadcopter.state[6:9]
        angular_vel_sim = quadcopter.state[9:12]

        a_0 = R_bw(init_eta[3:6]) @ np.array([0, 0, g]) + thrust_max * np.array([0, 0, np.sum(action)]) / m
        alpha_0 = thrust_max * np.array([l * (action[3] - action[1]), l * (action[2] - action[0]), lamb * (action[1] + action[3] - action[0] - action[2])]) / np.diagonal(Ig)
        # alpha_0 = ss.M_inv().dot(ss.B().dot(action))[3:6]

        position_true    = init_eta[0:3] + R_wb(init_eta[3:6]) @ (0.5 * a_0 * h ** 2)
        velocity_true    = a_0 * h
        orientation_true = init_eta[3:6] + T_wb(init_eta[3:6]) @ (0.5 * alpha_0 * h ** 2)
        angular_vel_true = alpha_0 * h

        orientation_true[0] = geom.ssa(orientation_true[0])
        orientation_true[1] = geom.ssa(orientation_true[1])
        orientation_true[2] = geom.ssa(orientation_true[2])

        print("\nposition_true:", position_true)
        print("position_sim :", position_sim)
        print("orientation_true:", orientation_true)
        print("orientation_sim :", orientation_sim)
        print("velocity_true:", velocity_true)
        print("velocity_sim :", velocity_sim)
        print("angular_vel_true:", angular_vel_true)
        print("angular_vel_sim :", angular_vel_sim)

        self.assertTrue(np.allclose(position_true, position_sim, rtol=1e-1, atol=1e-1))
        self.assertTrue(np.allclose(orientation_true, orientation_sim, rtol=1e-1, atol=1e-1))
        self.assertTrue(np.allclose(velocity_true, velocity_sim, rtol=1e-1, atol=1e-1))
        self.assertTrue(np.allclose(angular_vel_true, angular_vel_sim, rtol=1e-1, atol=1e-1))

        # 1000 timesteps, both gravitational force and drag acts
        for _ in range(1000):
            quadcopter.step(action)
            position_sim    = quadcopter.state[0:3]
            orientation_sim = quadcopter.state[3:6]
            velocity_sim    = quadcopter.state[6:9]
            angular_vel_sim = quadcopter.state[9:12]

            a = R_bw(orientation_true) @ np.array([0, 0, g]) + np.array([0, 0, thrust_max * np.sum(action)]) / m - k_lin * velocity_true / m # Gravitation + Thrust - Drag
            alpha = alpha_0 - np.array([(Ig[2][2] - Ig[1][1])*angular_vel_true[1]*angular_vel_true[2], (Ig[0][0] - Ig[2][2])*angular_vel_true[0]*angular_vel_true[2], 0]) / np.diagonal(Ig) - k_rot * np.square(angular_vel_true) / np.diagonal(Ig) # Thrust - Coriolis - Drag
            
            position_true    += R_wb(orientation_true) @ (velocity_true * h + 0.5 * a * h ** 2)
            velocity_true    += a * h
            orientation_true += T_wb(orientation_true) @ (angular_vel_true * h + 0.5 * alpha * h ** 2)
            angular_vel_true += alpha * h

            orientation_true[0] = geom.ssa(orientation_true[0])
            orientation_true[1] = geom.ssa(orientation_true[1])
            orientation_true[2] = geom.ssa(orientation_true[2])

            print(f"\n{_}")
            print("position_true:", position_true)
            print("position_sim :", position_sim)
            print("orientation_true:", orientation_true)
            print("orientation_sim :", orientation_sim)
            print("velocity_true:", velocity_true)
            print("velocity_sim :", velocity_sim)
            print("angular_vel_true:", angular_vel_true)
            print("angular_vel_sim :", angular_vel_sim)

            self.assertTrue(np.allclose(position_true, position_sim, rtol=1e-1, atol=1e-1))
            self.assertTrue(np.allclose(orientation_true, orientation_sim, rtol=1e-1, atol=1e-1))
            self.assertTrue(np.allclose(velocity_true, velocity_sim, rtol=1e-1, atol=1e-1))
            self.assertTrue(np.allclose(angular_vel_true, angular_vel_sim, rtol=1e-1, atol=1e-1))
        
        print("\nposition_true:", position_true)
        print("position_sim :", position_sim)
        print("orientation_true:", orientation_true)
        print("orientation_sim :", orientation_sim)
        print("velocity_true:", velocity_true)
        print("velocity_sim :", velocity_sim)
        print("angular_vel_true:", angular_vel_true)
        print("angular_vel_sim :", angular_vel_sim)

    # def test_without_torque_reproducable(self):
    #     """
    #     Test Quadcopter dynamics using 5 reproducable tests with different initializations.
    #     Initial orientation equal to zero.
    #     Applies 5 different initial positions.
    #     Applies 5 different sets of actions, not producing any torque.
    #     """
    #     for i in np.arange(0, 2.5, 0.5):
    #         init_eta = np.array([i, -2*i, 1-i, 0, 0, 0])
    #         action = np.array([-1, -1, -1, -1]) + i
    #         self.run_dynamics(init_eta=init_eta, action=action)

    # def test_without_torque_randomized(self):
    #     """
    #     Test Quadcopter dynamics using 3 tests with random initializations.
    #     Initial orientation equal to zero.
    #     Applies 3 different initial positions.
    #     Applies 3 different sets of actions, not producing any torque.
    #     """
    #     for random_numbers in np.random.uniform(low=-1, high=1, size=(3,4)):
    #         init_eta = np.array([random_numbers[0], random_numbers[1], random_numbers[2], 0, 0, 0])
    #         action = np.array([random_numbers[3], random_numbers[3], random_numbers[3], random_numbers[3]])
    #         self.run_dynamics(init_eta=init_eta, action=action)

    def test_with_torque(self):
        """
        Test Quadcopter dynamics using 5 reproducable tests with different initializations.
        Applies 5 different initial positions.
        Applies 5 different sets of actions.
        """
        init_eta = np.array([np.pi/4, np.pi/2, -5, 10*np.pi/180, -10*np.pi/180, np.pi])
        # init_eta = np.zeros(6)
        action = np.array([0.5, 0, 0.5, 0])
        self.run_dynamics(init_eta=init_eta, action=action)
        # for i in np.arange(0, 2.5, 0.5):
        #     init_eta = np.array([i, -2*i, 1-i, -i, 0, i+1])
        #     action = np.array([-1, -1, -1, -0.9]) + i
        #     self.run_dynamics(init_eta=init_eta, action=action)

if __name__ == '__main__':
    unittest.main()
