import numpy as np
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints
from gym_quad.objects.obstacle3d import Obstacle
from gym_quad.objects.current3d import Current


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
    init_attitude=np.array([0, 0, self.path.get_direction_angles(0)[0]])
    initial_state = np.hstack([init_pos, init_attitude])
    return initial_state

def scenario_3d_new(self):
    initial_state = np.zeros(6)
    waypoints = generate_random_waypoints(self.n_waypoints,'3d_new')
    self.path = QPMI(waypoints)
    # init_pos=[-10, -10, 0]
    init_pos = [np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-10,10)]
    init_pos=[0, 0, 0]
    init_attitude=np.array([0, 0, self.path.get_direction_angles(0)[0]])
    initial_state = np.hstack([init_pos, init_attitude])
    return initial_state


def scenario_intermediate(self):
    initial_state = self.scenario_3d_new()
    obstacle_radius = np.random.uniform(low=4,high=10)
    obstacle_coords = self.path(self.path.length/2)# + np.random.uniform(low=-obstacle_radius, high=obstacle_radius, size=(1,3))
    self.obstacles.append(Obstacle(radius=obstacle_radius, position=obstacle_coords))
    return initial_state


def scenario_proficient(self):
    initial_state = self.scenario_3d_new()
    obstacle_radius = np.random.uniform(low=4,high=10)
    obstacle_coords = self.path(self.path.length/2)# + np.random.uniform(low=-obstacle_radius, high=obstacle_radius, size=(1,3))
    self.obstacles.append(Obstacle(radius=obstacle_radius, position=obstacle_coords))

    lengths = np.linspace(self.path.length*1/6, self.path.length*5/6, 2)
    for l in lengths:
        obstacle_radius = np.random.uniform(low=4,high=10)
        obstacle_coords = self.path(l) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
        # print(self.path(l))
        # print(np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3)))
        # print(obstacle_coords)
        obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
        if self.check_object_overlap(obstacle):
            continue
        else:
            self.obstacles.append(obstacle)
    return initial_state


# def scenario_advanced(self):
#     initial_state = self.scenario_proficient()
#     while len(self.obstacles) < self.n_adv_obstacles: # Place the rest of the obstacles randomly
#         s = np.random.uniform(self.path.length*1/3, self.path.length*2/3)
#         obstacle_radius = np.random.uniform(low=4,high=10)
#         obstacle_coords = self.path(s) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
#         obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
#         if self.check_object_overlap(obstacle):
#             continue
#         else:
#             self.obstacles.append(obstacle)
#     return initial_state


def scenario_expert(self):
    initial_state = self.scenario_3d_new()
    obstacle_radius = np.random.uniform(low=4,high=10)
    obstacle_coords = self.path(self.path.length/2)# + np.random.uniform(low=-obstacle_radius, high=obstacle_radius, size=(1,3))
    self.obstacles.append(Obstacle(radius=obstacle_radius, position=obstacle_coords))

    lengths = np.linspace(self.path.length*1.5/6, self.path.length*5/6, 5)
    for l in lengths:
        obstacle_radius = np.random.uniform(low=4,high=10)
        obstacle_coords = self.path(l) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
        # print(self.path(l))
        # print(np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3)))
        # print(obstacle_coords)
        obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
        if self.check_object_overlap(obstacle):
            continue
        else:
            self.obstacles.append(obstacle)
    
    return initial_state


def scenario_test_path(self):
    # test_waypoints = np.array([np.array([0,0,0]), np.array([1,1,0]), np.array([9,9,0]), np.array([10,10,0])])
    # test_waypoints = np.array([np.array([0,0,0]), np.array([5,0,0]), np.array([10,0,0]), np.array([15,0,0])])
    test_waypoints = np.array([np.array([0,0,0]), np.array([10,1,0]), np.array([20,0,0]), np.array([70,0,0])])
    self.n_waypoints = len(test_waypoints)
    self.path = QPMI(test_waypoints)
    init_pos = [0,0,0]
    init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
    initial_state = np.hstack([init_pos, init_attitude])
    self.obstacles.append(Obstacle(radius=10, position=self.path(20)))
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
    waypoints = [(0,0,0), (50,0.1,0), (100,0,0)]
    self.path = QPMI(waypoints)
    self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
    self.obstacles = []
    for i in range(7):
        y = -30+10*i
        self.obstacles.append(Obstacle(radius=5, position=[50,y,0]))
    init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-5, high=5, size=(1,3))
    init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
    initial_state = np.hstack([init_pos[0], init_attitude])
    return initial_state


def scenario_vertical_test(self):
    waypoints = [(0,0,0), (50,0,1), (100,0,0)]
    self.path = QPMI(waypoints)
    self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
    self.obstacles = []
    for i in range(7):
        z = -30+10*i
        self.obstacles.append(Obstacle(radius=5, position=[50,0,z]))
    init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-5, high=5, size=(1,3))
    init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
    initial_state = np.hstack([init_pos[0], init_attitude])
    return initial_state


def scenario_deadend_test(self):
    waypoints = [(0,0,0), (50,0.5,0), (100,0,0)]
    self.path = QPMI(waypoints)
    self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
    radius = 10
    angles = np.linspace(-90, 90, 10)*np.pi/180
    obstalce_radius = (angles[1]-angles[0])*radius/2
    for ang1 in angles:
        for ang2 in angles:
            x = 45 + radius*np.cos(ang1)*np.cos(ang2)
            y = radius*np.cos(ang1)*np.sin(ang2)
            z = -radius*np.sin(ang1)
            self.obstacles.append(Obstacle(obstalce_radius, [x, y, z]))
    init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-5, high=5, size=(1,3))
    init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
    initial_state = np.hstack([init_pos[0], init_attitude])
    return initial_state


def scenario_helix(self):
    initial_state = np.zeros(6)
    waypoints = generate_random_waypoints(self.n_waypoints,'helix')
    self.path = QPMI(waypoints)
    # init_pos = helix_param(0)
    init_pos = np.array([110, 0, -26]) + np.random.uniform(low=-5, high=5, size=(1,3))
    init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
    # init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
    initial_state = np.hstack([init_pos[0], init_attitude])
    self.obstacles.append(Obstacle(radius=100, position=[0,0,0]))
    return initial_state