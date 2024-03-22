import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
import taichi as ti

class ObstacleType(Enum):
    WALL = 0
    SPHERE = 1
    BOX = 2
    CYLINDER = 3
    # Add more obstacle types as needed

#TODO Make the obstacle class into a superclass and then make subclasses for each type of obstacle
class Obstacle():
    def __init__(self, radius, position, type=ObstacleType.SPHERE):
        self.position = np.array(position)
        self.radius = radius
        self.type = type
        self.observed = False
        self.collided = False


    def return_plot_variables(self):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.position[0] + self.radius*np.cos(u)*np.sin(v)
        y = self.position[1] + self.radius*np.sin(u)*np.sin(v)
        z = self.position[2] + self.radius*np.cos(v)
        return [x,y,z]


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    o1 = Obstacle(3, [0,0,0])
    ax.plot_surface(*o1.return_plot_variables())
    o2 = Obstacle(5, [10,0,0])
    ax.plot_surface(*o2.return_plot_variables(), color='r')
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_zlim([-20,20])
    plt.show()
