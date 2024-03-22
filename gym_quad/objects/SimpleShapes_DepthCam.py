'''Use taichi inspired by cornell and sdf examples to create a new depth camera'''

##### #TODO fix this import stuff (below)
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

from gym_quad.objects.quad import Quad
from gym_quad.objects.obstacle3d import Obstacle, ObstacleType
##### #TODO fix this import stuff (above) such that only the two last lines are needed

import taichi as ti
import numpy as np
import time
from taichi.math import *
from enum import Enum


#IMPORTANT DECISIONS TO MAKE ASAP
#TODO DECIDE WETHER TO USE MESHES OR SIMPLE SHAPES FOR OBSTACLES
#IF SIMPLE SHAPES NEED TO CREATE A GLOBAL OBSTACLE LIST IN THE ENVIRONMENT CLASS SUCH THAT
#THE SDF FCN CAN ACCESS IT

ti.init(arch=ti.gpu)

res = (320, 240)
# res = (1280, 720)
sensor_span = (85,58) # horizontal, vertical in degrees
fov = [85.0*np.pi/180,58.0*np.pi/180] # horizontal, vertical in radians
sensor_range = 6 # meters
dist_limit = 100  

camera_pos = ti.Vector([0.0, 0.32, 2]) #NB #Y is up, X is right, -Z is forward #NB!!!!!!!!!!
#TODO add camera attitude and make it have an effect on the ray direction
camera_att = ti.Vector([0.0, 0.0, 0.0]) #roll, pitch, yaw in radians

inf = 1e10
depth_buffer = ti.Vector.field(1, float, res)

@ti.func
def taichi_coord_to_sim_coord(p):
    '''Converts a point from the taichi coordinate Xeast,Zsouth,Yup system to the simulation XeastYnorthZup coordinate system'''
    return ti.Vector([p[0], p[2], -p[1]])

@ti.func
def sim_coord_to_taichi_coord(p):
    '''Converts a point from the simulation XeastYnorthZup coordinate system to the taichi coordinate Xeast,Zsouth,Yup system'''
    return ti.Vector([p[0], -p[2], p[1]])

@ti.func
def rotate_camera(p, att): #TODO verify which coordinate system the attitude is in
    '''Rotate the camera position p by the attitude att'''
    roll = att[0]
    pitch = att[1]
    yaw = att[2]
    R = ti.Matrix([
        [ti.cos(yaw)*ti.cos(pitch), ti.cos(yaw)*ti.sin(pitch)*ti.sin(roll) - ti.sin(yaw)*ti.cos(roll), ti.cos(yaw)*ti.sin(pitch)*ti.cos(roll) + ti.sin(yaw)*ti.sin(roll)],
        [ti.sin(yaw)*ti.cos(pitch), ti.sin(yaw)*ti.sin(pitch)*ti.sin(roll) + ti.cos(yaw)*ti.cos(roll), ti.sin(yaw)*ti.sin(pitch)*ti.cos(roll) - ti.cos(yaw)*ti.sin(roll)],
        [-ti.sin(pitch), ti.cos(pitch)*ti.sin(roll), ti.cos(pitch)*ti.cos(roll)]
    ])
    return R @ p

@ti.func
def sphere_sdf(o, p, r):
    '''Signed distance function for a sphere.
        p: center of the sphere
        r: radius
        o: distance along the ray from the camera to the point of interest
    '''
    return (o-p).norm() - r

def wall_sdf(o,offset_x=0.5,offset_y=0.5,offset_z=0.5):
    '''Signed distance function for a wall
        o: distance along the ray from the camera to the point of interest
        offset_x_y_z: offset from the origin in the x, y, z directions if inf passed no wall in that direction
    '''
    if offset_x == inf:
        return ti.min(o[1] + offset_y, o[2] + offset_z)
    elif offset_y == inf:
        return ti.min(o[0] + offset_x, o[2] + offset_z)
    elif offset_z == inf:
        return ti.min(o[0] + offset_x, o[1] + offset_y)
    elif offset_x == inf and offset_y == inf:
        return ti.min(o[2] + offset_z)
    elif offset_x == inf and offset_z == inf:
        return ti.min(o[1] + offset_y)
    elif offset_y == inf and offset_z == inf:
        return ti.min(o[0] + offset_x)
    else:
        return ti.min(o[0] + offset_x, o[1] + offset_y, o[2] + offset_z)

# https://iquilezles.org/articles/distfunctions/ 
@ti.func
def sdf(o): #TODO understand how objects are defined such that we can define them in our simulation
    '''Signed distance function for the scene geometry
    input: o - position vector (camera position + distance along the ray direction)
    output: signed distance to the closest object from the position o'''

    wall = ti.min(o[1] + 0.1, o[2] + 0.4)
    sphere = (o - ti.Vector([0.0, 0.35, 0.0])).norm() - 0.36

    # q = ti.abs(o - ti.Vector([0.8, 0.3, 0])) - ti.Vector([0.3, 0.3, 0.3])
    # box = ti.Vector([ti.max(0, q[0]), ti.max(0, q[1]), ti.max(0, q[2])]).norm() + ti.min(q.max(), 0)

    # O = o - ti.Vector([-0.8, 0.3, 0])
    # d = ti.Vector([ti.Vector([O[0], O[2]]).norm() - 0.3, abs(O[1]) - 0.3])
    # cylinder = ti.min(d.max(), 0.0) + ti.Vector([ti.max(0, d[0]), ti.max(0, d[1])]).norm()
    
    # geometry = make_nested(ti.min(sphere, box, cylinder))
    geometry = sphere
    # geometry = ti.max(geometry, -(0.32 - (o[1] * 0.6 + o[2] * 0.8)))
    return ti.min(wall, geometry)

@ti.func
def ray_march(p, d):
    '''Returns the distance to the closest object along the ray direction d starting from point p'''
    j = 0
    dist = 0.0
    while j < 100 and sdf(p + dist * d) > 1e-6 and dist < inf:
        dist += sdf(p + dist * d)
        j += 1
    return ti.min(inf, dist)

@ti.func
def next_hit(pos, d):
    '''
    Input: position of the camera, direction of the ray d
    Returns: the distance to the closest object along the ray direction d starting from point pos, 
    '''
    closest = inf
    ray_march_dist = ray_march(pos, d)
    if ray_march_dist < dist_limit and ray_march_dist < closest:
        closest = ray_march_dist
    return closest

@ti.kernel
def capture_depth_image():
    for u, v in depth_buffer:
        aspect_ratio = res[0] / res[1]
        pos = camera_pos
        d = ti.Vector( #the ray direction
            [
                (2 * fov[1] * (u) / res[1] - fov[1] * aspect_ratio - 1e-5), #could add u*ti.random() and v*ti.random() to make ray not match pixel perfectly
                2 * fov[0] * (v) / res[1] - fov[0] - 1e-5,
                -1.0,
            ]
        )
        d = d.normalized()

        depth = 0

        while depth < sensor_range:
            closest = next_hit(pos, d)
            depth += 1
            if closest < inf:
                depth_buffer[u, v] = closest
                depth = sensor_range
            else:
                hit_pos = pos + closest * d
                pos = hit_pos + 1e-4 * d

def main():
    import matplotlib.pyplot as plt
    
    mode = 'plot' # 'render' or 'plot'
    
    obstacle1 = Obstacle(0.36,[0.0, 0.35, 0.0])
    obstacle2 = Obstacle(0.3, [0.8, 0.3, 0])

    if mode == 'render':
        gui = ti.GUI("SDF Path Tracer", res)
        last_t = 0

        for i in range(50000):
            capture_depth_image()
            interval = 10
            if i % interval == 0 and i > 0:
                print(f"{interval / (time.time() - last_t):.2f} samples/s")
                last_t = time.time()
                img = depth_buffer.to_numpy() * (1 / (i + 1))
                img = img / img.mean() * 0.24
                gui.set_image(np.sqrt(img))
                gui.show()
    elif mode == 'plot':
        capture_depth_image()
        img = depth_buffer.to_numpy()
        img = np.rot90(img, 1)
        plt.imshow(img,cmap='viridis')
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()