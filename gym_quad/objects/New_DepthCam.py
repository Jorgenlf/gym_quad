'''Use taichi inspired by cornell and sdf examples to create a new depth camera'''

import taichi as ti
import numpy as np
import time
from taichi.math import *

ti.init(arch=ti.gpu)

res = (320, 240)
# res = (1280, 720)
sensor_span = (85,58) # horizontal, vertical in degrees
fov = [85.0*np.pi/180,58.0*np.pi/180] # horizontal, vertical in radians
sensor_range = 6 # meters
dist_limit = 100  
camera_pos = ti.Vector([0.0, 0.32, 2]) #NB #Y is up, X is right, -Z is forward #NB!!!!!!!!!!
inf = 1e10
depth_buffer = ti.Vector.field(1, float, res)

# @ti.func
# def make_nested(f):
#     '''Returns a nested pattern based on the input float f'''
#     f = f * 40
#     i = int(f)
#     if f < 0:
#         if i % 2 == 1:
#             f -= ti.floor(f)
#         else:
#             f = ti.floor(f) + 1 - f
#     f = (f - 0.2) / 40
#     return f

# https://iquilezles.org/articles/distfunctions/ 
@ti.func
def sdf(o):
    '''Signed distance function for the scene geometry
    input: o - position vector
    output: signed distance to the closest object from the position o'''
    wall = ti.min(o[1] + 0.1, o[2] + 0.4)
    sphere = (o - ti.Vector([0.0, 0.35, 0.0])).norm() - 0.36

    # q = ti.abs(o - ti.Vector([0.8, 0.3, 0])) - ti.Vector([0.3, 0.3, 0.3])
    # box = ti.Vector([ti.max(0, q[0]), ti.max(0, q[1]), ti.max(0, q[2])]).norm() + ti.min(q.max(), 0)

    # O = o - ti.Vector([-0.8, 0.3, 0])
    # d = ti.Vector([ti.Vector([O[0], O[2]]).norm() - 0.3, abs(O[1]) - 0.3])
    # cylinder = ti.min(d.max(), 0.0) + ti.Vector([ti.max(0, d[0]), ti.max(0, d[1])]).norm()

    # geometry = make_nested(ti.min(sphere, box, cylinder))
    # geometry = ti.max(geometry, -(0.32 - (o[1] * 0.6 + o[2] * 0.8)))
    geometry = sphere
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
        d = ti.Vector( #NB: this is the ray direction
            [
                (2 * fov[1] * (u) / res[1] - fov[1] * aspect_ratio - 1e-5), #could add u*ti.random() to make ray not match pixel perfectly
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