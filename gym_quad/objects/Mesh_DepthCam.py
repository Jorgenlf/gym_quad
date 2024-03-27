#Use package trimesh (and mesh-depth library if trimesh is slow)
#Pyvista for visualization of the mesh make a polydata object
#Pyvista oceanic colormaps are nice.
#Pyvista can create a mesh (might have to run .triangulate() on the mesh object)

#IF MESH2DEPTH ACTUALLY WAS ABLE TO BE DOWNLOADED THIS WOULD BE HOW IT WOULD BEHAVE:
# import numpy as np
# import mesh_to_depth as m2d

# params = []

# params.append({
#     'cam_pos': [1, 1, 1], 'cam_lookat': [0, 0, 0], 'cam_up': [0, 1, 0],
#     'x_fov': 0.349,  # End-to-end field of view in radians
#     'near': 0.1, 'far': 10,
#     'height': 240, 'width': 320,
#     'is_depth': True,  # If false, output a ray displacement map, i.e. from the mesh surface to the camera center.
# })
# # Append more camera parameters if you want batch processing.

# # Load triangle mesh data. See python/resources/airplane/models/model_normalized.obj
# vertices = ...  # An array of shape (num_vertices, 3) and type np.float32.
# faces = ...  # An array of shape (num_faces, 3) and type np.uint32.

# depth_maps = m2d.mesh2depth(vertices, faces, params, empty_pixel_value=np.nan)

#Per now will use trimesh to create a depth map from a mesh object...

import numpy as np
import PIL.Image
import trimesh
import matplotlib.pyplot as plt
import time
from trimesh.ray.ray_pyembree import RayMeshIntersector
if __name__ == "__main__":

    # test on a simple mesh
    mesh = trimesh.primitives.Sphere()
    # trimesh.load("../models/featuretype.STL")

    intersector = RayMeshIntersector(mesh)

    # scene will have automatically generated camera and lights
    scene = mesh.scene()

    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [320, 240]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = 60 * (scene.camera.resolution / scene.camera.resolution.max())

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # do the actual ray- mesh queries
    _s = time.time()
    points, index_ray, index_tri = intersector.intersects_location(
        origins, vectors, multiple_hits=False
    )

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = (depth - depth.min()) / depth.ptp()

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    print(f"Depthimage made in {time.time() - _s:0.2f} seconds")

    #mpl visualization
    img = a
    img = np.rot90(img, 1)
    plt.imshow(img,cmap='viridis')
    plt.colorbar()
    plt.show()

    #PIL visualization
    # create a PIL image from the depth queries
    # img = PIL.Image.fromarray(a)

    # # show the resulting image
    # img.show()
    # create a raster render of the same scene using OpenGL
    # rendered = PIL.Image.open(trimesh.util.wrap_as_stream(scene.save_image()))