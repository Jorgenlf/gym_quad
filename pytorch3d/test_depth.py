import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftSilhouetteShader,
    SoftPhongShader,
    Materials,
    TexturesVertex,
)


import utils.plot_image_grid as plot_image_grid
from pytorch3d.renderer.mesh.textures import Textures

# Camera globals
IMG_SIZE = (480, 640)       # (H, W) of physical depth cam images
FOV = 60                    # Field of view in degrees, init to correct value later
MAX_MEASURABLE_DEPTH = 4.0  # Maximum measurable depth in meters, initialized to 4 here but is 10 IRL

devices = [torch.device("cuda"), torch.device("cpu")]
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print(f'Using device: {device}')

for device in devices:
    print(f'Using device: {device} for cpu vs. gpu test')

    # Create mesh from .obj file (obj links to mtl file which links to texture file)
    obj_path = "./cow_mesh/cow.obj"
    meshes = load_objs_as_meshes([obj_path], device=device)

    R, T = look_at_view_transform(2, 1, 180) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV, zfar=MAX_MEASURABLE_DEPTH)

    raster_settings = RasterizationSettings(
        image_size=IMG_SIZE, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )

    fragments = rasterizer(meshes)

    zbuf = fragments.zbuf#.cpu().numpy() # (N, H, W, K), **zbuf**: FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving the NDC z-coordinates of the nearest faces at each pixel, sorted in ascending z-order.
    depth = torch.squeeze(zbuf)
    depth[depth == -1.0] = MAX_MEASURABLE_DEPTH
    print(f"Depth map shape: {depth.shape}")

    plt.figure()
    plt.imshow(depth.cpu().numpy(), cmap="magma")
    plt.colorbar()
    plt.axis("off")
    plt.savefig("./cow_depth.png")

    #prnit min and max depth
    print(f"Min depth: {zbuf.squeeze().min()}")
    print(f"Max depth: {zbuf.squeeze().max()}")

    # In the zbuf object, the value -1.0 is used to indicate that there was no valid face at that pixel, while the value 0.0 is used to indicate that the face closest to the camera is at the pixel.
    # We set all -1 values to MAX_MEASURABLE_DEPTH as this is how such values are handled in the physical depth cam


    # Time how many full depth images can be rendered per second for given device
    import time
    start = time.time()
    for i in range(100):
        fragments = rasterizer(meshes)
        zbuf = fragments.zbuf
        depth = torch.squeeze(zbuf)
        depth[depth == -1.0] = MAX_MEASURABLE_DEPTH
    end = time.time()
    print(f"Time for 100 renders with {device}: {end-start} seconds")
    print(f"FPS with {device}: {100/(end-start)}") # FPS: 198.45497392947712






