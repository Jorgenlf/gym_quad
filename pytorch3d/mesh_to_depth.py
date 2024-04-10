import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
import utils.plot_image_grid as plot_image_grid
from pytorch3d.renderer.mesh.textures import Textures
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


# Camera globals
IMG_SIZE = (240, 320)           # (H, W) of physical depth cam images AFTER the preprocessing pipeline
FOV = 60                        # Field of view in degrees, init to correct value later
MAX_MEASURABLE_DEPTH = 2.0     # Maximum measurable depth, initialized to k here but is 10 IRL

# Get GPU if available (it should)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}')

# Create mesh from .obj file (.obj links to .mtl file which links to texture file(s))
#obj_path = "./cow_mesh/cow.obj"
obj_paths = ["./room_mesh/model.obj"]
obj_paths = ["./sphere.obj"] # We do not need a .mtl file for mapping mesh to depth. If we wanna visualize we need mtl (i think)...

meshes = load_objs_as_meshes(obj_paths, device=device) # Used for multiple meshes originally, could use load_obj for single mesh (faster..?)

# Initialize camera
"""
For cameras, there are four different coordinate systems (or spaces)
- World coordinate system: This is the system the object lives - the world.
- Camera view coordinate system: This is the system that has its origin on the camera and the and the Z-axis perpendicular to the image plane. 
In PyTorch3D, it is assumed that +X points left, and +Y points up and +Z points out from the image plane. 
The transformation from world -> view happens after applying a rotation (R) and translation (T).
"""
 
# In the sim we plan to get the R,T matrices from the drone and use them to update the camera object at evry timestep (must handle transfomations and stuff properly)
R, T = look_at_view_transform(2, 10, 0)
camera = FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV)

# Initialize rasterizer
# Rasterization maps the 3D mesh to a 2D image for the given camera.
raster_settings = RasterizationSettings(
    image_size=IMG_SIZE, 
    blur_radius=0.0, 
    faces_per_pixel=1, # Keep at 1, dont change
)
rasterizer = MeshRasterizer(
    cameras=camera, 
    raster_settings=raster_settings
)

fragments = rasterizer(meshes)

# **zbuf**: FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving the NDC z-coordinates of the nearest faces at each pixel, sorted in ascending z-order.
zbuf = fragments.zbuf#.cpu().numpy() # (N, H, W, K), 
depth = torch.squeeze(zbuf) # Get only the depth values at each pixel. Shape: (H, W) when N = K = 1

# In the zbuf object, the value -1.0 is used to indicate that there was no valid face at that pixel, while the value 0.0 is used to indicate that the face closest to the camera is at the pixel.
# We set all -1 values to MAX_MEASURABLE_DEPTH as this is how such values are handled in the physical depth cam (could this be optimized with some raster settings or something?)
depth[depth == -1.0] = MAX_MEASURABLE_DEPTH

plt.figure()
plt.imshow(depth.cpu().numpy(), cmap="magma")
plt.colorbar()
plt.axis("off")
plt.savefig("./depth_map.png")

#prnit min and max depth
print(f"Min depth: {depth.min()}")
print(f"Max depth: {depth.max()}")


# Time how many full depth images can be rendered per second for given device
import time
start = time.time()
for i in range(100):

    # Testing if updating R and T for the camera object is tideous - it is not.
    R, T = look_at_view_transform(2, 10, 0)
    camera.T = T.to(device)
    camera.R = R.to(device)

    fragments = rasterizer(meshes)
    zbuf = fragments.zbuf
    depth = torch.squeeze(zbuf)
    depth[depth == -1.0] = MAX_MEASURABLE_DEPTH

end = time.time()
print(f"Time for 100 renders with {device}: {end-start} seconds")
print(f"FPS with {device}: {100/(end-start)}") # FPS: 