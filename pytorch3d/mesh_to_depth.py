import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
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
MAX_MEASURABLE_DEPTH = 5.0     # Maximum measurable depth, initialized to k here but is 10 IRL

# Get GPU if available (it should)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}')

# Create mesh from .obj file (.obj links to .mtl file which links to texture file(s))
#obj_path = "./cow_mesh/cow.obj"
obj_paths = ["./room_mesh/model.obj"]
obj_paths = ["./sphere.obj"] # We do not need a .mtl file for mapping mesh to depth. If we wanna visualize we need mtl (i think)...

class SphereMeshObstacle:
    def __init__(self, 
                 device: torch.device, 
                 path: str,
                 radius: float,
                 center_position: torch.Tensor):
        self.device = device
        self.path = path                                    # Assumes path points to UNIT sphere .obj file
        self.radius = radius
        self.center_position = center_position.to(device)   # Centre of the sphere in world frame
        # May include textures later for visualization in non-depth mode

        self.mesh = load_objs_as_meshes([path], device=self.device)
        self.mesh.scale_verts_(scale=self.radius)
        self.mesh.offset_verts_(vert_offsets_packed=self.center_position)
    
    def resize(self, new_radius: float):
        self.mesh.scale_verts_(scale=new_radius/self.radius)
        self.radius = new_radius
    
    def move(self, new_center_position: torch.Tensor):
        new_center_position = new_center_position.to(self.device)
        self.mesh.offset_verts_(vert_offsets_packed=new_center_position-self.center_position)
        self.center_position = new_center_position

    def set_device(self, new_device: torch.device):
        self.device = new_device
        self.mesh.to(new_device)
        self.center_position.to(new_device)
    

class SphereScene:
    def __init__(self, 
                 device: torch.device, 
                 sphere_obstacles: list):
        self.device = device
        self.sphere_obstacles = sphere_obstacles # List of SphereMeshObstacle objects
        self.meshes = [sphere.mesh for sphere in sphere_obstacles]
        self.scene = join_meshes_as_scene(meshes=self.meshes, include_textures=False) # May inlude textures later

    def resize_sphere(self, sphere_idx: int, new_radius: float):
        self.sphere_obstacles[sphere_idx].resize(new_radius)
    
    def move_sphere(self, sphere_idx: int, new_center_position: torch.Tensor):
        self.sphere_obstacles[sphere_idx].move(new_center_position)
    
    def add_sphere(self, new_sphere: SphereMeshObstacle):
        self.sphere_obstacles.append(new_sphere)
        self.meshes.append(new_sphere.mesh)
        self.scene = join_meshes_as_scene(meshes=self.meshes, include_textures=False)
    
    def remove_sphere(self, sphere_idx: int):
        self.sphere_obstacles.pop(sphere_idx)
        self.meshes.pop(sphere_idx)
        self.scene = join_meshes_as_scene(meshes=self.meshes, include_textures=False)
    
    
    def set_device(self, new_device: torch.device):
        self.device = new_device
        for sphere in self.sphere_obstacles:
            sphere.set_device(new_device)
    

    
# Test sphere obstacle and scene
# For the position vectors, +X points left, and +Y points up and +Z points inwards
pos1 = torch.tensor([0.0, 0.0, 0.0])
pos2 = torch.tensor([0.0, 2.0, 0.0])
#pos3 = torch.tensor([4.0, 0.0, 1.5])
unit_sphere_path = "./sphere.obj"
sphere1 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=pos1)
sphere2 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=pos2)
#sphere3 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=4.0, center_position=pos3)

# Sphere 3 is only partly visible, as it is at the max measurable depth

spheres = [sphere1, sphere2]#, sphere3]
spherescene = SphereScene(device=device, sphere_obstacles=spheres)
meshes = spherescene.scene


# Initialize camera
"""
For cameras, there are four different coordinate systems (or spaces)
- World coordinate system: This is the system the object lives - the world.
- Camera view coordinate system: This is the system that has its origin on the camera and the and the Z-axis perpendicular to the image plane. 
In PyTorch3D, it is assumed that +X points left, and +Y points up and +Z points out from the image plane. 
The transformation from world -> view happens after applying a rotation (R) and translation (T).
"""
 
# In the sim we plan to get the R,T matrices from the drone and use them to update the camera object at evry timestep (must handle transfomations and stuff properly)
# For now we just initialize the camera to no rotation, looking at the origin from a distance of 10 units in the +Z direction
R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0).to(device)
print(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).shape)
print(R.shape)
T = torch.zeros((1, 3)).to(device)
T[0, 0] = 0.0 # Move camera x units in the +X direction (inwards)
T[0, 1] = 0.0 # Move camera x units in the +Y direction (upwards)
T[0, 2] = -5.0 # Move camera x units in the +Z direction (in/out)
#T = -T
print(f"R:\n {R}")
print(f"\nT orig:\n {T}")

# The rotation matrix R rotates the whole camera about the world origin, not around the camera frame axes. 
# The translation matrix T moves the camera in the world coordinate system.

#K = torch.zeros((1, 4, 4)).to(device)
#K[0, :, :] = torch.tensor([[639.0849609375, 0.0, 644.4653930664062],[0.0, 639.0849609375, 364.2340393066406], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
#camera = FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV)

#R1 = look_at_rotation(camera_position=camera.get_camera_center(), at = pos1)[0].unsqueeze(0).to(device)
#R1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).unsqueeze(0).to(device)
#print(f"\nRw_to_v from look_at:\n {R1}")    
#camera.R = R1
camera = FoVPerspectiveCameras(device=device, R=R, T=-T, fov=FOV)
# camera1 = FoVPerspectiveCameras(device=device, R=R1, T=T, fov=FOV)
print(f'Camera position in world frame: {camera.get_camera_center()}')





# Render the plotly figure
#fig = plot_scene({
#        "subplot1_title": {
#            "mesh_trace_title": meshes
#        },
#    }, viewpoint_cameras=camera) 
#fig.show()


# pos = camera.get_camera_center()
# print(f"\nCamera position:\n {pos}")
# camera_orientation = camera.get_world_to_view_transform().inverse().get_matrix()[:, :3].cpu().numpy()
# print(f"\nCamera orientation:\n {camera_orientation}")
# print(f'wordl to view:\n {camera.get_world_to_view_transform().get_matrix()}')
# print(f'full proj transform:\n {camera.get_full_projection_transform().get_matrix()}')

# wtv = camera.get_world_to_view_transform().get_matrix()
# print(f"\nWorld to view transform:\n {wtv}")
# vtw = wtv.inverse()
# print(f"\nView to world transform:\n {vtw}")

# # translate camera in world space
# T = T + torch.tensor([[1.0, 0.0, 0.0]]).to(device)
# camera.T = -T

# print(f"\nCamera position after translation:\n {camera.get_camera_center()}")

# pos = camera.get_camera_center()
# print(f"\nCamera position:\n {pos}")
# camera_orientation = camera.get_world_to_view_transform().inverse().get_matrix()[:, :3].cpu().numpy()
# print(f"\nCamera orientation:\n {camera_orientation}")
# print(f'wordl to view:\n {camera.get_world_to_view_transform().get_matrix()}')
# print(f'full proj transform:\n {camera.get_full_projection_transform().get_matrix()}')

# wtv = camera.get_world_to_view_transform().get_matrix()
# print(f"\nWorld to view transform:\n {wtv}")
# vtw = wtv.inverse()
# print(f"\nView to world transform:\n {vtw}")


# Initialize rasterizer
# Rasterization maps the 3D mesh to a 2D image for the given camera.
raster_settings = RasterizationSettings(
    image_size=IMG_SIZE, 
    blur_radius=0.0, 
    faces_per_pixel=1, # Keep at 1, dont change
    #perspective_correct=True
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
# For measurements beyond the max measurable depth, the depth value is also set to MAX_MEASURABLE_DEPTH
depth[depth >= MAX_MEASURABLE_DEPTH] = MAX_MEASURABLE_DEPTH

plt.figure()
plt.imshow(depth.cpu().numpy(), cmap="magma")
plt.colorbar()
plt.axis("off")
plt.savefig("./depth_map.png")

print(f"\nMin depth: {depth.min()}")
print(f"Max depth: {depth.max()}")


# Time how many full depth images can be rendered per second for given device
# import time
# start = time.time()
# for i in range(100):

#     # Testing if updating R and T for the camera object is tideous - it is not.
#     R, T = look_at_view_transform(2, 10, 0)
#     camera.T = T.to(device)
#     camera.R = R.to(device)

#     fragments = rasterizer(meshes)
#     zbuf = fragments.zbuf
#     depth = torch.squeeze(zbuf)
#     depth[depth == -1.0] = MAX_MEASURABLE_DEPTH

# end = time.time()
# print(f"Time for 100 renders with {device}: {end-start} seconds")
# print(f"FPS with {device}: {100/(end-start)}") # FPS: 