import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    PerspectiveCameras,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftSilhouetteShader,
    SoftPhongShader,
    Materials,
    TexturesVertex,
)

# Helper functions to transform between ENU and pytorch3D coordinate systems
def enu_to_pytorch3d(enu_position: torch.Tensor) -> torch.Tensor:
    '''ENU is x-east, y-north, z-up. 
    Pytorch3D is x-left, y-up, z-forward. 
    This function converts from ENU to Pytorch3D coordinate system.'''
    return torch.tensor([enu_position[1], enu_position[2], enu_position[0]])

def pytorch3d_to_enu(pytorch3d_position: torch.Tensor) -> torch.Tensor:
    '''ENU is x-east, y-north, z-up. 
    Pytorch3D is x-left, y-up, z-forward. 
    This function converts from ENU to Pytorch3D coordinate system.'''
    return torch.tensor([pytorch3d_position[2], pytorch3d_position[0], pytorch3d_position[1]])

def np_to_torch(np_array: np.array) -> torch.Tensor:
    return torch.tensor(np_array)

def torch_to_np(torch_tensor: torch.Tensor) -> np.array:
    return torch_tensor.cpu().numpy()

# Camera globals
IMG_SIZE = (240, 320)           # (H, W) of physical depth cam images AFTER the preprocessing pipeline
FOV = 60                        # Field of view in degrees, init to correct value later
MAX_MEASURABLE_DEPTH = 10.0     # Maximum measurable depth, initialized to k here but is 10 IRL

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
pos2 = torch.tensor([0.0, 3.0, 0.0])
pos3 = torch.tensor([0.0, 0.0, 1.5])
unit_sphere_path = "./sphere.obj"
sphere1 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.0, center_position=pos1)
sphere2 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.5, center_position=pos2)
sphere3 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=pos3)

spheres = [sphere1,sphere2,sphere3]
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
T = torch.zeros((1, 3)).to(device)
T[0, 0] = 0.0
T[0, 1] = 0.0
T[0, 2] = 6.0
print(f"R: {R} with shape {R.shape}")
print(f"T: {T} with shape {T.shape}")

camera = FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV,znear=0.1,zfar=10)

# Initialize rasterizer
# Rasterization maps the 3D mesh to a 2D image for the given camera.
raster_settings = RasterizationSettings(
    image_size=IMG_SIZE, 
    blur_radius=0.0, 
    faces_per_pixel=1, # Keep at 1, dont change
    perspective_correct=True
)

rasterizer = MeshRasterizer(
    cameras=camera, 
    raster_settings=raster_settings
)

#Generating n_steps positions and orientations for the camera in ENU frame
n_steps = 8
positions = np.zeros((n_steps, 3)) # x, y, z in meters
orientations = np.zeros((n_steps, 3)) # roll about x, pitch about y, yaw about z (ENU) in radians
for i in range (n_steps):
    param = n_steps/2
    positions[i] = np.array([6*np.cos(i/param*np.pi), 6*np.sin(i/param*np.pi), 0]) # x, y, z in meters 
    orientations[i] = np.array([0, 0, i/param*np.pi]) # roll, pitch, yaw in radians

actual_camera_positions_enu = np.zeros((n_steps, 3))
for i in range(n_steps):
    T_enu = np_to_torch(positions[i])
    Tstep = enu_to_pytorch3d(T_enu)
    Tstep = Tstep.unsqueeze(0)

    #Create R matrix from ENU euler angles in radians
    orientation_enu = np_to_torch(orientations[i])
    orientation_pt3d = enu_to_pytorch3d(orientation_enu) #want roll,pitch,yaw ->  pitch,yaw,roll
    Rstep = euler_angles_to_matrix(orientation_pt3d, convention="XYZ").unsqueeze(0) #TODO can probably use one convention which allows direct passage of enu angles ZXY maybe?

    camera.T = Tstep.to(device)
    camera.R = Rstep.to(device)

    cam_pos = camera.get_camera_center()
    print(f"Camera position: {cam_pos} with shape {cam_pos.shape}")
    actual_camera_positions_enu[i] = torch_to_np(pytorch3d_to_enu(cam_pos.reshape(3)))
    cam_orientation = camera.get_world_to_view_transform().get_matrix()    

    # print(f"\nRstep: {Rstep} with shape {Rstep.shape}")
    # print(f"Camera orientation: {cam_orientation}\n")

    #Render depth map
    rasterizer.cameras = camera
    fragments = rasterizer(meshes)

    zbuf = fragments.zbuf
    depth = torch.squeeze(zbuf) 

    depth[depth == -1.0] = MAX_MEASURABLE_DEPTH
    depth[depth >= MAX_MEASURABLE_DEPTH] = MAX_MEASURABLE_DEPTH

    plt.figure()
    plt.imshow(depth.cpu().numpy(), cmap="magma")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(f"./test_img/depth_maps/depth_map{i}.png")
    plt.close()

#Plot the reference and actual positions in enu frame
plt.figure()
plt.plot(positions[:,0], positions[:,1], 'ro', label='Reference camera positions')
for i in range(n_steps):
    plt.text(positions[i,0], positions[i,1], str(i))
    #heading direction
    plt.plot([positions[i,0], positions[i,0]+np.cos(orientations[i,2])], [positions[i,1], positions[i,1]+np.sin(orientations[i,2])], 'r-')
plt.plot(actual_camera_positions_enu[:,0], actual_camera_positions_enu[:,1], 'bo', label='Actual camera positions')
for i in range(n_steps):
    plt.text(actual_camera_positions_enu[i,0], actual_camera_positions_enu[i,1], str(i))
plt.xlabel('East X')
plt.ylabel('North Y')
plt.title('Reference and actual camera positions in ENU frame')
plt.grid()
plt.axis('equal')
plt.legend()
plt.savefig('./test_img/camera_positions.png')
plt.close()    

#Plot of only reference camera positions
plt.figure()
plt.plot(positions[:,0], positions[:,1], 'ro', label='Reference camera positions')
for i in range(n_steps):
    plt.text(positions[i,0], positions[i,1], str(i))
    plt.plot([positions[i,0], positions[i,0]+np.cos(orientations[i,2])], [positions[i,1], positions[i,1]+np.sin(orientations[i,2])], 'r-')
plt.xlabel('East X')
plt.ylabel('North Y')
plt.title('Reference camera positions in ENU frame')
plt.grid()
plt.axis('equal')
plt.legend()
plt.savefig('./test_img/ref_camera_positions.png')
plt.close()    

#Plot of only actual camera positions
plt.figure()
plt.plot(actual_camera_positions_enu[:,0], actual_camera_positions_enu[:,1], 'bo', label='Actual camera positions')
for i in range(n_steps):
    plt.text(actual_camera_positions_enu[i,0], actual_camera_positions_enu[i,1], str(i))
plt.xlabel('East X')
plt.ylabel('North Y')
plt.title('Reference and actual camera positions in ENU frame')
plt.grid()
plt.axis('equal')
plt.legend()
plt.savefig('./test_img/actual_camera_positions.png')
plt.close()    



# fragments = rasterizer(meshes)

# # **zbuf**: FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving the NDC z-coordinates of the nearest faces at each pixel, sorted in ascending z-order.
# zbuf = fragments.zbuf#.cpu().numpy() # (N, H, W, K), 
# depth = torch.squeeze(zbuf) # Get only the depth values at each pixel. Shape: (H, W) when N = K = 1

# # In the zbuf object, the value -1.0 is used to indicate that there was no valid face at that pixel, while the value 0.0 is used to indicate that the face closest to the camera is at the pixel.
# # We set all -1 values to MAX_MEASURABLE_DEPTH as this is how such values are handled in the physical depth cam (could this be optimized with some raster settings or something?)
# depth[depth == -1.0] = MAX_MEASURABLE_DEPTH
# # For measurements beyond the max measurable depth, the depth value is also set to MAX_MEASURABLE_DEPTH
# depth[depth >= MAX_MEASURABLE_DEPTH] = MAX_MEASURABLE_DEPTH

# plt.figure()
# plt.imshow(depth.cpu().numpy(), cmap="magma")
# plt.colorbar()
# plt.axis("off")
# plt.savefig("./depth_map.png")

# #print min and max depth
# print(f"Min depth: {depth.min()}")
# print(f"Max depth: {depth.max()}")


# # Time how many full depth images can be rendered per second for given device
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