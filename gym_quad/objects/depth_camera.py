import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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

import gym_quad.utils.geomutils as geom


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


class SphereMeshObstacle:
    def __init__(self, 
                 device: torch.device, 
                 path: str,
                 radius: float,
                 center_position: torch.Tensor):
        self.device = device
        self.path = path                                    # Assumes path points to UNIT sphere .obj file
        self.radius = radius

        self.center_position = center_position.to(device=self.device)   # Centre of the sphere in camera world frame
        #Not specified in name to simplify rewriting of code

        self.position = pytorch3d_to_enu(center_position).to(device=self.device) # Centre of the sphere in ENU frame
        #Not specified in name to simplify rewriting of code

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

    def return_plot_variables(self):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.position[0] + self.radius*np.cos(u)*np.sin(v)
        y = self.position[1] + self.radius*np.sin(u)*np.sin(v)
        z = self.position[2] + self.radius*np.cos(v)
        return [x,y,z]
    

class SphereScene:
    def __init__(self, 
                 device: torch.device, 
                 sphere_obstacles: list):
        self.device = device
        self.sphere_obstacles = sphere_obstacles # List of SphereMeshObstacle objects
        self.meshes = [sphere.mesh for sphere in sphere_obstacles]
        self.joined_scene = self.update_scene() # Textures not included by default
    
    def update_scene(self, include_textures: bool = False):
        return join_meshes_as_scene(meshes=self.meshes, include_textures=include_textures)

    def resize_sphere(self, sphere_idx: int, new_radius: float):
        self.sphere_obstacles[sphere_idx].resize(new_radius)
    
    def move_sphere(self, sphere_idx: int, new_center_position: torch.Tensor):
        self.sphere_obstacles[sphere_idx].move(new_center_position)
    
    def add_sphere(self, new_sphere: SphereMeshObstacle):
        self.sphere_obstacles.append(new_sphere)
        self.meshes.append(new_sphere.mesh)
        self.joined_scene = join_meshes_as_scene(meshes=self.meshes, include_textures=False)
    
    def remove_sphere(self, sphere_idx: int):
        self.sphere_obstacles.pop(sphere_idx)
        self.meshes.pop(sphere_idx)
        self.joined_scene = join_meshes_as_scene(meshes=self.meshes, include_textures=False)
    
    def set_device(self, new_device: torch.device):
        self.device = new_device
        for sphere in self.sphere_obstacles:
            sphere.set_device(new_device)


class DepthMapRenderer:
    def __init__(self, 
                 device: torch.device, 
                 scene: SphereScene, 
                 camera: FoVPerspectiveCameras, 
                 raster_settings: RasterizationSettings,
                 MAX_MEASURABLE_DEPTH: float = 10.0,
                 img_size: tuple = (240, 320)):
        self.device = device
        self.scene = scene
        self.camera = camera
        self.raster_settings = raster_settings
        self.max_measurable_depth = MAX_MEASURABLE_DEPTH
        self.rasterizer = MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        )
        # k is a scaling factor for the distortion correction and is a function of FOV, sensor size, and focal length, etc.
        # Initilized to 62.5 for the default FoVPerspectiveCamera settings with a 60 degree FOV and image size of 240x320
        # Initilized to 46.6 for the default FoVPerspectiveCamera settings with a 75 degree FOV and image size of 240x320

        self.k = 46.6
        self.img_size = img_size
    
    def render_depth_map(self):
        """
        Renders a depth map of the scene seen from the current camera position and orientation by performing rasterization 
        and then correcting for perspective distortions. Also performs saturation of infinite and NaN depth values.
        """
        # Render depth map via rasterization
        fragments = self.rasterizer(self.scene.joined_scene)
        zbuf = fragments.zbuf
        depth = torch.squeeze(zbuf).to(self.device)

        # Saturate infinite and NaN depth values
        depth[depth == -1.0] = self.max_measurable_depth
        depth[depth >= self.max_measurable_depth] = self.max_measurable_depth

        # Correct for perspective distortion
        depth = self.correct_distorion(depth)
        return depth

    def correct_distorion(self, depth: torch.Tensor):
        """
        Corrects for perspective distortion in the depth map by calculating the depth values to the camera center
        instead of to the image plane for each pixel in the image.
        """
        # Create grids of x and y coordinates
        x_grid, y_grid = torch.meshgrid(torch.arange(self.img_size[0], device=self.device), torch.arange(self.img_size[1], device=self.device), indexing='ij')

        # Compute distance from center for each pixel
        center = torch.tensor(self.img_size, device=self.device)/2
        dist_from_center = torch.norm(torch.stack([x_grid, y_grid], dim=-1) - center, dim=-1)

        # Amplify dist tensor by 1/k
        dist_from_center = dist_from_center / self.k

        # Correct for perspective distortion at indices where depth is not infinite
        depth = torch.where(depth < self.max_measurable_depth, torch.sqrt(torch.pow(depth,2) + torch.pow(dist_from_center,2)), depth).to(self.device)
        depth[depth >= self.max_measurable_depth] = self.max_measurable_depth
        return depth
    
    def render_scene(self, light_location=(5, 5, 0)):
        """
        Renders the scene from the current camera position and orientation with the given point light location.
        Returns the rendered image (not the depth!)
        """
        # Recreate scene with textures
        textured_scene = self.scene.update_scene(include_textures=True)
        # Basic rendering setup with point light and soft phong shader
        light_location = (light_location,)
        lights = PointLights(device=self.device, location=light_location)
        renderer_rgb = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=SoftPhongShader(
                device=self.device, 
                cameras=self.camera,
                lights = lights
            )
        )
        img = renderer_rgb(textured_scene)
        return img
    
    # Function to find the R and T matrices for the camera object in Pytorch3D
    def camera_R_T_from_quad_pos_orient(self, position: np.array, orientation: np.array) -> tuple:
        '''Given a position and orientation of the quadrotor in ENU frame, 
        this function returns the R and T matrices for the camera object in Pytorch3D.
        The camera is assumed to be looking at a point in front of the quadrotor along the body x-axis.'''
        # Convert position and orientation to torch tensors
        at = position + geom.Rzyx(*orientation) @ np.array([1, 0, 0])  # Look at the point in front of the camera along body x-axis
        
        at_torch = torch.from_numpy(at).to(self.device)
        position_torch = torch.from_numpy(position).to(self.device)

        at_pt3d = enu_to_pytorch3d(at_torch).to(self.device).float()
        position_pt3d = enu_to_pytorch3d(position_torch).to(self.device).float()
        # orientation_torch = torch.from_numpy(orientation).to(device)
        
        # Calculate rotation matrix
        Rstep = look_at_rotation(position_pt3d[None, :], device=self.device, at=at_pt3d[None, :])  # (1, 3, 3)
        # Calculate translation vector
        Tstep = -torch.bmm(Rstep.transpose(1, 2), position_pt3d[None, :, None])[:, :, 0]   # (1, 3)
        
        return Rstep, Tstep


    def set_device(self, new_device: torch.device):
        self.device = new_device
        self.scene.set_device(new_device)
        self.camera.to(new_device)
    
    def update_T(self, new_T: torch.Tensor):
        new_T = new_T.to(self.device)
        if new_T.shape != (1, 3):
            raise ValueError("T must be a 1x3 tensor")
        self.camera.T = new_T
    
    def update_R(self, new_R: torch.Tensor):
        if new_R.shape == (3, 3):
            new_R = new_R.unsqueeze(0).to(self.device)
        self.camera.R = new_R

    def save_depth_map(self, path: str, depth: torch.Tensor = None):
        if depth is None:
            depth = self.render_depth_map()
        
        plt.style.use('ggplot')
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('axes', labelsize=12)

        plt.figure(figsize=(8, 6))
        plt.imshow(depth.cpu().numpy(), cmap="magma")
        plt.clim(0.0, 10.0)
        plt.colorbar(label="Depth [m]", aspect=30, orientation="vertical", fraction=0.0235, pad=0.04)
        plt.axis("off")
        plt.savefig(path, bbox_inches='tight')
    
    def save_rendered_scene(self, path:str):
        img = self.render_scene()

        plt.style.use('ggplot')
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('axes', labelsize=12)

        plt.figure(figsize=(8, 6))
        plt.imshow(img[0, ..., :3].cpu().numpy())
        plt.axis("off")
        plt.savefig(path, bbox_inches='tight')

if __name__ == "__main__":

    # Camera globals
    IMG_SIZE = (240, 320)           # (H, W) of physical depth cam images AFTER the preprocessing pipeline
    FOV = 60                        # Field of view in degrees, init to correct value later
    MAX_MEASURABLE_DEPTH = 10.0      # Maximum measurable depth, initialized to k here but is 10 IRL
    
    #init device to use gpu if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    #init camera
    camera = FoVPerspectiveCameras(device=device, fov=FOV)

    #init scene
    unit_sphere_path = "gym_quad/meshes/sphere.obj"
    obs1 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.0, center_position=torch.tensor([4, 0, 8]))
    obs2 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=4.0, center_position=torch.tensor([2, 4, 5]))
    obs3 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.3, center_position=torch.tensor([-4, 0, 12]))
    obs4 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.1, center_position=torch.tensor([3, 0, 15]))
    obs5 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=3.2, center_position=torch.tensor([0, -4, 17]))
    obs6 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.3, center_position=torch.tensor([-2.7, 0, 18]))
    obs7 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=torch.tensor([-0, 0, 26]))

    spherescene = SphereScene(device=device, sphere_obstacles=[obs1, obs2, obs3, obs4, obs5, obs6, obs7])

    #Init rasterizer
    raster_settings = RasterizationSettings(
    image_size=IMG_SIZE, 
    blur_radius=0.0, 
    faces_per_pixel=1, # Keep at 1, dont change
    perspective_correct=True, # Doesn't do anything(??), but seems to improve speed
    cull_backfaces=True # Do not render backfaces. MAKE SURE THIS IS OK WITH THE GIVEN MESH.
    )

    renderer = DepthMapRenderer(device=device,camera=camera,scene =spherescene, raster_settings=raster_settings, MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH, img_size=IMG_SIZE)

    #Generating n_steps positions and orientations for the camera in ENU frame
    n_steps = 24
    positions = np.zeros((n_steps, 3)) # x, y, z in meters
    orientations = np.zeros((n_steps, 3)) # roll about x, pitch about y, yaw about z (ENU) in radians

    referencetype = 'line' # 'circle', 'line', 

    for i in range (n_steps):
        param = n_steps/2
        #Circle around origin with radius 6
        if referencetype == 'circle':
            positions[i] = np.array([6*np.cos(i/param*np.pi), 6*np.sin(i/param*np.pi), 0]) # x, y, z in meters 
            orientations[i] = np.array([0, 0, np.pi + i/param*np.pi]) # roll, pitch, yaw in radians
        elif referencetype == 'line':
            positions[i] = np.array([i, 0, 0])
            orientations[i] = np.array([0, 0, 0])

    #NB SAVES THEM TO THE TEST_IMG FOLDER IN THE TESTS FOLDER
    path_to_save_depth_maps = os.path.join(grand_parent_dir, "gym_quad/tests/test_img/depth_maps/")

    for i in tqdm(range(n_steps)):
        position = positions[i]
        orientation = orientations[i]

        R, T = renderer.camera_R_T_from_quad_pos_orient(position, orientation)
        renderer.update_R(R)
        renderer.update_T(T)
        depth_map = renderer.render_depth_map()
        renderer.save_depth_map(path_to_save_depth_maps+f"depth_map{i}.png", depth_map)
        