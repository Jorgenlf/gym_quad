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
FOV = 75                        # Field of view in degrees, init to correct value later
MAX_MEASURABLE_DEPTH = 5.0      # Maximum measurable depth, initialized to k here but is 10 IRL

# Get GPU if available (it should)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}')


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
        x_grid, y_grid = torch.meshgrid(torch.arange(self.img_size[0], device=device), torch.arange(self.img_size[1], device=self.device), indexing='ij')

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
        lights = PointLights(device=device, location=light_location)
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
            new_R = new_R.unsqueeze(0).to(device)
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

#unit_sphere_path = "./unit_sphere_mesh/unit_sphere_lower_poly.obj"
unit_sphere_path = "./unit_sphere_mesh/unit_sphere.obj"

pos1 = torch.tensor([0.0, 0.0, 0.0])
pos2 = torch.tensor([2.0, 0.0, 0.0])
pos3 = torch.tensor([0.0, -2.0, 0.0])
pos4 = torch.tensor([2.0, 0.0, 0.0])
pos5 = torch.tensor([2.0, 2.0, 0.0])
pos6 = torch.tensor([-2.0, -2.0, 0.0])
pos7 = torch.tensor([-2.0, 0.0, 0.0])

sphere1 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=pos1)
sphere2 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=pos2)
sphere3 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=pos3)
sphere4 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=pos4)
sphere5 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.0, center_position=pos5)
sphere6 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.8, center_position=pos6)
sphere7 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.5, center_position=pos7)

spheres = [sphere1, sphere2, sphere3, sphere4, sphere5, sphere6, sphere7]
spherescene = SphereScene(device=device, sphere_obstacles=spheres)
meshes = spherescene.joined_scene


R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0).to(device)
T = torch.zeros((1, 3)).to(device)
T[0, 0] = 0.0 # Move camera x units in the +X direction (inwards)
T[0, 1] = 0.0 # Move camera x units in the +Y direction (upwards)
T[0, 2] = 1.5 # Move camera x units in the +Z direction (in/out)
#T = -T

camera = FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV)
#zfar_screen = camera.transform_points_screen(torch.tensor([[0.0, 0.0, MAX_MEASURABLE_DEPTH]]).to(device))[0, 2].item()
#print(f"zfar_screen: {zfar_screen}")
#camera = FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV, zfar=zfar_ndc)


# Initialize rasterizer
raster_settings = RasterizationSettings(
    image_size=IMG_SIZE, 
    blur_radius=0.0, 
    faces_per_pixel=1, # Keep at 1, dont change
    perspective_correct=True, # Doesn't do anything(??), but seems to improve speed
    cull_backfaces=True # Do not render backfaces. MAKE SURE THIS IS OK WITH THE GIVEN MESH.
)

depth_renderer = DepthMapRenderer(device=device, 
                            scene=spherescene, 
                            camera=camera, 
                            raster_settings=raster_settings, 
                            MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH,
                            img_size=IMG_SIZE)


# ks = [33.3, 45, 46, 46.5, 46.7, 47, 48, 49, 50, 58.5, 62.5, 1000]
# for k in ks:
#     depth_renderer.k = k
#     depth = depth_renderer.render_depth_map()
#     # Black region of blobs at [0,2,0] etc. should be as large as possible without falling apart
#     delta = 0.1
#     lower_mask = torch.where(depth > 4.385 - delta, 1, 0)
#     upper_mask = torch.where(depth < 4.385 + delta, 1, 0)
#     mask = lower_mask * upper_mask
#     mask = 1 - mask
#     depth = depth * mask
#     depth_renderer.save_depth_map(f"./depth_map_k{k}.pdf", depth)



depth = depth_renderer.render_depth_map()
depth_renderer.save_depth_map("./depth_map.pdf", depth)
depth_renderer.save_rendered_scene("./rendered_scene.pdf")


print(f"R:\n {R}")
print(f"\nT:\n {T}")
print(f'\nCamera position in world frame:\n {camera.get_camera_center()}')
print(f"\nMin depth: {depth.min()}")
print(f"\nMax depth: {depth.max()}")

# #Time the rendering process fort 100 renders
import time
# start = time.time()
# n = 10000
# for i in range(n):
#     depth = depth_renderer.render_depth_map()
# end = time.time()
# print(f"Time taken for {n} renders: {end-start} seconds")
# print(f"FPS: {n/(end-start)} seconds")

# Create a plot of time taken per render
# import matplotlib.pyplot as plt
# import numpy as np
# times = []
# n = 1000
# for i in range(n):
#     start = time.time()
#     depth = depth_renderer.render_depth_map()
#     end = time.time()
#     times.append(end-start)

# plt.clf()
# plt.plot(np.arange(n), times)
# plt.xlabel("Render number")
# plt.ylabel("Time taken (s)")
# plt.savefig("./render_times.pdf", bbox_inches='tight')

# #plot moving average of times with a window of 10
# times = np.convolve(times, np.ones(50), 'valid') / 50
# plt.clf()
# plt.plot(times)
# plt.xlabel("Render number")
# plt.ylabel("Time taken (s)")
# plt.savefig("./render_times_moving_avg.pdf", bbox_inches='tight')

# print(f'avg time: {np.mean(times)}')