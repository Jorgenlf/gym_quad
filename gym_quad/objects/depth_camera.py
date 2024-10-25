import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene, Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_rotation,
    PerspectiveCameras,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
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

from gym_quad.utils.geomutils import enu_to_pytorch3d, Rzyx, enu_to_tri
from gym_quad.objects.mesh_obstacles import Scene, SphereMeshObstacle, CubeMeshObstacle, ImportedMeshObstacle, get_scene_bounds
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints



class DepthMapRenderer:
    def __init__(self,
                 device: torch.device,
                 scene:  Scene,
                 camera: FoVPerspectiveCameras,
                 #camera: PerspectiveCameras,
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
        depth.masked_fill_(depth == -1.0, self.max_measurable_depth)
        depth.clamp_(max=self.max_measurable_depth)

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
    def camera_R_T_from_quad_pos_orient(self, position: np.array, orientation: np.array, look_direction:np.array = np.array([1, 0, 0]) ) -> tuple:
        '''
        Input:
        position: np.array of shape (3,) - x, y, z in meters
        orientation: np.array of shape (3,) - roll, pitch, yaw in radians
        look_direction: np.array of shape (3,) - x, y, z in meters where to look at from the quads position
        deaults to [1, 0, 0] which means that the camera is looking along the the x-axis of the quad

        Output:
        Rstep: torch.Tensor of shape (1, 3, 3) - Rotation matrix for the camera object in Pytorch3D
        Tstep: torch.Tensor of shape (1, 3) - Translation vector for the camera object in Pytorch3D

        Given a position and orientation of the quadrotor in ENU frame,
        this function returns the R and T matrices for the camera object in Pytorch3D.
        The camera is assumed to be looking at a point in front of the quadrotor along the body x-axis.'''
        # Convert position and orientation to torch tensors
        at = position + Rzyx(*orientation) @ look_direction  # Look at the point in front of the camera along body x-axis

        at_torch = torch.from_numpy(at).to(self.device)
        position_torch = torch.from_numpy(position).to(self.device)

        at_pt3d = enu_to_pytorch3d(at_torch,device=self.device).float()
        position_pt3d = enu_to_pytorch3d(position_torch,device=self.device).float()
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
        plt.clim(0.0, self.max_measurable_depth)
        plt.colorbar(label="Depth [m]", aspect=30, orientation="vertical", fraction=0.0235, pad=0.04)
        plt.axis("off")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

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
    FOV = 75                        # Field of view in degrees, init to correct value later
    MAX_MEASURABLE_DEPTH = 10.0      # Maximum measurable depth, initialized to k here but is 10 IRL

    #init device to use gpu if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #init camera
    camera = FoVPerspectiveCameras(device=device, fov=FOV, znear=0.1, zfar=MAX_MEASURABLE_DEPTH)
    print(camera.get_projection_transform().get_matrix().cpu().numpy())
    K = camera.get_projection_transform().get_matrix()

    # focal_length = (-0.5*IMG_SIZE[1]/np.tan(FOV/2), )
    # principal_point = ((IMG_SIZE[1] / 2, IMG_SIZE[0] / 2),) # Assuming a perfect camera
    # print(focal_length)
    # img_size = (IMG_SIZE,)
    # is_ndc = False
    #camera = PerspectiveCameras(K=K, device=device, in_ndc=True)

    print(camera.get_projection_transform())

    #obstacle creation
    unit_sphere_path = "gym_quad/meshes/sphere.obj"
    obs1 = SphereMeshObstacle(device, unit_sphere_path, 2.0, torch.tensor([4, 0, 8]))
    obs2 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=4.0, center_position=torch.tensor([2, 4, 5]))
    obs3 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.3, center_position=torch.tensor([-4, 0, 12]))
    obs4 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.1, center_position=torch.tensor([3, 0, 15]))
    obs5 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=3.2, center_position=torch.tensor([0, -4, 17]))
    obs6 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=2.3, center_position=torch.tensor([-2.7, 0, 18]))
    obs7 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=torch.tensor([-0, 0, 20 ]))

    #Test of returning plot variables
    # x,y,z = obs1.return_plot_variables()

    #init of scene
    spherescene = Scene(device=device, obstacles=[obs1, obs2, obs3, obs4, obs5, obs6, obs7])
    spherescene = Scene(device=device, obstacles=[obs1])

    #Init rasterizer
    raster_settings = RasterizationSettings(
    image_size=IMG_SIZE, 
    blur_radius=0.0,
    faces_per_pixel=1, # Keep at 1, dont change
    perspective_correct=True, # Doesn't do anything(??), but seems to improve speed
    cull_backfaces=True # Do not render backfaces. MAKE SURE THIS IS OK WITH THE GIVEN MESH.
    )

    sphere_renderer = DepthMapRenderer(device=device,camera=camera,scene =spherescene, raster_settings=raster_settings, MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH, img_size=IMG_SIZE)

    #Generating n_steps positions and orientations for the camera in ENU frame
    #This will also decide how many depthmaps get saved
    n_steps = 24
    positions = np.zeros((n_steps, 3)) # x, y, z in meters
    orientations = np.zeros((n_steps, 3)) # roll about x, pitch about y, yaw about z (ENU) in radians


    ####Change this to visualize different scenes and movement of the camera
    referencetype = 'enclosed_spin'
    referencetype = 'line'
    # 'line' - Camera moves in a line along the x-axis
    # 'circle' - Camera moves in a circle around the origin
    # 'spin' - Camera spins about its z axis (ENU) with position equal to the the origin
    # 'enclosed_spin' - Camera spins about its z axis (ENU) with position equal to the the origin and is enclosed by a cube made based on path and other obstacles
    # 'enclosed_circle' - Camera moves in a circle around the origin and is enclosed by a cube made based on path and other obstacles
    # 'spin_in_house' - Camera spins about its z axis (ENU) with position equal to the the origin and is enclosed by a house mesh
    ####

    #Create movement for the camera
    for i in range (n_steps):
        param = n_steps/2
        #Circle around origin with radius 6
        if referencetype == 'line': #Uses the sphere_renderer already created above
            positions[i] = np.array([i, 0, 0])
            orientations[i] = np.array([0, 0, 0])
        elif referencetype == 'circle':
            positions[i] = np.array([6*np.cos(i/param*np.pi), 6*np.sin(i/param*np.pi), 0]) # x, y, z in meters
            orientations[i] = np.array([0, 0, np.pi + i/param*np.pi]) # roll, pitch, yaw in radians
        elif referencetype == 'spin':
            positions[i] = np.array([0.01, 0, 0]) #OFFSET HERE WHEN CUBE USED TO AVOID SEEING THROUGH THE CORNER OF THE CUBE
            orientations[i] = np.array([0, 0, i/param*np.pi])
        elif referencetype == 'spin_in_house':
            positions[i] = np.array([(-1.96102, 0.695626,  2.45523)])
            orientations[i] = np.array([0, 0, i/param*np.pi])
        elif referencetype == 'enclosed_spin':
            positions[i] = np.array([0.01, 0, 0.05])
            orientations[i] = np.array([0, 0, i/param*np.pi])
        elif referencetype == 'enclosed_circle':
            positions[i] = np.array([6*np.cos(i/param*np.pi), 6*np.sin(i/param*np.pi), 0]) # x, y, z in meters pytorch3d coords
            orientations[i] = np.array([0, 0, np.pi + i/param*np.pi]) # roll, pitch, yaw in radians

    circle_renderer = None
    spin_renderer = None
    unit_cube_path = "gym_quad/meshes/cube.obj"

    #Create the scene for the different reference types
    if referencetype == 'circle':
        obs1 = CubeMeshObstacle(device, unit_cube_path, 5.0, torch.tensor([0, 0, 0]))
        # obs2 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.0, center_position=torch.tensor([5, 0, 0]))
        circle_scene = Scene(device=device, obstacles=[obs1])
        circle_renderer = DepthMapRenderer(device=device,camera=camera, scene=circle_scene, raster_settings=raster_settings, MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH, img_size=IMG_SIZE)
    elif referencetype == 'spin':
        obs1 = CubeMeshObstacle(device, unit_cube_path, 8.0, torch.tensor([0, 0, 0]))

        spin_scene = Scene(device=device, obstacles=[obs1])
        spin_renderer = DepthMapRenderer(device=device,camera=camera, scene=spin_scene, raster_settings=raster_settings, MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH, img_size=IMG_SIZE)
    elif referencetype == 'enclosed_spin':
        #obs gen
        obs1 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.5, center_position=torch.tensor([-3, 0, 0]))
        obs2 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.2, center_position=torch.tensor([3, 0, 0]))

        obs3 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.8, center_position=torch.tensor([0, 0, 3]))
        obs4 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.5, center_position=torch.tensor([0, 0, -3]))

        obs5 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.2, center_position=torch.tensor([0, 3, 0]))
        obs6 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.2, center_position=torch.tensor([0, -3, 0]))
        obs_list = [obs1, obs2, obs3, obs4, obs5, obs6]

        #path gen
        n_waypoints = generate_random_waypoints(3,"line")
        path = QPMI(n_waypoints)
        path = None

        #find bounds
        bounds, _ = get_scene_bounds(obs_list,path,padding=1)

        #Finding width, height and depth of the cube to encase the scene
        c_width = bounds[1] - bounds[0]
        c_height = bounds[3] - bounds[2]
        c_depth = bounds[5] - bounds[4]
        #Finding the center of the cube
        c_center = torch.tensor([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        #Creating the cube obstacle
        cube = CubeMeshObstacle(device=device, width=c_width, height=c_height, depth=c_depth, center_position = c_center)

        obs_list.append(cube)
        enclosed_scene = Scene(device=device, obstacles=obs_list)
        enclosed_renderer = DepthMapRenderer(device=device,camera=camera, scene=enclosed_scene, raster_settings=raster_settings, MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH, img_size=IMG_SIZE)
    
    elif referencetype == "spin_in_house":
        obs = ImportedMeshObstacle(device, "gym_quad/meshes/house_TRI.obj", torch.tensor([0, 0, 0]))
        spin_house_scene = Scene(device=device, obstacles=[obs])
        spin_house_renderer = DepthMapRenderer(device=device,camera=camera, scene=spin_house_scene, raster_settings=raster_settings, MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH, img_size=IMG_SIZE)

    elif referencetype == "enclosed_circle":
        #obs gen
        # obs1 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.3, center_position=torch.tensor([-2, 0, 0]))
        # obs2 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=1.1, center_position=torch.tensor([2, 0.5, 0]))
        obs3 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.5, center_position=torch.tensor([0, 0, 0]))
        # obs4 = SphereMeshObstacle(device=device, path=unit_sphere_path, radius=0.5, center_position=torch.tensor([0, -1, 0]))
        obs_list = [obs3]

        #path gen
        n_waypoints = generate_random_waypoints(3,"line")
        path = QPMI(n_waypoints)
        path = None

        #Homemade bounds calculation
        bounds, scaled_bounds = get_scene_bounds(obs_list,path,padding=1)
        #Finding width, height and depth of the cube to encase the scene
        c_width = bounds[1] - bounds[0]
        c_height = bounds[3] - bounds[2]
        c_depth = bounds[5] - bounds[4]
        #Finding the center of the cube
        c_center = torch.tensor([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        #Creating the cube obstacle
        cube_obstacle = CubeMeshObstacle(device=device, width=c_width, height=c_height, depth=c_depth, center_position = c_center, inverted=True)

        obs_list.append(cube_obstacle)
        enclosed_scene = Scene(device=device, obstacles=obs_list)
        enclosed_circle_renderer = DepthMapRenderer(device=device,camera=camera, scene=enclosed_scene, raster_settings=raster_settings, MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH, img_size=IMG_SIZE)


    #NB SAVES THE DEPTHMAPS TO THE TEST_IMG FOLDER IN THE TESTS FOLDER
    path_to_save_depth_maps = "tests/test_img/depth_maps/" #If you run from cli
    # path_to_save_depth_maps = os.path.join(grand_parent_dir, "../tests/test_img/depth_maps/")
    os.makedirs(path_to_save_depth_maps, exist_ok=True)

    for i in tqdm(range(n_steps)):
        position = positions[i]
        orientation = orientations[i]

        if referencetype == 'line':
            R, T = sphere_renderer.camera_R_T_from_quad_pos_orient(position, orientation)
            sphere_renderer.update_R(R)
            sphere_renderer.update_T(T)
            depth_map = sphere_renderer.render_depth_map()
            #Save the depth map as an npy file
            np.save(path_to_save_depth_maps+f"depth_map{i}.npy", depth_map.cpu().numpy())
            print("min depth: ", torch.min(depth_map).item(), "max depth: ", torch.max(depth_map).item())  
            sphere_renderer.save_depth_map(path_to_save_depth_maps+f"depth_map{i}.png", depth_map)

        elif referencetype == 'circle':
            R, T = circle_renderer.camera_R_T_from_quad_pos_orient(position, orientation)
            circle_renderer.update_R(R)
            circle_renderer.update_T(T)
            depth_map = circle_renderer.render_depth_map()
            circle_renderer.save_depth_map(path_to_save_depth_maps+f"depth_map{i}.png", depth_map)

        elif referencetype == 'spin':
            R, T = spin_renderer.camera_R_T_from_quad_pos_orient(position, orientation)
            spin_renderer.update_R(R)
            spin_renderer.update_T(T)
            depth_map = spin_renderer.render_depth_map()
            spin_renderer.save_depth_map(path_to_save_depth_maps+f"depth_map{i}.png", depth_map)

        elif referencetype == 'enclosed_spin':
            R, T = sphere_renderer.camera_R_T_from_quad_pos_orient(position, orientation)
            enclosed_renderer.update_R(R)
            enclosed_renderer.update_T(T)
            depth_map = enclosed_renderer.render_depth_map()
            enclosed_renderer.save_depth_map(path_to_save_depth_maps+f"depth_map{i}.png", depth_map)

        elif referencetype == 'spin_in_house':
            R, T = spin_house_renderer.camera_R_T_from_quad_pos_orient(position, orientation)
            spin_house_renderer.update_R(R)
            spin_house_renderer.update_T(T)
            depth_map = spin_house_renderer.render_depth_map()
            spin_house_renderer.save_depth_map(path_to_save_depth_maps+f"depth_map{i}.png", depth_map)

        elif referencetype == 'enclosed_circle':
            R, T = sphere_renderer.camera_R_T_from_quad_pos_orient(position, orientation)
            enclosed_circle_renderer.update_R(R)
            enclosed_circle_renderer.update_T(T)
            depth_map = enclosed_circle_renderer.render_depth_map()
            enclosed_circle_renderer.save_depth_map(path_to_save_depth_maps+f"depth_map{i}.png", depth_map)