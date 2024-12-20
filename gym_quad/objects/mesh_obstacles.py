import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene

#For collisionchecking
from pytorch3d.structures import Meshes

import numpy as np

#For mesh creation
import trimesh

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grand_parent_dir = os.path.dirname(parent_dir)
sys.path.append(grand_parent_dir)
from gym_quad.utils.geomutils import pytorch3d_to_enu, enu_to_pytorch3d, enu_to_tri, Rzyx, tri_Rotmat 
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints

#Utility funcition to get the bounds of the scene given the obstacles and the path used to enclose the scene in a box
def get_scene_bounds(obstacles: list, path: QPMI, padding=10):
    """
    Input:
        obstacles: List of obstacle objects (theyre poisitions are in pt3d and tri frame)
        path: QPMI object (waypoints are in enu frame and gets converted to tri frame inside this funciton)
        padding: Padding to add to the scene bounds [m]

    Returns: [xmin, xmax, ymin, ymax, zmin, zmax] for the scene and the path in tri/pt3d frame"""
    inf = 1000
    bounds = [inf, -inf, inf, -inf, inf, -inf]

    if obstacles != [] :
        for obs in obstacles:
            min_vals, max_vals = obs.get_bounding_box()
            bounds[0] = min(bounds[0], min_vals[0])
            bounds[1] = max(bounds[1], max_vals[0])
            bounds[2] = min(bounds[2], min_vals[1])
            bounds[3] = max(bounds[3], max_vals[1])
            bounds[4] = min(bounds[4], min_vals[2])
            bounds[5] = max(bounds[5], max_vals[2])

    if path != None:
        for point in path.waypoints:
            point = enu_to_tri(point)
            bounds[0] = min(bounds[0], point[0])
            bounds[1] = max(bounds[1], point[0])
            bounds[2] = min(bounds[2], point[1])
            bounds[3] = max(bounds[3], point[1])
            bounds[4] = min(bounds[4], point[2])
            bounds[5] = max(bounds[5], point[2])

    for i in range(6):
        if i % 2 == 0:
            bounds[i] -= padding  # Decrease min values
        else:
            bounds[i] += padding  # Increase max values

    # Calculate the scaled bounds that makes the x,y and z axis equal length but still contains the whole scene
        scene_width = bounds[1] - bounds[0]
        scene_height = bounds[3] - bounds[2]
        scene_depth = bounds[5] - bounds[4]
        max_scene_extent = max(scene_width, scene_height, scene_depth)
        center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        scaled_bounds = [center[0] - max_scene_extent / 2, center[0] + max_scene_extent / 2,
                            center[1] - max_scene_extent / 2, center[1] + max_scene_extent / 2,
                            center[2] - max_scene_extent / 2, center[2] + max_scene_extent / 2]

    return bounds, scaled_bounds


#CREATION OF MESHES#
#Create a unit cube mesh with inwards facing normals
def create_cube(width=1, height=1, depth=1, inverted=True):
    cube = trimesh.creation.box(extents=[width, height, depth],inscribed=False)
    if inverted:
        cube.invert()
    else:
        cube.fix_normals()
    return cube

#Create a unit cylinder mesh with inwards facing normals
def advanced_create_cylinder(radius=1, height=1, sections=8, rot90 =False, inverted=True):
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    if inverted:
        cylinder.invert()
    if rot90:
        cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    return cylinder

def create_cylinder():
    cylinder = trimesh.creation.cylinder(radius=1, height=1, sections=8)
    cylinder.invert()
    return cylinder

#Create a unit sphere mesh with outwar facing normals
def create_sphere():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1)
    return sphere
###


# New obstacle setup with superclass-------------------
#TODO: Make this. Per now using separate classes as its fastest to implement
# New obstacle setup with superclass-------------------


### Mesh Obstacle Classes 
class SphereMeshObstacle: #TODO can make the mesh here as done in cubemehsobstacle below rather than reading from file?
    def __init__(self,
                 device: torch.device,
                 path: str,
                 radius: float,
                 center_position: torch.Tensor,
                 isDummy: bool = False):
        
        self.device = device
        self.path = path                                    # Assumes path points to UNIT sphere .obj file
        self.radius = radius
        self.isDummy = isDummy

        self.center_position = center_position.to(device=self.device).float()   # Centre of the sphere in camera world frame
        #Not specified in name to simplify rewriting of code

        self.position = pytorch3d_to_enu(center_position,device=self.device).float() # Centre of the sphere in ENU frame
        #Not specified in name to simplify rewriting of code

        self.mesh = load_objs_as_meshes([path], device=self.device)
        self.mesh.scale_verts_(scale=float(self.radius))
        self.mesh.offset_verts_(vert_offsets_packed=self.center_position.float())

    def resize(self, new_radius: float):
        self.mesh.scale_verts_(scale=new_radius/self.radius)
        self.radius = new_radius

    def move(self, new_center_position: torch.Tensor):
        new_center_position = new_center_position.to(self.device).float()
        self.mesh.offset_verts_(vert_offsets_packed=new_center_position-self.center_position)
        self.center_position = new_center_position

    def set_device(self, new_device: torch.device):
        self.device = new_device
        self.mesh.to(new_device)
        self.center_position.to(new_device)

    def get_bounding_box(self):
        # Compute the bounding box by taking the min and max of vertices
        vertices = self.mesh.verts_packed()  # This will return all vertices in the mesh
        min_vals, _ = torch.min(vertices, dim=0)
        max_vals, _ = torch.max(vertices, dim=0)
        return min_vals, max_vals


    def return_plot_variables(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.position[0].item() + self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.position[1].item() + self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.position[2].item() + self.radius * np.outer(np.ones(np.size(u)), np.cos(v))

        return [x,y,z]
    
class CylinderMeshObstacle:
    def __init__(self,
                 device: torch.device,
                 radius: float,
                 height: float,
                 center_position: torch.Tensor,
                 inverted: bool = True,
                 isDummy: bool = False):

        self.isDummy = isDummy
        self.device = device
        self.center_position = center_position.to(device=self.device).float() # Centre of the cylinder in camera world frame
        self.position = pytorch3d_to_enu(center_position,device=self.device).float() # Centre of the cylinder in ENU frame

        self.radius = radius
        self.height = height

        self.mesh = self.create_cylinder(radius, height, inverted)
        self.mesh.offset_verts_(vert_offsets_packed=self.center_position)
    
    def set_device(self, new_device: torch.device):
        self.device = new_device
        self.mesh.to(new_device)
        self.center_position.to(new_device)

    def move(self, new_center_position):
        # Check if new_center_position is a list and convert to tensor if necessary
        if isinstance(new_center_position, list):
            new_center_position = torch.tensor(new_center_position, device=self.device, dtype=torch.float32)
        elif isinstance(new_center_position, torch.Tensor) and new_center_position.device != self.device:
            new_center_position = new_center_position.to(self.device)
        
        # Ensure the tensor is of dtype float32
        new_center_position = new_center_position.float()

        # Calculate displacement and update mesh position and center_position attribute
        displacement = new_center_position - self.center_position
        self.mesh.offset_verts_(vert_offsets_packed=displacement)
        self.center_position = new_center_position

    def resize(self, new_radius: float, new_height: float):
        scale_factor = new_radius / self.radius
        self.mesh.scale_verts_(scale=scale_factor)
        self.radius = new_radius
        self.height = new_height

    def get_bounding_box(self):
        # Compute the bounding box by taking the min and max of vertices
        vertices = self.mesh.verts_packed()  # This will return all vertices in the mesh
        min_vals, _ = torch.min(vertices, dim=0)
        max_vals, _ = torch.max(vertices, dim=0)
        return min_vals, max_vals

    def create_cylinder(self, radius=1, height=1, inverted=False):
        cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=8)
        if inverted:
            cylinder.invert()
        else:
            cylinder.fix_normals()
        #Converting from trimesh to pytorch3d mesh
        verts = torch.tensor(cylinder.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(cylinder.faces, dtype=torch.long, device=self.device)

        return Meshes(verts=[verts], faces=[faces])
    
class CubeMeshObstacle:
    def __init__(self,
                 device: torch.device,
                 center_position: torch.Tensor,
                 width: float = None,
                 height: float = None,
                 depth: float = None,
                #  radius: float= None,
                 inverted: bool = True,
                 isDummy: bool = False):

        self.isDummy = isDummy
        self.device = device
        self.center_position = center_position.to(device=self.device).float() # Centre of the cube in camera world frame
        self.position = pytorch3d_to_enu(center_position,device=self.device).float() # Centre of the cube in ENU frame

        self.width = width
        self.height = height
        self.depth = depth
        self.radius = self.width #TODO update to stuff below if time. This works for now

        # if self.radius != None:
        #     self.width = (2/np.sqrt(3))*radius #"Sphere encompasses cube"
        #     self.height = (2/np.sqrt(3))*radius
        #     self.depth = (2/np.sqrt(3))*radius
        #     self.radius = radius
        #     print("CubeMeshObstacle: Using radius to set width, height and depth")
        
        # if width==None and height==None and depth==None and radius==None:
        #     raise ValueError("Either width, height and depth or radius must be specified")

        self.mesh = self.create_cube(width, height, depth, inverted)
        self.mesh.offset_verts_(vert_offsets_packed=self.center_position)
    
    def set_device(self, new_device: torch.device):
        self.device = new_device
        self.mesh.to(new_device)
        self.center_position.to(new_device)

    def move(self, new_center_position):
        # Check if new_center_position is a list and convert to tensor if necessary
        if isinstance(new_center_position, list):
            new_center_position = torch.tensor(new_center_position, device=self.device, dtype=torch.float32)
        elif isinstance(new_center_position, torch.Tensor) and new_center_position.device != self.device:
            new_center_position = new_center_position.to(self.device)
        
        # Ensure the tensor is of dtype float32
        new_center_position = new_center_position.float()

        # Calculate displacement and update mesh position and center_position attribute
        displacement = new_center_position - self.center_position
        self.mesh.offset_verts_(vert_offsets_packed=displacement)
        self.center_position = new_center_position

    def resize(self, new_side_length: float, type: str):
        if type == "width":
            scale_factor = new_side_length / self.width
            self.mesh.scale_verts_(scale=scale_factor)
            self.width = new_side_length
        elif type == "height":
            scale_factor = new_side_length / self.height
            self.mesh.scale_verts_(scale=scale_factor)
            self.height = new_side_length
        elif type == "depth":
            scale_factor = new_side_length / self.depth
            self.depth = new_side_length
            self.mesh.scale_verts_(scale=scale_factor)


    def get_bounding_box(self):
        # Compute the bounding box by taking the min and max of vertices
        vertices = self.mesh.verts_packed()  # This will return all vertices in the mesh
        min_vals, _ = torch.min(vertices, dim=0)
        max_vals, _ = torch.max(vertices, dim=0)
        return min_vals, max_vals

    def create_cube(self, width=1, height=1, depth=1, inverted=True):
        if isinstance(width, torch.Tensor):
            width = width.item()
        if isinstance(height, torch.Tensor):
            height = height.item()
        if isinstance(depth, torch.Tensor):
            depth = depth.item()
        
        cube = trimesh.creation.box(extents=[width, height, depth],inscribed=False)
        if inverted:
            cube.invert()
        else:
            cube.fix_normals()
        #Converting from trimesh to pytorch3d mesh
        verts = torch.tensor(cube.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(cube.faces, dtype=torch.long, device=self.device)
        
        return Meshes(verts=[verts], faces=[faces])

    def return_plot_variables(self):
        x = np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
                      [-1, -1, 1, 1, -1, -1, 1, 1],
                      [-1, -1, -1, -1, 1, 1, 1, 1]])
        x = self.center_position[0].item() + 0.5*self.width*x
        y = np.array([[-1, -1, 1, 1, -1, -1, 1, 1],
                      [-1, 1, 1, -1, -1, 1, 1, -1],
                      [-1, -1, -1, -1, 1, 1, 1, 1]])
        y = self.center_position[1].item() + 0.5*self.depth*y
        z = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, 1, 1, 1, 1],
                      [-1, 1, 1, -1, -1, -1, -1, 1]])
        return [x,y,z]

class ImportedMeshObstacle:
    def __init__(self, device: torch.device, path: str, center_position: torch.Tensor, isDummy: bool = False):
        self.device = device
        self.isDummy = isDummy
        self.path = path
        self.center_position = center_position.to(device=self.device).float()
        self.position = pytorch3d_to_enu(center_position,device=self.device).float()
        self.mesh = load_objs_as_meshes([path], device=self.device)
        self.mesh.offset_verts_(vert_offsets_packed=self.center_position)

    def move(self, new_center_position):
        # Check if new_center_position is a list and convert to tensor if necessary
        if isinstance(new_center_position, list):
            new_center_position = torch.tensor(new_center_position, device=self.device, dtype=torch.float32)
        elif isinstance(new_center_position, torch.Tensor) and new_center_position.device != self.device:
            new_center_position = new_center_position.to(self.device)
        
        # Ensure the tensor is of dtype float32
        new_center_position = new_center_position.float()

        # Calculate displacement and update mesh position and center_position attribute
        displacement = new_center_position - self.center_position
        self.mesh.offset_verts_(vert_offsets_packed=displacement)
        self.center_position = new_center_position

    def set_device(self, new_device: torch.device):
        self.device = new_device
        self.mesh.to(new_device)
        self.center_position.to(new_device)
    
    def get_bounding_box(self):
        # Compute the bounding box by taking the min and max of vertices
        vertices = self.mesh.verts_packed()  # This will return all vertices in the mesh
        min_vals, _ = torch.min(vertices, dim=0)
        max_vals, _ = torch.max(vertices, dim=0)
        return min_vals, max_vals
    
#OUR SCENE CLASS
class Scene:
    def __init__(self,
                 device: torch.device,
                 obstacles: list):

        self.device = device
        self.geometries = obstacles
        self.meshes = [o.mesh for o in obstacles]
        self.joined_scene = self.update_scene() # Textures not included by default

    def update_scene(self, include_textures: bool = False):
        return join_meshes_as_scene(meshes=self.meshes, include_textures=include_textures)

    def resize_mesh(self, mesh_idx: int, new_radius: float):
        self.meshes[mesh_idx].resize(new_radius)

    def move_mesh(self, mesh_idx: int, new_center_position: torch.Tensor):
        self.meshes[mesh_idx].move(new_center_position)

    def add_mesh(self, new_mesh):
        self.meshes.append(new_mesh)
        self.joined_scene = join_meshes_as_scene(meshes=self.meshes, include_textures=False)

    def remove_mesh(self, mesh_idx: int):
        self.meshes.pop(mesh_idx)
        self.joined_scene = join_meshes_as_scene(meshes=self.meshes, include_textures=False)

    def set_device(self, new_device: torch.device):
        self.device = new_device
        for m in self.meshes:
            m.set_device(new_device)


if __name__ == "__main__":

    ''' # "Mesh" creation and display (trimesh)
        # "Camera" how to use the obstacle classes for camera (pytorch3d)
        # "Collision" how to use the obstacle classes for collision checking (trimesh)
        # "Line_path_collision" how to use the obstacle classes for collision checking with a room generated depending on path (trimesh)
        # "convert_to_obj" how to convert e.g. dae files to obj files
        # "rotate_mesh_ENU_to_TRI" how to rotate and save a mesh from ENU to TRI/PT3D frame
    '''
    # mode = "rotate_mesh_ENU_to_TRI" 
    mode = "collision"

    if mode == "mesh":
        ## MESH CREATION ### 
        # cube = create_cube(1,4,1)
        # cylinder = create_cylinder()
        
        #Export the meshes to .obj files
        # cube.export("cube.obj")
        # cylinder.export("cylinder.obj")
        
        #Load the meshes
        house = trimesh.load("gym_quad/meshes/house_TRI.obj")

        #Visualize the meshes
        scene = trimesh.Scene([house])

        axis = trimesh.creation.axis(origin_size=0.1, axis_radius=0.01, axis_length=6.0)
        scene.add_geometry(axis)
        scene.show()

    elif mode == "convert_to_obj":
        #Convert e.g. dae files to obj files
        #Import the file to be converted
        mesh = trimesh.load("gym_quad/meshes/house.dae")
        #Export the mesh to an obj file
        mesh.export("gym_quad/meshes/house_ENU.obj")

    elif mode == "rotate_mesh_ENU_to_TRI":
        #Load the mesh you want to rotate
        mesh = trimesh.load("gym_quad/meshes/house_ENU.obj")
        #Rotate the mesh to be in the TRI/PT3D frame
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])) #Rotate 90 degrees around z-axis
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])) #Rotate -90 degrees around x-axis
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])) #Rotate 180 degrees around y-axis
        #Export the rotated mesh to an obj file
        mesh.export("gym_quad/meshes/house_TRI.obj")

    elif mode == "camera":
        ####PYTORCH3D FOR CAMERA For more use see depth_camera.py
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        s1 = SphereMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/sphere.obj", radius=0.25, center_position=torch.tensor([0, 4, 0]))
        cu1 = CubeMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/cube.obj", side_length=8, center_position=torch.tensor([0, 0, 0]))

        obs = [s1, cu1]
        obs_scene_for_camera = Scene(device=torch.device("cuda"), obstacles=obs)
        ####

    elif mode == "collision":
        ###TRIMESH FOR COLLISION
        #THE OBSTACLES
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        s1 = SphereMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/sphere.obj", radius=0.25, center_position=torch.tensor([0, 4, 0]))
        s2 = SphereMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/sphere.obj", radius=0.4, center_position=torch.tensor([0, 0, 3.5]))
        cu1 = CubeMeshObstacle(device=torch.device("cuda"), width=8, height=8, depth=8 , center_position=torch.tensor([0, 0, 0]))

        obs = [s1, s2, cu1]
        #Converting the pytorch3d meshes to trimesh meshes
        obs_meshes = [o.mesh for o in obs] #extract the meshes from the obstacles

        #Convert the meshes to trimesh meshes
        tri_obs_meshes = [trimesh.Trimesh(vertices=o.verts_packed().cpu().numpy(), faces=o.faces_packed().cpu().numpy()) for o in obs_meshes]

        #Join the obstacle meshes into one mesh
        tri_joined_obs_mesh = trimesh.util.concatenate(tri_obs_meshes)
        #Fix the normals of the joined mesh (if an inverted box is included well get errors later if this step is skipped)
        tri_joined_obs_mesh.fix_normals() #NB This will make the normals point outwards so cant use this for the camera only collision handling

        #THE QUADCOPTER
        #OLD
        #Create a quadcopter mesh as a sphere with r=0.25 at position [0, 0, 0] to do collision checking
        # tri_quad_mesh = trimesh.load("gym_quad/meshes/sphere.obj")
        # #resize the quadcopter sphere to have radius 0.25
        # tri_quad_mesh.apply_scale(0.25)


        #Create a quadcopter mesh as a cylinder with r=0.25 and h=0.21 at position [0, 0, 0] to do collision checking
        tri_quad_mesh = advanced_create_cylinder(radius=0.13, height=0.11, sections=16, rot90=True, inverted=False)
        #Create quadcopter as the actual quadcopter mesh! #Keep using cylinder in sim as drone mesh is comp heavy.
        # tri_quad_mesh = trimesh.load("gym_quad/meshes/drone_TRI.obj")
        #Create the quad as the uncaged drone mesh
        tri_quad_mesh_uncaged = trimesh.load("gym_quad/meshes/uncaged_drone_TRI.obj")

        #Move the quadcopter mesh to start at the quadcopter initial position
        quadcopter_initial_position = np.array([0, 0, 0]) #ENU
        tri_quad_init_pos = enu_to_tri(quadcopter_initial_position)
        tri_quad_mesh.apply_translation(tri_quad_init_pos)
        #Rotate 90 degrees about y axis
        tri_quad_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]))
        
        # #Uncaged
        # tri_quad_init_pos = enu_to_tri(np.array([0.3, 0, 0]))
        # tri_quad_mesh_uncaged.apply_translation(tri_quad_init_pos)
        # #Rotate 90 degrees about y axis
        # tri_quad_mesh_uncaged.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]))        

        # #For comparing the cylinder and the quadcopter mesh
        # trimesh_scene = trimesh.Scene(tri_quad_mesh_uncaged) #The one with inverted box
        # axis = trimesh.creation.axis(origin_size=0.1, axis_radius=0.01, axis_length=1.0)
        # trimesh_scene.add_geometry(axis)
        # trimesh_scene.add_geometry(tri_quad_mesh)
        # trimesh_scene.show()


        #PROCESSING THEM
        #Add the joined obstacle mesh and the quadcopter mesh to the collision manager
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object("obs", tri_joined_obs_mesh)

        n_steps = 8 
        #Creating a quadcopter position reference for n timesteps:
        ref = "move_enu_x" #Choose between "move_enu_x", "move_enu_y", "move_enu_z"
        
        quad_pos_ref = None
        
        if ref == "move_enu_x":
            quad_pos_ref = [quadcopter_initial_position + np.array([i, 0, 0]) for i in range(n_steps)]
            quad_att_ref = [np.array([0, np.pi/4, 0]) for i in range(n_steps)]
        elif ref == "move_enu_y":
            quad_pos_ref = [quadcopter_initial_position + np.array([0, i, 0]) for i in range(n_steps)]
            quad_att_ref = [np.array([0, np.pi/4, 0]) for i in range(n_steps)]
        elif ref == "move_enu_z":
            quad_pos_ref = [quadcopter_initial_position + np.array([0, 0, i]) for i in range(n_steps)] 
            quad_att_ref = [np.array([0, np.pi/4, 0]) for i in range(n_steps)]
        
        quad_pos_ref = np.array(quad_pos_ref) #ENU

        
        tri_translation = None
        # Save the initial mesh
        initial_mesh = tri_quad_mesh.copy()

        for i in range(n_steps):
            # Get the desired position and attitude (roll, pitch, yaw) for this timestep
            current_pos_enu = quad_pos_ref[i]
            roll_enu, pitch_enu, yaw_enu = quad_att_ref[i]  # ENU Rzyx

            # Reset the mesh to the initial state (pos and att)
            tri_quad_mesh = initial_mesh.copy()
            #move tri_quad_mesh to the origin Such that the rotation is around the center of the mesh
            tri_quad_mesh.apply_translation(-tri_quad_mesh.centroid)

            # Generate the rotation matrix using Euler angles
            R = tri_Rotmat(roll_enu, pitch_enu, yaw_enu)

            # Apply the rotation matrix
            tri_quad_mesh.apply_transform(R)

            # Translate mesh to the desired position
            tri_quad_mesh.apply_translation(enu_to_tri(current_pos_enu))

            collision_detected = collision_manager.in_collision_single(tri_quad_mesh)

            print("Collision detected at time step: ", i, collision_detected)
            if collision_detected:
                print("\nCollision detected at time step: ", i,"\n")
                break
            elif i == n_steps-1:
                print("\nNo collision detected\n")
            
            
            #Trimesh visualization: PER ITERATION FOR DEBUGGING
            # #Color the joined obstacle mesh red
            # tri_joined_obs_mesh.visual.face_colors = (255, 0, 0, 100)

            # #Change the color of the quadcopter mesh to blue
            # tri_quad_mesh.visual.face_colors = (0, 0, 255, 100)

            # #Choose which mesh to visualize inverted box or not
            # trimesh_scene = trimesh.Scene(tri_obs_meshes) #The one with inverted box
            # # trimesh_scene = trimesh.Scene(tri_joined_obs_mesh) #The one without inverted box

            # # Create lines representing the axes
            # #XYZ: Red, Green, Blue
            # axis = trimesh.creation.axis(origin_size=0.1, axis_radius=0.01, axis_length=6.0)
            # trimesh_scene.add_geometry(axis)

            # trimesh_scene.add_geometry(tri_quad_mesh)
            # trimesh_scene.show()


        ###### #Use trimesh or pyvista to visualize the meshes
        #Trimesh visualization:

        #Color the joined obstacle mesh red
        tri_joined_obs_mesh.visual.face_colors = (255, 0, 0, 100)

        #Change the color of the quadcopter mesh to blue
        tri_quad_mesh.visual.face_colors = (0, 0, 255, 100)

        #Choose which mesh to visualize inverted box or not
        trimesh_scene = trimesh.Scene(tri_obs_meshes) #The one with inverted box
        # trimesh_scene = trimesh.Scene(tri_joined_obs_mesh) #The one without inverted box

        # Create lines representing the axes
        #XYZ: Red, Green, Blue
        axis = trimesh.creation.axis(origin_size=0.1, axis_radius=0.01, axis_length=6.0)
        trimesh_scene.add_geometry(axis)

        trimesh_scene.add_geometry(tri_quad_mesh)
        trimesh_scene.show()

    elif mode == "Line_path_collision":
        #path
        n_wps = generate_random_waypoints(3,"line")
        path = QPMI(n_wps)

        #obstacles
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # s1 = SphereMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/sphere.obj", radius=0.25, center_position=torch.tensor([0, 4, 0]))
        # s2 = SphereMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/sphere.obj", radius=0.25, center_position=torch.tensor([0, 0, 4]))
        dummy_obs_for_line = SphereMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/sphere.obj", radius=0.1, center_position=torch.tensor([0, 10, 0]),isDummy=True)
        obs = [dummy_obs_for_line]
        obs_meshes = [o.mesh for o in obs] #extract the meshes from the obstacles
        #Convert the meshes to trimesh meshes
        tri_obs_meshes = [trimesh.Trimesh(vertices=o.verts_packed().cpu().numpy(), faces=o.faces_packed().cpu().numpy()) for o in obs_meshes]

        tri_joined_obs_mesh = None
        #Room generation from path
        if obs[0].isDummy:
            bounds, sbounds = get_scene_bounds([], path)
            width = bounds[1] - bounds[0]
            height = bounds[3] - bounds[2]
            depth = bounds[5] - bounds[4]
            center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]

            room = CubeMeshObstacle(device=torch.device("cuda"), width=width, height=height, depth=depth, center_position=torch.tensor(center))
            mesh = room.mesh
            room_tri_mesh = trimesh.Trimesh(vertices=mesh.verts_packed().cpu().numpy(), faces=mesh.faces_packed().cpu().numpy())

            #Join the obstacle meshes and room mesh into one mesh    
            tri_joined_obs_mesh = trimesh.util.concatenate(tri_obs_meshes)
            tri_joined_obs_mesh = trimesh.util.concatenate([tri_joined_obs_mesh, room_tri_mesh])
            tri_joined_obs_mesh.fix_normals() #uninvert the room for collision checking
            obs = []
        else:
            bounds, sbounds = get_scene_bounds(obs, path)
            width = bounds[1] - bounds[0]
            height = bounds[3] - bounds[2]
            depth = bounds[5] - bounds[4]
            center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]

            room = CubeMeshObstacle(device=torch.device("cuda"), width=width, height=height, depth=depth, center_position=torch.tensor(center))
            mesh = room.mesh
            room_tri_mesh = trimesh.Trimesh(vertices=mesh.verts_packed().cpu().numpy(), faces=mesh.faces_packed().cpu().numpy())

            #Join the obstacle meshes and room mesh into one mesh    
            tri_joined_obs_mesh = trimesh.util.concatenate(tri_obs_meshes)
            tri_joined_obs_mesh = trimesh.util.concatenate([tri_joined_obs_mesh, room_tri_mesh])
            tri_joined_obs_mesh.fix_normals() #uninvert the room for collision checking

        #quadcopter
        tri_quad_mesh = trimesh.load("gym_quad/meshes/sphere.obj")
        r = 1
        tri_quad_mesh.apply_scale(r)
        quadcopter_initial_position = np.array([0, 0, 0]) #ENU
        tri_quad_init_pos = enu_to_tri(quadcopter_initial_position)
        tri_quad_mesh.apply_translation(tri_quad_init_pos)        

        #collision manager
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object("room", tri_joined_obs_mesh)

        n_steps = 15
        quad_pos_ref = [quadcopter_initial_position + np.array([-i, 0, 0]) for i in range(n_steps)]
        quad_pos_ref = np.array(quad_pos_ref)

        tri_translation = None
        for i in range(n_steps):
            if i == 0:
                tri_translation = enu_to_tri(quad_pos_ref[i] - quadcopter_initial_position)
            else:
                tri_translation = enu_to_tri(quad_pos_ref[i] - quad_pos_ref[i-1])

            tri_quad_mesh.apply_translation(tri_translation)

            collision_detected = collision_manager.in_collision_single(tri_quad_mesh)

            print("Collision detected at time step: ", i, collision_detected)
            if collision_detected:
                print("\nCollision detected at time step: ", i,"\n")
                break
            elif i == n_steps-1:
                print("\nNo collision detected\n")

        #Trimesh visualization:
        scene = trimesh.Scene([room_tri_mesh, tri_quad_mesh, tri_joined_obs_mesh])
        axis = trimesh.creation.axis(origin_size=0.1, axis_radius=0.01, axis_length=6.0)
        scene.add_geometry(axis)
        scene.show()