import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene

#For collisionchecking
from pytorch3d.structures import Meshes
from pytorch3d.ops import utils as pt3d_utils

import numpy as np

#For mesh creation
import trimesh

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grand_parent_dir = os.path.dirname(parent_dir)
sys.path.append(grand_parent_dir)
from gym_quad.utils.geomutils import pytorch3d_to_enu

#CREATION OF MESHES#
#Create a unit cube mesh with inwards facing normals
def create_cube():
    cube = trimesh.creation.box(extents=[1, 1, 1], inscribed=False)
    cube.invert()
    return cube

#Create a unit cylinder mesh with inwards facing normals
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


#A mesh class of the quadcopter representing the quadcopter as a sphere with a given radius
class QuadcopterMesh:
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


### Mesh Obstacle Classes
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
        x = self.position[0].item() + self.radius*np.cos(u)*np.sin(v)
        y = self.position[1].item() + self.radius*np.sin(u)*np.sin(v)
        z = self.position[2].item() + self.radius*np.cos(v)
        return [x,y,z]
    
class CubeMeshObstacle:
    def __init__(self, 
                 device: torch.device, 
                 path: str,
                 side_length: float,
                 center_position: torch.Tensor):
        
        self.device = device
        self.path = path

        self.side_length = side_length
        
        self.center_position = center_position.to(device=self.device) # Centre of the cube in camera world frame
        self.position = pytorch3d_to_enu(center_position).to(device=self.device) # Centre of the cube in ENU frame
        
        self.mesh = load_objs_as_meshes([self.path], device=self.device)
        self.mesh.scale_verts_(scale=self.side_length)
        self.mesh.offset_verts_(vert_offsets_packed=self.center_position)
    
    def resize(self, new_side_length: float):
        self.mesh.scale_verts_(scale=new_side_length/self.side_length)
        self.side_length = new_side_length
    
    def move(self, new_center_position: torch.Tensor):
        new_center_position = new_center_position.to(self.device)
        self.mesh.offset_verts_(vert_offsets_packed=new_center_position-self.center_position)
        self.center_position = new_center_position

    def set_device(self, new_device: torch.device):
        self.device = new_device
        self.mesh.to(new_device)
        self.center_position.to(new_device)

    def return_plot_variables(self):
        x = np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
                      [-1, -1, 1, 1, -1, -1, 1, 1],
                      [-1, -1, -1, -1, 1, 1, 1, 1]])
        x = self.center_position[0].item() + 0.5*self.side_length*x
        y = np.array([[-1, -1, 1, 1, -1, -1, 1, 1],
                      [-1, 1, 1, -1, -1, 1, 1, -1],
                      [-1, -1, -1, -1, 1, 1, 1, 1]])
        y = self.center_position[1].item() + 0.5*self.side_length*y
        z = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, 1, 1, 1, 1],
                      [-1, 1, 1, -1, -1, -1, -1, 1]])
        return [x,y,z]        


#OUR SCENE CLASS------------------- #WORKS
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
#OUR SCENE CLASS-------------------

if __name__ == "__main__":
    
    ### MESH CREATION ###
    # cube = create_cube()
    # cylinder = create_cylinder()
    # #Export the meshes to .obj files
    # cube.export("cube.obj")
    # cylinder.export("cylinder.obj")            
    ### MESH CREATION ###


    # #TODO possibly important tjaa. 
    #Either create mesh.obj objects at runtime or load from file ONCE and store in memory to speed up training

    #init device to use gpu if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    quad = QuadcopterMesh(device=torch.device("cuda"), path="gym_quad/meshes/sphere.obj", radius=1, center_position=torch.tensor([0, 0, 0]))

    s1 = SphereMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/sphere.obj", radius=1, center_position=torch.tensor([0, 0, 0]))

    cu1 = CubeMeshObstacle(device=torch.device("cuda"), path="gym_quad/meshes/cube.obj", side_length=1, center_position=torch.tensor([5, 0, 0]))


    obs = [s1, cu1]
    
    # our_scene = Scene(device=torch.device("cuda"), obstacles=obs)    
    
    # #Use trimesh or pyvista to visualize the meshes
    #Trimesh visualization:
    # Visualize the mesh
    
    # trimesh_scene = trimesh.Scene()
    # trimesh_scene.add_geometry(cu1.mesh)
    # trimesh_scene.add_geometry(s1.mesh)

    # trimesh_scene.show()