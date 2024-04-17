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

def camera_R_T_from_quad_pos_orient(position: np.array, orientation: np.array) -> tuple:
    '''Given a position and orientation of the quad in ENU frame, this function returns the R and T matrices for the camera object in Pytorch3D.'''
    # Convert position and orientation to torch tensors
    at = position + geom.Rzyx(*orientation) @ np.array([1, 0, 0])  # Look at the point in front of the camera along body x-axis
    
    at_torch = torch.from_numpy(at).to(device)
    position_torch = torch.from_numpy(position).to(device)

    at_pt3d = enu_to_pytorch3d(at_torch).to(device).float()
    position_pt3d = enu_to_pytorch3d(position_torch).to(device).float()
    # orientation_torch = torch.from_numpy(orientation).to(device)
    
    # Calculate rotation matrix
    Rstep = look_at_rotation(position_pt3d[None, :], device=device, at=at_pt3d[None, :])  # (1, 3, 3)
    # Calculate translation vector
    Tstep = -torch.bmm(Rstep.transpose(1, 2), position_pt3d[None, :, None])[:, :, 0]   # (1, 3)
    
    return Rstep, Tstep

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
