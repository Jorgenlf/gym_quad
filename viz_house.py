from gym_quad.objects.mesh_obstacles import Scene, get_scene_bounds, ImportedMeshObstacle
from gym_quad.utils.geomutils import enu_to_pytorch3d, enu_to_tri, pytorch3d_to_enu, tri_to_enu
import trimesh
from pytorch3d.io import load_objs_as_meshes
import numpy as np
import torch
import pyvista as pv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


if __name__ == '__main__':
    
    # Params
    nosave = False
    room_color = '#d3f8ff'
    save_path = "./house_mesh.png"
    read_path = "./nowall_house.obj" 
    #read_path = "./house_TRI_new.obj"


    # Globs
    grid_kw_args = {
            'color': 'gray',
            'xtitle': 'x [m]',
            'ytitle': 'y [m]',
            'ztitle': 'z [m]',
            'font_size': 40,
            'padding': 0.0, #maybe change later
            'font_family': 'times',
            'fmt':'%.0f',
            'ticks': 'outside',
        }


    # Load the mesh
    
    obstacle_coords = torch.tensor([0,0,0],device=device).float()
    pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=device)
    house = ImportedMeshObstacle(device=device, path=read_path, center_position=pt3d_obs_coords)

    house_mesh = load_objs_as_meshes([read_path], device=device)
    house_mesh_tri = trimesh.Trimesh(vertices=tri_to_enu(house_mesh.verts_packed().cpu().numpy().T).T, faces=tri_to_enu(house_mesh.faces_packed().cpu().numpy().T).T)
    house_mesh_pv = pv.wrap(house_mesh_tri)
    
    plotter = pv.Plotter(window_size=[4000, 4000], off_screen=not nosave)

    #plotter.add_mesh(pv.Cube(bounds=bounds), opacity=0.0)
    plotter.add_mesh(house_mesh_pv, color=room_color, split_sharp_edges=True, opacity=0.7)

    #plotter.show_grid(**grid_kw_args)

    # Change camera location to be 
    plotter.view_yz()
    plotter.view_vector([0.9, -0.30, 0.1])


    plotter.show()
    if save_path and not nosave:
        plotter.screenshot(save_path, scale=40, window_size = [1000, 1000])



