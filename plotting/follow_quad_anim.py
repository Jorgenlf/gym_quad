import pandas as pd
import numpy as np
import pyvista as pv
import sys
import os
import trimesh
import warnings
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from tqdm import tqdm
from gym_quad.envs.LV_VAE_MESH import LV_VAE_MESH
from drl_config import lv_vae_config
from gym_quad.utils.geomutils import enu_to_tri, tri_Rotmat, tri_to_enu, Rzyx
from gym_quad.objects.QPMI import QPMI
from gym_quad.objects.mesh_obstacles import ImportedMeshObstacle

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Attempting to set window_size on an unavailable render widow.')
warnings.filterwarnings('ignore', message='This plotter is closed and cannot be scaled. Using the last saved image.')

def deg2rad(deg):
    return deg * np.pi / 180

def dash_path(path: np.ndarray, n=50):
    """Creates a new path that consists of every nth element of the original path"""
    new_path = np.ndarray((0,3))
    for i in range(0, len(path), n):
        new_path = np.vstack((new_path, path[i]))
    return new_path

def get_path_as_arr(path: QPMI):
    """Converts QPMI object to numpy array"""
    u = np.linspace(path.us[0], path.us[-1], 10000)
    quadratic_path = []
    for du in u:
        quadratic_path.append(path(du))
        path.get_direction_angles(du)
    quadratic_path = np.array(quadratic_path)
    return quadratic_path

def initialize_plotter(obstacles, path:QPMI, dim, scene="none", save=True):
    #Init values
    plotter = pv.Plotter(window_size=dim, off_screen=save)
    drone_path_color = '#00BA38'   # greenish
    path_color = '#4780ff'         # blueish
    obstacles_color = '#ff3400'    # redish
    room_color = '#f3f0ec'         # deserty/gray

    plotter.camera_position = [10, 10, 10]
    plotter.camera_focal_point = [0, 0, 0]
    plotter.camera_view_up = [0, 0, 1]

    #Extracing obstacles by type
    house_index = None
    for o in obstacles:
        if isinstance(o, ImportedMeshObstacle): # Where "ImportedMeshObstacle" is the house, TODO change to "HouseObstacle" or something
            house_index = obstacles.index(o)

    obs_meshes = [o.mesh for o in obstacles] #Conv from pt3d to trimesh
    tri_obs_meshes = [trimesh.Trimesh(vertices=tri_to_enu(o.verts_packed().cpu().numpy().T).T, faces=tri_to_enu(o.faces_packed().cpu().numpy().T).T) for o in obs_meshes]

    obs_meshes = []
    room_mesh = None
    house_mesh = None
    for i, mesh in enumerate(tri_obs_meshes): #conv from trimesh to pyvista
        # If last mesh, it is the room, do not include in meshes
        if i == len(tri_obs_meshes) - 1 and not scene == "cave":
            room_mesh = pv.wrap(mesh)
            continue
        elif i == house_index:
            house_mesh = pv.wrap(mesh)
            continue
        else:
            pv_mesh = pv.wrap(mesh)
            obs_meshes.append(pv_mesh)

    backface_params = dict(opacity=0.0) # To see through outer walls of enclosing room
    op = 1

    # Add all obstacles
    for i, mesh in enumerate(obs_meshes):
        if i == 0: # Check index to only get one label. Needed bc. custom legends
            plotter.add_mesh(mesh, color=obstacles_color, show_edges=False, label="Obstacles", smooth_shading=False,backface_params=backface_params, opacity = op)
        else:
            plotter.add_mesh(mesh, color=obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = op)
    # Add the room, only plot if not house scenario or cave scenario
    if room_mesh != None and house_mesh == None and not scene == "cave":
        plotter.add_mesh(room_mesh, color=room_color, show_edges=False, backface_params=backface_params)
    if house_mesh != None:
        plotter.add_mesh(house_mesh, color=room_color, show_edges=False, opacity=0.15)

    # Add path
    quadratic_path = dash_path(get_path_as_arr(path))
    plotter.add_points(quadratic_path, color=path_color, point_size=5, label="Path", render_points_as_spheres=True)

    return plotter


if __name__ == "__main__":
    #loading of data
    #exp32 - random #cave "47" #horizontal 37 #deadend 4
    #exp10005 - locked conv #horizontal 1 #house_easy 1 #house_easy_obstacles 1 #helix 1
    
    exp_dir = 'Experiment 10005'
    test_scen = "helix" 
    test_nr = 1
    
    retrieve_data_path = os.path.join(parent_dir, 'log', 'LV_VAE_MESH-v0', exp_dir , test_scen, "tests", f"test{test_nr}")
    
    output_path = os.path.join(parent_dir, 'plotting', 'replotting_results', "follow_quad_imgs" , test_scen, f"test{test_nr}")
    os.makedirs(output_path, exist_ok=True)

    try:
        sim_data = pd.read_csv(retrieve_data_path + '/sim_df.csv')
    except FileNotFoundError:
            print(f"File not found: {retrieve_data_path + '/sim_df.csv'}")
            sim_data = pd.read_csv(retrieve_data_path + '/test_sim.csv')

    #Init of depthmap dimensions to match depthmaps and scene images
    depthmap_dim = (693,474)

    #Init of scene
    replot_config = lv_vae_config.copy()
    replot_config["mesh_path"] = os.path.join(parent_dir, 'gym_quad', 'meshes', 'sphere.obj')
    replot_config["max_t_steps"] = 0
    replot_config["use_uncaged_drone_mesh"] = True
    env = LV_VAE_MESH(env_config=replot_config,scenario=test_scen)
    obstacles = env.unwrapped.obstacles
    path = env.unwrapped.path

    #Init of drone mesh
    tri_quad_mesh = None
    if replot_config["use_uncaged_drone_mesh"]:
        tri_quad_mesh = trimesh.load("gym_quad/meshes/uncaged_drone_TRI.obj")
        #Move mesh to origin to rotate it correctly
        tri_quad_mesh.apply_translation(np.array([0, 0, 0]))
        #Rotate -90 degrees about trimesh y axis #Fix attitude of the drone DONE TO MAKE UP FOR DISCREPANCY FROM BLENDER EXPORT
        tri_quad_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]))

    pos = sim_data.iloc[0][[r"$X$", r"$Y$", r"$Z$"]].values
    att = sim_data.iloc[0][[r'$\phi$', r'$\theta$', r'$\psi$']].values
    tri_quad_init_pos = enu_to_tri(pos)
    tri_quad_mesh.apply_translation(-tri_quad_mesh.centroid)
    tri_quad_mesh.apply_transform(tri_Rotmat(*att))
    initial_tri_quad_mesh = tri_quad_mesh.copy() 
    tri_quad_mesh.apply_translation(tri_quad_init_pos)

    quad_color = '#00BA38'
    plotter = initialize_plotter(obstacles, path, dim=depthmap_dim, scene=test_scen, save=True) #Move down into loop to make several images after each other
    for i in tqdm(range(len(sim_data))):
        pos = sim_data.iloc[i][[r"$X$", r"$Y$", r"$Z$"]].values
        att = sim_data.iloc[i][[r'$\phi$', r'$\theta$', r'$\psi$']].values

        #QUAD UPDATE
        tri_quad_mesh = initial_tri_quad_mesh.copy()
        R = tri_Rotmat(*att)
        tri_quad_mesh.apply_transform(R)
        tri_quad_mesh.apply_translation(enu_to_tri(pos))

        # Convert the trimesh to a pyvista mesh
        vertices_enu = np.array([tri_to_enu(v) for v in tri_quad_mesh.vertices])
        tri_quad_enu_mesh = trimesh.Trimesh(vertices=vertices_enu, faces=tri_quad_mesh.faces)
        quad_mesh = pv.wrap(tri_quad_enu_mesh)
        plotter.add_mesh(quad_mesh, color=quad_color, smooth_shading=True, name="Drone")

        # CAMERA UPDATE
        cam_pos_enu_b = np.array([-0.9, 0, -0.15])
        modified_att = att.copy()
        modified_att[0] = 0
        modified_att[1] = 0

        cam_pos_enu_w = pos + Rzyx(*modified_att).dot(cam_pos_enu_b)
        plotter.camera.position = cam_pos_enu_w
        plotter.camera.focal_point = pos

        # Save the rendered image
        plotter.show()
        if output_path:
            save_file = os.path.join(output_path, f"scene_timestep_{i}.png")
            plotter.screenshot(save_file, scale=40, window_size=depthmap_dim)

    print("Done!")


