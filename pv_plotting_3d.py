import vtk
import pyvista as pv
import numpy as np
import torch
import trimesh

from gym_quad.objects.mesh_obstacles import SphereMeshObstacle, CubeMeshObstacle, Scene
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints
from gym_quad.utils.geomutils import enu_to_tri, tri_to_enu

class Plotter3D:
    def __init__(self, obstacles, path: QPMI, drone_traj: np.ndarray, initial_position: np.ndarray, nosave=False):
        self.obstacles = obstacles
        self.path = path
        self.drone_traj = drone_traj
        self.initial_position = initial_position
        self.nosave = nosave

        self.meshes = self.obstacles_to_pyvista_meshes(obstacles)
        self.quadratic_path = self.dash_path(self.get_path_as_arr(path))
        self.bounds, self.scaled_bounds = self.get_scene_bounds(obstacles, path, drone_traj, padding=0)

        
        self.plotter = pv.Plotter(window_size=[4000, 4000], off_screen=not nosave)
    
    def dash_path(self, path: np.ndarray, n=100):
        """Makes the path equal zeros except for evety nth element"""
        new_path = np.zeros_like(path)
        new_path[::n] = path[::n]
        return new_path

    def get_path_as_arr(self, path: QPMI):
        u = np.linspace(path.us[0], path.us[-1], 10000)
        quadratic_path = []
        for du in u:
            quadratic_path.append(path(du))
            path.get_direction_angles(du)
        quadratic_path = np.array(quadratic_path)
        return quadratic_path
    
    def get_scene_bounds(self, obstacles: list, path: QPMI, drone_traj:np.ndarray, padding=10):
        """Returns [xmin, xmax, ymin, ymax, zmin, zmax] for the scene and the path,
            with padding [m] added to each dimension"""
        inf = 1000
        bounds = [inf, -inf, inf, -inf, inf, -inf]
    
        if obstacles != [] :
            obs_meshes = [o.mesh for o in obstacles]
            tri_obs_meshes = [trimesh.Trimesh(vertices=tri_to_enu(o.verts_packed().cpu().numpy().T).T, faces=tri_to_enu(o.faces_packed().cpu().numpy().T).T) for o in obs_meshes]
            for mesh in tri_obs_meshes:
                for point in mesh.vertices:
                    bounds[0] = min(bounds[0], point[0])
                    bounds[1] = max(bounds[1], point[0])
                    bounds[2] = min(bounds[2], point[1])
                    bounds[3] = max(bounds[3], point[1])
                    bounds[4] = min(bounds[4], point[2])
                    bounds[5] = max(bounds[5], point[2])
    
        if path != None:
            for point in path.waypoints:
                bounds[0] = min(bounds[0], point[0])
                bounds[1] = max(bounds[1], point[0])
                bounds[2] = min(bounds[2], point[1])
                bounds[3] = max(bounds[3], point[1])
                bounds[4] = min(bounds[4], point[2])
                bounds[5] = max(bounds[5], point[2])
        
        if drone_traj.any():
            for point in drone_traj:
                bounds[0] = min(bounds[0], point[0])
                bounds[1] = max(bounds[1], point[0])
                bounds[2] = min(bounds[2], point[1])
                bounds[3] = max(bounds[3], point[1])
                bounds[4] = min(bounds[4], point[2])
                bounds[5] = max(bounds[5], point[2])
    
        for i in range(6):
            bounds[i] += padding if i % 2 == 0 else -padding
    
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
    
    def obstacles_to_pyvista_meshes(self, obstacles: list):
        obs_meshes = [o.mesh for o in obstacles]
        tri_obs_meshes = [trimesh.Trimesh(vertices=tri_to_enu(o.verts_packed().cpu().numpy().T).T, faces=tri_to_enu(o.faces_packed().cpu().numpy().T).T) for o in obs_meshes]
        meshes = []    
        for mesh in tri_obs_meshes:
            pv_mesh = pv.wrap(mesh)
            meshes.append(pv_mesh)
        return meshes
    
    def plot_scene_and_trajs(self, save_path=None):
        grid_kw_args = {
            'color': 'gray',
            'xtitle': 'x [m]',
            'ytitle': 'y [m]',
            'ztitle': 'z [m]',
            'font_size': 50,
            'padding': 0.5, #maybe change later
            'font_family': 'times',
            'fmt':'%.0f',
            'ticks': 'outside',
            # axes_ranges just changes values on the axes, not the actual scene, do not use
        }
        self.plotter.add_mesh(pv.Cube(bounds=self.scaled_bounds), opacity=0.0)
        for i, mesh in enumerate(self.meshes):
            if i == 0:
                self.plotter.add_mesh(mesh, color="red", show_edges=False, label="Obstacles", smooth_shading=False)
            else:
                self.plotter.add_mesh(mesh, color="red", show_edges=False, smooth_shading=True)
        self.plotter.add_points(self.quadratic_path, color="#619CFF", point_size=8, label="Path")
        self.plotter.add_points(self.drone_traj, color="#00BA38", point_size=8, label="Drone Trajectory ")
        self.plotter.add_points(self.initial_position, color="black", point_size=30, label="Initial Position")
        self.plotter.show_grid(**grid_kw_args)
        #self.plotter.add_legend(border=False, bcolor='w', face=None, size=(0.12,0.12)) #bcolor="#eaeae8"

        # Custom legend
        offset_x = 900
        offset_y = 500
        offset_between = 100
        legend_pos = [self.plotter.window_size[0] - offset_x, self.plotter.window_size[1] - offset_y]
        drone_color = '#00BA38'
        path_color = '#00B6EB'
        obs_color = 'red'
        initial_pos_color = 'black'
        self.plotter.add_text("Obstacles", position=legend_pos, font_size=40, color=obs_color, font='times')
        self.plotter.add_text("Path", position=[legend_pos[0], legend_pos[1] - offset_between], font_size=40, color=path_color, font='times')
        self.plotter.add_text("Drone Trajectory", position=[legend_pos[0], legend_pos[1] - 2*offset_between], font_size=40, color=drone_color, font='times')
        self.plotter.add_text("Initial Position", position=[legend_pos[0], legend_pos[1] - 3*offset_between], font_size=40, color=initial_pos_color, font='times')

        # Camera stuff
        self.plotter.camera.zoom(0.9)
        
        #if self.nosave: self.plotter.show()
        self.plotter.show()
        if save_path and not self.nosave:
            self.plotter.screenshot(save_path, scale=40, window_size = [1000, 1000])



if __name__ == "__main__":
    pass