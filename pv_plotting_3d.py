import vtk
import pyvista as pv
import numpy as np
import torch
import trimesh


from gym_quad.objects.mesh_obstacles import SphereMeshObstacle, CubeMeshObstacle, Scene, ImportedMeshObstacle
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints
from gym_quad.utils.geomutils import enu_to_tri, tri_to_enu

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# hex_colors = ['#FFFF00','#990099']
# custom_cmap = ListedColormap(hex_colors)

# Define start and end colors
start_color = '#FFFF00'  # Yellow
end_color = '#990099'    # Purple

# Create a colormap that interpolates between the two colors
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [start_color, end_color], N=256)


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

#TODO make these classes inherit from a common class to avoid code duplication
#TODO make them take in a flag of what scene was used as this givs more flexibility in plotting
class Plotter3D: # TODO change so that it is like Plotter3DMultiTraj
    def __init__(self, obstacles, path: QPMI, drone_traj: np.ndarray, initial_position: np.ndarray, nosave=False, force_transparency = False):
        self.obstacles = obstacles
        self.path = path
        self.drone_traj = drone_traj
        self.initial_position = initial_position
        self.nosave = nosave
        self.force_transparency = force_transparency

        self.meshes, self.room_mesh, self.house_mesh = self.obstacles_to_pyvista_meshes(obstacles)
        self.quadratic_path = self.dash_path(self.get_path_as_arr(path))
        #self.quadratic_path = self.get_path_as_arr(path)
        # if any of the obstacles are imported mesh obstacle, set padding to negative value
        padding = 2 if any([isinstance(o, ImportedMeshObstacle) for o in obstacles]) else 0
        self.bounds, self.scaled_bounds = self.get_scene_bounds(obstacles, path, drone_traj, padding=0)

        self.plotter = pv.Plotter(window_size=[4000, 4000], 
                                  off_screen=not nosave)
        
        # Plotting parameters
        self.drone_path_color = '#00BA38' #TODO find suitable color
        self.path_color = '#4780ff' # blueish
        self.obstacles_color = '#ff3400' # redish
        self.room_color = '#f3f0ec' # deserty
        self.initial_pos_color = 'black'
        self.grid_kw_args = {
            'color': 'gray',
            'xtitle': 'x [m]',
            'ytitle': 'y [m]',
            'ztitle': 'z [m]',
            'font_size': 40,
            'padding': 0.0, #maybe change later
            'font_family': 'times',
            'fmt':'%.0f',
            'ticks': 'outside',
            # axes_ranges just changes values on the axes, not the actual scene, do not use
        }
    
    def dash_path(self, path: np.ndarray, n=100):
        """Creates a new path which just consists of every nth element of the original path"""
        new_path = np.ndarray((0,3))
        for i in range(0, len(path), n):
            new_path = np.vstack((new_path, path[i]))
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
        house_index = None
        for o in obstacles:
            if isinstance(o, ImportedMeshObstacle):
                house_index = obstacles.index(o)
        obs_meshes = [o.mesh for o in obstacles]
        tri_obs_meshes = [trimesh.Trimesh(vertices=tri_to_enu(o.verts_packed().cpu().numpy().T).T, faces=tri_to_enu(o.faces_packed().cpu().numpy().T).T) for o in obs_meshes]
        meshes = []
        room = None   
        house = None 
        for i, mesh in enumerate(tri_obs_meshes):
            # If last mesh, it is the room, do not include in meshes
            if i == len(tri_obs_meshes) - 1:
                room = pv.wrap(mesh)
                continue
            if i == house_index:
                house = pv.wrap(mesh)
                continue
            pv_mesh = pv.wrap(mesh)
            meshes.append(pv_mesh)
        return meshes, room, house
    
    def plot_scene_and_trajs(self, save_path=None, azimuth=90, elevation=None, see_from_plane=None, scene="none"): # TODO fix
        """Azimuth is the angle of the camera around the scene, + is anti-clockwise rotation about scene center, 0, 90, 180, 270 are the best angles (from each corner)
           elevation is the angle of the camera above the scene, + makes you see the scene from higher above, default is fine for corner angles so need not change
           see_from_plane is the plane to see the scene from (only if no azimuth is given), can be "xy", "xz" or "yz"
        """
        self.plotter.add_mesh(pv.Cube(bounds=self.scaled_bounds), opacity=0.0)
        backface_params = dict(opacity=0.0) #  To see through outer walls of enclosing room 

        opacity=1.0
        if self.force_transparency:
            opacity = 0.07

        if scene == "cave":
            opacity = 0.07
            self.plotter.add_mesh(self.room_mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = opacity) #Hacky fix to not plot the last obstacle in room colors..

        # Add all obstacles
        for i, mesh in enumerate(self.meshes):
            if i == 0:
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, label="Obstacles", smooth_shading=False,backface_params=backface_params, opacity = opacity)
            else:
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = opacity)

        # Add the room, only plot if not house scenario
        if self.room_mesh != None and self.house_mesh == None and not scene == "cave":
            self.plotter.add_mesh(self.room_mesh, color=self.room_color, show_edges=False, backface_params=backface_params)
        
        if self.house_mesh != None:
            self.plotter.add_mesh(self.house_mesh, color=self.room_color, show_edges=False, opacity=0.25)


        # Add drone traj and path
        spline = pv.Spline(self.drone_traj, 1000)
        self.plotter.add_mesh(spline, line_width=8, label="Drone Trajectory", color=self.drone_path_color)
        self.plotter.add_points(self.quadratic_path, color=self.path_color, point_size=10, label="Path", render_points_as_spheres=True)
        self.plotter.add_points(self.initial_position, color="black", point_size=30, label="Initial Position", render_points_as_spheres=True)
        self.plotter.show_grid(**self.grid_kw_args)

        # Custom legend
        offset_x = 800
        offset_y = 700
        offset_between = 100
        legend_pos = [self.plotter.window_size[0] - offset_x, self.plotter.window_size[1] - offset_y]

        self.plotter.add_text("Obstacles", position=legend_pos, font_size=40, color=self.obstacles_color, font='times')
        self.plotter.add_text("Path", position=[legend_pos[0], legend_pos[1] - offset_between], font_size=40, color=self.path_color, font='times')
        self.plotter.add_text("Drone Trajectory", position=[legend_pos[0], legend_pos[1] - 2*offset_between], font_size=40, color=self.drone_path_color, font='times')
        self.plotter.add_text("Initial Position", position=[legend_pos[0], legend_pos[1] - 3*offset_between], font_size=40, color=self.initial_pos_color, font='times')

        # Camera stuff
        self.plotter.camera.zoom(0.9) # Zoom a bit out to include axes from corner views
        
        if see_from_plane == None:
            self.plotter.camera.azimuth = azimuth # Controls the rotation of the camera around the scene, + is anti-clockwise rotation about scene center
        if elevation is not None:
            self.plotter.camera.elevation = elevation # Controls the angle of the camera above the scene, + makes you see the scene from higher above

        if see_from_plane == "xy":
            self.plotter.view_xy()
        elif see_from_plane == "xz":
            self.plotter.view_xz(negative=True)
        elif see_from_plane == "yz":
            self.plotter.view_yz()
                
        #if self.nosave: self.plotter.show()
        self.plotter.show()
        if save_path and not self.nosave:
            self.plotter.screenshot(save_path, scale=40, window_size = [1000, 1000])
        
    def plot_only_scene(self, save_path=None, azimuth=90, elevation=None, see_from_plane=None, scene = "none"):
        self.plotter.add_mesh(pv.Cube(bounds=self.scaled_bounds), opacity=0.0)
        backface_params = dict(opacity=0.0)
        opacity=1.0
        if self.force_transparency:
            opacity = 0.07

        
        if scene == "cave":
            self.plotter.add_mesh(self.room_mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = opacity) #Hacky fix to not plot the last obstacle in room colors..

        # Add all obstacles
        for i, mesh in enumerate(self.meshes):
            if i == 0:
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, label="Obstacles", smooth_shading=False,backface_params=backface_params, opacity = opacity)
            else:
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = opacity)

        # Add the room, only plot if not house scenario
        if self.room_mesh != None and self.house_mesh == None and not scene == "cave":
            self.plotter.add_mesh(self.room_mesh, color=self.room_color, show_edges=False, backface_params=backface_params)
        
        if self.house_mesh != None:
            self.plotter.add_mesh(self.house_mesh, color=self.room_color, show_edges=False, opacity=0.25)

        # Add drone traj and path
        self.plotter.add_points(self.quadratic_path, color=self.path_color, point_size=20, label="Path", render_points_as_spheres=True)
        self.plotter.add_points(self.initial_position, color="black", point_size=30, label="Initial Position", render_points_as_spheres=True)
        self.plotter.show_grid(**self.grid_kw_args)

        # Custom legend
        offset_x = 800
        offset_y = 700
        offset_between = 100
        legend_pos = [self.plotter.window_size[0] - offset_x, self.plotter.window_size[1] - offset_y]

        self.plotter.add_text("Path", position=legend_pos, font_size=40, color=self.path_color, font='times')
        self.plotter.add_text("Obstacles", position=[legend_pos[0], legend_pos[1] - 1*offset_between], font_size=40, color=self.obstacles_color, font='times')
        self.plotter.add_text("Initial Position", position=[legend_pos[0], legend_pos[1] - 2*offset_between], font_size=40, color=self.initial_pos_color, font='times')

        # Camera stuff
        self.plotter.camera.zoom(0.9) # Zoom a bit out to include axes from corner views

        if see_from_plane == None:
            self.plotter.camera.azimuth = azimuth
        if elevation is not None:
            self.plotter.camera.elevation = elevation

        if see_from_plane == "xy":
            self.plotter.view_xy()
        elif see_from_plane == "xz":
            self.plotter.view_xz(negative=True)
        elif see_from_plane == "yz":
            self.plotter.view_yz()

        #if self.nosave: self.plotter.show()
        self.plotter.show()
        if save_path and not self.nosave:
            self.plotter.screenshot(save_path, scale=40, window_size = [1000, 1000])
        

    


class Plotter3DMultiTraj(): # Might inherit from Plotter3D and stuff later for improved code quality
    def __init__(self, obstacles, path: QPMI, drone_trajectories: dict, cum_rewards: dict, nosave=False, force_transparency = False):
        self.obstacles = obstacles
        self.path = path
        self.drone_trajectories = drone_trajectories 
        self.cum_rewards = cum_rewards
        self.nosave = nosave
        self.force_transparency = force_transparency

        self.meshes, self.room_mesh, self.house_mesh = self.obstacles_to_pyvista_meshes(obstacles)
        self.quadratic_path = self.dash_path(self.get_path_as_arr(path))

        padding = 2 if any([isinstance(o, ImportedMeshObstacle) for o in obstacles]) else 0        
        self.bounds, self.scaled_bounds = self.get_scene_bounds(obstacles, path, list(drone_trajectories.values()), padding=padding)

        self.min_rew, self.max_rew = self.find_min_max_rewards(cum_rewards)

        
        self.plotter = pv.Plotter(window_size=[4000, 4000], 
                                  off_screen=not nosave)
        

        # Plotting parameters
        self.clim = [self.min_rew, self.max_rew]
        self.cmap = custom_cmap # OLD 'YlGn'
        self.flip_scalars = True # To invert the colorbar
        self.path_color = '#4780ff' # blueish
        self.obstacles_color = '#ff3400' # redish
        self.room_color = '#f3f0ec' # deserty
        self.grid_kw_args = {
            'color': 'gray',
            'xtitle': 'x [m]',
            'ytitle': 'y [m]',
            'ztitle': 'z [m]',
            'font_size': 40,
            'padding': 0.0, #maybe change later
            'font_family': 'times',
            'fmt':'%.0f',
            'ticks': 'outside',
            # axes_ranges just changes values on the axes, not the actual scene, do not use
        }
        self.scalar_bar_args = {
            'title' : 'Reward',
            'n_labels' : 3,
            'title_font_size' : 60,
            'label_font_size' : 55,
            'font_family' : 'times',
            'vertical' : True,
            'width' : 0.08,
            'height' : 0.5,
            'position_x' : 0.9,
            'position_y' : 0.3,

        }

    def find_min_max_rewards(self, cum_rewards: dict):
        min_reward = 100000
        max_reward = -100000
        for ep, rew in cum_rewards.items():
            min_reward = min(min_reward, rew)
            max_reward = max(max_reward, rew)
        return min_reward, max_reward
    
    def dash_path(self, path: np.ndarray, n=100):
        """Creates a new path which just consists of every nth element of the original path"""
        new_path = np.ndarray((0,3))
        for i in range(0, len(path), n):
            new_path = np.vstack((new_path, path[i]))
        return new_path
    
    
    def get_path_as_arr(self, path: QPMI):
        u = np.linspace(path.us[0], path.us[-1], 10000)
        quadratic_path = []
        for du in u:
            quadratic_path.append(path(du))
            path.get_direction_angles(du)
        quadratic_path = np.array(quadratic_path)
        return quadratic_path
    
    def get_scene_bounds(self, obstacles: list, path: QPMI, drone_trajectories:list, padding=2):
        """Returns [xmin, xmax, ymin, ymax, zmin, zmax] for the scene and the paths,
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
        
        for drone_traj in drone_trajectories:
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
        house_index = None
        for o in obstacles:
            if isinstance(o, ImportedMeshObstacle):
                house_index = obstacles.index(o)
        obs_meshes = [o.mesh for o in obstacles]
        tri_obs_meshes = [trimesh.Trimesh(vertices=tri_to_enu(o.verts_packed().cpu().numpy().T).T, faces=tri_to_enu(o.faces_packed().cpu().numpy().T).T) for o in obs_meshes]
        meshes = []
        room = None   
        house = None 
        for i, mesh in enumerate(tri_obs_meshes):
            # If last mesh, it is the room, do not include in meshes
            if i == len(tri_obs_meshes) - 1:
                room = pv.wrap(mesh)
                continue
            if i == house_index:
                house = pv.wrap(mesh)
                continue
            pv_mesh = pv.wrap(mesh)
            meshes.append(pv_mesh)
        return meshes, room, house
    
    def plot_scene_and_trajs(self, save_path=None, azimuth=90, elevation=None, see_from_plane=None, scene="none"): 
        self.plotter.add_mesh(pv.Cube(bounds=self.scaled_bounds), opacity=0.0)
        backface_params = dict(opacity=0.0) #  To see through outer walls of enclosing room 

        opacity=1.0
        if self.force_transparency:
            opacity = 0.07
            azimuth = 0 #Hacky fix to get the right view for the cave scenario as its the only one where we use force transparency...
            self.plotter.add_mesh(self.room_mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = opacity) #Hacky fix to not plot the last obstacle in room colors..


        # Add all obstacles
        for i, mesh in enumerate(self.meshes):
            if i == 0:
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, label="Obstacles", smooth_shading=False,backface_params=backface_params, opacity=opacity)
            else:
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity=opacity)

        # Add the room, only plot if not house scenario
        if self.room_mesh != None and self.house_mesh == None and not self.force_transparency: #hacky fix here as well
            self.plotter.add_mesh(self.room_mesh, color=self.room_color, show_edges=False, backface_params=backface_params)
        
        if self.house_mesh != None:
            self.plotter.add_mesh(self.house_mesh, color=self.room_color, show_edges=False, opacity=0.2, backface_params=backface_params)

        # Add each drone trajectory
        for i, (key, drone_traj) in enumerate(self.drone_trajectories.items()):
            #color = self.get_rew_color_ # TODO
            spline = pv.Spline(drone_traj, 1000)
            # To color the paths taken by reward we add scalars to the spline and plot colorbar between min and max reward
            spline['cumulative_reward'] = self.cum_rewards[key]   
            if i == 0:
                self.plotter.add_mesh(spline, line_width=8, label="Drone Trajectories", scalars='cumulative_reward', cmap=self.cmap, clim=self.clim, flip_scalars=self.flip_scalars, scalar_bar_args=self.scalar_bar_args)
            else:
                self.plotter.add_mesh(spline, line_width=8, scalars='cumulative_reward', cmap=self.cmap, clim=self.clim, flip_scalars=self.flip_scalars, scalar_bar_args=self.scalar_bar_args)

        self.plotter.add_points(self.quadratic_path, color=self.path_color, point_size=10, label="Path", render_points_as_spheres=True)
        self.plotter.show_grid(**self.grid_kw_args)

        # Custom legend
        offset_x = 800
        offset_y = 700
        offset_between = 100
        legend_pos = [self.plotter.window_size[0] - offset_x, self.plotter.window_size[1] - offset_y]

        self.plotter.add_text("Obstacles", position=legend_pos, font_size=40, color=self.obstacles_color, font='times')
        self.plotter.add_text("Path", position=[legend_pos[0], legend_pos[1] - offset_between], font_size=40, color=self.path_color, font='times')
        #self.plotter.add_text("Drone Trajectory", position=[legend_pos[0], legend_pos[1] - 2*offset_between], font_size=40, color=drone_color, font='times')

        # Camera stuff
        self.plotter.camera.zoom(0.9) # Zoom a bit out to include axes from corner views
        
        if see_from_plane == None:
            self.plotter.camera.azimuth = azimuth # Controls the rotation of the camera around the scene, + is anti-clockwise rotation about scene center
        if elevation is not None:
            self.plotter.camera.elevation = elevation # Controls the angle of the camera above the scene, + makes you see the scene from higher above

        if see_from_plane == "xy":
            self.plotter.view_xy()
        elif see_from_plane == "xz":
            self.plotter.view_xz(negative=True)
        elif see_from_plane == "yz":
            self.plotter.view_yz()
        #if self.nosave: self.plotter.show()
        self.plotter.show()
        if save_path and not self.nosave:
            self.plotter.screenshot(save_path, scale=40, window_size = [1000, 1000])



if __name__ == "__main__":
    pass