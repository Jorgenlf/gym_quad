import vtk
import pyvista as pv
import numpy as np
import torch
import trimesh
from gym_quad.objects.mesh_obstacles import SphereMeshObstacle, CubeMeshObstacle, Scene, ImportedMeshObstacle
from gym_quad.objects.QPMI import QPMI
from gym_quad.utils.geomutils import enu_to_tri, tri_to_enu
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# Define start and end colors and interpolate a colormap between them
start_color = '#FFFF00'  # Yellow
end_color = '#990099'    # Purple
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [start_color, end_color], N=256)


class Plotter3D: 
    def __init__(self, obstacles, path:QPMI, drone_traj: np.ndarray, initial_position:np.ndarray, scene:str="none", save=True):
        self.obstacles = obstacles
        self.path = path
        self.drone_traj = drone_traj
        self.initial_position = initial_position
        self.scene = scene
        self.save = save

        self.meshes, self.room_mesh, self.house_mesh = self.obstacles_to_pyvista_meshes(obstacles)
        self.quadratic_path = self.dash_path(self.get_path_as_arr(path))
        self.bounds, self.scaled_bounds = self.get_scene_bounds(drone_trajectories=[drone_traj], padding=0)

        self.plotter = pv.Plotter(window_size=[4000, 4000], 
                                  off_screen=save)
        
        # Plotting parameters
        self.drone_path_color = '#00BA38'   #TODO find suitable color - is it not suitable?
        self.path_color = '#4780ff'         # blueish
        self.obstacles_color = '#ff3400'    # redish
        self.room_color = '#f3f0ec'         # deserty/gray
        self.initial_pos_color = 'black'
        self.grid_kw_args = {
            'color': 'gray',
            'xtitle': 'x [m]',
            'ytitle': 'y [m]',
            'ztitle': 'z [m]',
            'font_size': 40,
            'padding': 0.0, 
            'font_family': 'times',
            'fmt':'%.0f',
            'ticks': 'outside',
            # axes_ranges just changes values on the axes, not the actual scene, do not use
        }
    
    def dash_path(self, path: np.ndarray, n=50):
        """Creates a new path that consists of every nth element of the original path"""
        new_path = np.ndarray((0,3))
        for i in range(0, len(path), n):
            new_path = np.vstack((new_path, path[i]))
        return new_path

    def get_path_as_arr(self, path: QPMI):
        """Converts QPMI object to numpy array"""
        u = np.linspace(path.us[0], path.us[-1], 10000)
        quadratic_path = []
        for du in u:
            quadratic_path.append(path(du))
            path.get_direction_angles(du)
        quadratic_path = np.array(quadratic_path)
        return quadratic_path
    
    
    def get_scene_bounds(self, drone_trajectories:list, padding=1):
        """Returns [xmin, xmax, ymin, ymax, zmin, zmax] for the scene and the paths,
            with padding [m] added to each dimension"""
        inf = 1000
        bounds = [inf, -inf, inf, -inf, inf, -inf]
    
        if self.obstacles != [] :
            obs_meshes = [o.mesh for o in self.obstacles]
            tri_obs_meshes = [trimesh.Trimesh(vertices=tri_to_enu(o.verts_packed().cpu().numpy().T).T, faces=tri_to_enu(o.faces_packed().cpu().numpy().T).T) for o in obs_meshes]
            for mesh in tri_obs_meshes:
                for point in mesh.vertices:
                    bounds[0] = min(bounds[0], point[0])
                    bounds[1] = max(bounds[1], point[0])
                    bounds[2] = min(bounds[2], point[1])
                    bounds[3] = max(bounds[3], point[1])
                    bounds[4] = min(bounds[4], point[2])
                    bounds[5] = max(bounds[5], point[2])
    
        if self.path != None:
            for point in self.path.waypoints:
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
        """Converts list of obstacles to list of pyvista meshes, also returns room mesh and house mesh if present. These are handled differently in the plotter"""
        house_index = None
        for o in obstacles:
            if isinstance(o, ImportedMeshObstacle): # Where "ImportedMeshObstacle" is the house, TODO change to "HouseObstacle" or something
                house_index = obstacles.index(o)

        obs_meshes = [o.mesh for o in obstacles]
        tri_obs_meshes = [trimesh.Trimesh(vertices=tri_to_enu(o.verts_packed().cpu().numpy().T).T, faces=tri_to_enu(o.faces_packed().cpu().numpy().T).T) for o in obs_meshes]
        
        obs_meshes = []
        room_mesh = None   
        house_mesh = None 
        for i, mesh in enumerate(tri_obs_meshes):
            # If last mesh, it is the room, do not include in meshes
            if i == len(tri_obs_meshes) - 1:
                room_mesh = pv.wrap(mesh)
                continue
            elif i == house_index:
                house_mesh = pv.wrap(mesh)
                continue
            else:
                pv_mesh = pv.wrap(mesh)
                obs_meshes.append(pv_mesh)

        return obs_meshes, room_mesh, house_mesh
    
    def plot_scene_and_trajs(self, save_path=None, azimuth=90,  only_scene=False, hv=1): # TODO could be split into multiple methods
        """
        Plots scene and drone trajectory in 3D using pyvista based on the scenario. House-scenarios are handled differently.
        View from plane removed bc. not needed in our report. If needed, se 'pl.view_xy()' etc.
        """
        # For normal scenarios, add invisible cube to the scaled bounds get cubical grid aroun dthe scene
        if self.scene not in ["house_easy", "house_hard", "house_easy_obstacles", "house_hard_obstacles"]:
            self.plotter.add_mesh(pv.Cube(bounds=self.scaled_bounds), opacity=0.0)
        backface_params = dict(opacity=0.0) # To see through outer walls of enclosing room 

        op = 1
        if self.scene == "cave":
            azimuth = 0
            op = 0.07
            self.plotter.add_mesh(self.room_mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = 0.07) 

        # Add all obstacles
        for i, mesh in enumerate(self.meshes):
            if i == 0: # Check index to only get one label. Needed bc. custom legends
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, label="Obstacles", smooth_shading=False,backface_params=backface_params, opacity = op)
            else:
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = op)

        # Add the room, only plot if not house scenario or cave scenario
        if self.room_mesh != None and self.house_mesh == None and not self.scene == "cave":
            self.plotter.add_mesh(self.room_mesh, color=self.room_color, show_edges=False, backface_params=backface_params)
        
        if self.house_mesh != None:
            self.plotter.add_mesh(self.house_mesh, color=self.room_color, show_edges=False, opacity=0.15)

        # Add drone traj and path
        spline = pv.Spline(self.drone_traj, 1000)
        self.plotter.add_mesh(spline, line_width=8, label="Drone Trajectory", color=self.drone_path_color)
        self.plotter.add_points(self.quadratic_path, color=self.path_color, point_size=20, label="Path", render_points_as_spheres=True)
        self.plotter.add_points(self.initial_position, color=self.initial_pos_color, point_size=30, label="Initial Position", render_points_as_spheres=True)

        # Camera stuff:
        if self.scene in ["house_easy", "house_easy_obstacles"]:
            # Offsets, angles and pos found manually...
            # Good viewing angle and camera position for house easy scenarios
            offset_x, offset_y = 2845,3570
            self.plotter.camera.position = (-4.963928349952159, -44.604219590185856, -9.031343052406486) 
            self.plotter.camera.azimuth = 33
            self.plotter.camera.elevation = 35
            self.plotter.camera.zoom(1.25)


        elif self.scene in ["house_hard", "house_hard_obstacles"]:
            # Offsets, angles and pos found manually...
            # VIEW 1:
            if hv == 1:
                offset_x, offset_y = 3200,3500
                self.plotter.camera.position = (-26.432109314195003, -1.5749345781232873, 3.0105659899075774) 

            # VIEW 2:
            elif hv == 2:
                offset_x, offset_y = 3200,3570
                self.plotter.camera.position = (8.296159959828376, -33.34163639867404, -2.393561657703904)
                self.plotter.camera.elevation = 25
                #self.plotter.camera.zoom(1.05)

            else: pass # Implement more views etc...
        
        else:
            offset_x, offset_y = 800, 700
            self.plotter.show_grid(**self.grid_kw_args) # Include grid only for non-house scenarios
            self.plotter.camera.zoom(0.9) 
            self.plotter.camera.azimuth = azimuth

        
        # Legends, positions are set in the scenario checks above
        offset_between = 100
        self.add_legends(offset_x, offset_y, only_scene=only_scene, offset_between=offset_between)
                
        cpos = self.plotter.show(return_cpos=True)
        #print(cpos) # For debugging camera position
        if save_path and self.save:
            self.plotter.screenshot(save_path, scale=40, window_size = [1000, 1000])

    def add_legends(self, offset_x, offset_y, offset_between=100, only_scene=False):
        legend_pos = [self.plotter.window_size[0] - offset_x, self.plotter.window_size[1] - offset_y]
        if self.scene not in ["house_easy", "house_hard"]: # Do not include obstacle legend in house easy and hard 
            self.plotter.add_text("Path", position=legend_pos, font_size=40, color=self.path_color, font='times')
            self.plotter.add_text("Obstacles", position=[legend_pos[0], legend_pos[1] - offset_between], font_size=40, color=self.obstacles_color, font='times')
        else:
            self.plotter.add_text("Path", position=[legend_pos[0], legend_pos[1] - offset_between], font_size=40, color=self.path_color, font='times')
        if not only_scene:
            self.plotter.add_text("Drone Trajectory", position=[legend_pos[0], legend_pos[1] - 2*offset_between], font_size=40, color=self.drone_path_color, font='times')
            self.plotter.add_text("Initial Position", position=[legend_pos[0], legend_pos[1] - 3*offset_between], font_size=40, color=self.initial_pos_color, font='times')

    def plot_only_scene(self, save_path=None, azimuth=90, elevation=None, see_from_plane=None):
        self.plot_scene_and_trajs(save_path=save_path, azimuth=azimuth, elevation=elevation, see_from_plane=see_from_plane, only_scene=True)


class Plotter3DMultiTraj(Plotter3D):
    def __init__(self, obstacles, path: QPMI, drone_trajs: dict, initial_position:np.ndarray, cum_rewards: dict, scene: str, save=True):
        super().__init__(obstacles = obstacles, 
                         path = path, 
                         drone_traj = drone_trajs[list(drone_trajs.keys())[0]],  # dummy, we plot all but initialize parent with first trajectory
                         initial_position = initial_position,              # dummy, not plotted in multi
                         scene = scene, 
                         save = save) 
        self.drone_trajs = drone_trajs
        self.cum_rewards = cum_rewards

        self.meshes, self.room_mesh, self.house_mesh = self.obstacles_to_pyvista_meshes(obstacles)
        self.quadratic_path = self.dash_path(self.get_path_as_arr(path))
        if self.scene in ["house_easy", "house_hard", "house_easy_obstacles", "house_hard_obstacles"]:
            self.padding = 2
        else:
            self.padding = 0
        self.bounds, self.scaled_bounds = self.get_scene_bounds(drone_trajectories=list(drone_trajs.values()), padding=self.padding)
        self.min_rew, self.max_rew = self.find_min_max_rewards(cum_rewards)

        # Additional lotting parameters spcific to multitraj
        self.clim = [self.min_rew, self.max_rew]
        self.cmap = custom_cmap
        self.flip_scalars = True # To invert the colorbar
        self.scalar_bar_args = self.set_scalar_bar_args() # Set scalar bar args based on scene, this is for the reward colorbar
    
    def set_scalar_bar_args(self):
        """Sets scalar bar args (reward colorbar) based on the scene type, house scenarios have horzontal bar, others vertical."""
        scalar_bar_args = {
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
        if self.scene in ["house_easy", "house_easy_obstacles"]:
            scalar_bar_args['vertical'] = False
            scalar_bar_args['position_x'] = 0.29
            scalar_bar_args['position_y'] = 0.02
            scalar_bar_args['width'] = 0.5
            scalar_bar_args['height'] = 0.08
            scalar_bar_args['n_labels'] = 2
            return scalar_bar_args
            pass
        elif self.scene in ["house_hard", "house_hard_obstacles"]:
            # handle both views with the same scalar bar
            scalar_bar_args['vertical'] = False
            scalar_bar_args['position_x'] = 0.2
            scalar_bar_args['position_y'] = 0.02
            scalar_bar_args['width'] = 0.5
            scalar_bar_args['height'] = 0.08
            scalar_bar_args['n_labels'] = 2
            return scalar_bar_args
                    
        return scalar_bar_args
    
    def find_min_max_rewards(self, cum_rewards: dict):
        min_reward = 100000
        max_reward = -100000
        for ep, rew in cum_rewards.items():
            min_reward = min(min_reward, rew)
            max_reward = max(max_reward, rew)
        return min_reward, max_reward
    
    def plot_scene_and_trajs(self, save_path=None, azimuth=90, only_scene=False, hv=1): 
        """
        Override of the plot_scene_and_trajs method in Plotter3D. 
        This method is used to plot multiple drone trajectories in the same scene and color with reward.
        A bit DRY but whatevs, need different view angles to include reward bar and so on
        """
        # For normal scenarios, add invisible cube to the scaled bounds get cubical grid aroun dthe scene
        if self.scene not in ["house_easy", "house_hard", "house_easy_obstacles", "house_hard_obstacles"]:
            self.plotter.add_mesh(pv.Cube(bounds=self.scaled_bounds), opacity=0.0)

        backface_params = dict(opacity=0.0) # To see through outer walls of enclosing room 
        
        op = 1
        if self.scene == "cave":
            azimuth = 0
            op = 0.07
            self.plotter.add_mesh(self.room_mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = op) 

        # Add all obstacles
        for i, mesh in enumerate(self.meshes):
            if i == 0: # Check index to only get one label. Needed bc. custom legends
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, label="Obstacles", smooth_shading=False,backface_params=backface_params, opacity = op)
            else:
                self.plotter.add_mesh(mesh, color=self.obstacles_color, show_edges=False, smooth_shading=False, backface_params=backface_params, opacity = op)

        # Add the room, only plot if not house scenario or cave scenario
        if self.room_mesh != None and self.house_mesh == None and not self.scene == "cave":
            self.plotter.add_mesh(self.room_mesh, color=self.room_color, show_edges=False, backface_params=backface_params)
        
        if self.house_mesh != None:
            self.plotter.add_mesh(self.house_mesh, color=self.room_color, show_edges=False, opacity=0.15)
        
        # Add each drone trajectory, done after camera stuff to get scalar bar args based on scene # TODO this could be a function or something
        for i, (key, drone_traj) in enumerate(self.drone_trajs.items()):
            #color = self.get_rew_color_ # TODO
            spline = pv.Spline(drone_traj, 1000)
            # To color the paths taken by reward we add scalars to the spline and plot colorbar between min and max reward
            spline['cumulative_reward'] = self.cum_rewards[key]   
            if i == 0:
                self.plotter.add_mesh(spline, line_width=8, label="Drone Trajectories", scalars='cumulative_reward', cmap=self.cmap, clim=self.clim, flip_scalars=self.flip_scalars, scalar_bar_args=self.scalar_bar_args)
            else:
                self.plotter.add_mesh(spline, line_width=8, scalars='cumulative_reward', cmap=self.cmap, clim=self.clim, flip_scalars=self.flip_scalars, scalar_bar_args=self.scalar_bar_args)

        self.plotter.add_points(self.quadratic_path, color=self.path_color, point_size=20, label="Path", render_points_as_spheres=True)

        # Camera stuff:
        if self.scene in ["house_easy", "house_easy_obstacles"]:
            # Offsets, angles and pos found manually...
            # Good viewing angle and camera position for house easy scenarios
            offset_x, offset_y = 2845,3570
            self.plotter.camera.position = (-4.963928349952159, -44.604219590185856, -9.031343052406486) 
            self.plotter.camera.azimuth = 33
            self.plotter.camera.elevation = 35
            self.plotter.camera.zoom(1.25)


        elif self.scene in ["house_hard", "house_hard_obstacles"]:
            # Offsets, angles and pos found manually...
            # VIEW 1:
            if hv == 1:
                offset_x, offset_y = 3200,3500
                self.plotter.camera.position = (-26.432109314195003, -1.5749345781232873, 3.0105659899075774) 

            # VIEW 2:
            elif hv == 2:
                offset_x, offset_y = 3200,3570
                self.plotter.camera.position = (8.296159959828376, -33.34163639867404, -2.393561657703904)
                self.plotter.camera.elevation = 25
                #self.plotter.camera.zoom(1.05)

            else: pass # Implement more views etc...
        
        else:
            offset_x, offset_y = 800, 700
            self.plotter.show_grid(**self.grid_kw_args) # Include grid only for non-house scenarios
            self.plotter.camera.zoom(0.9) 
            self.plotter.camera.azimuth = azimuth

    
        # Legends, positions are set in the scenario checks above
        offset_between = 100
        legend_pos = [self.plotter.window_size[0] - offset_x, self.plotter.window_size[1] - offset_y]
        if self.scene not in ["house_easy", "house_hard"]: # Do not include obstacle legend in house easy and hard 
            self.plotter.add_text("Path", position=legend_pos, font_size=40, color=self.path_color, font='times')
            self.plotter.add_text("Obstacles", position=[legend_pos[0], legend_pos[1] - offset_between], font_size=40, color=self.obstacles_color, font='times')
        else:
            self.plotter.add_text("Path", position=[legend_pos[0], legend_pos[1] - offset_between], font_size=40, color=self.path_color, font='times')

        #if self.nosave: self.plotter.show()
        self.plotter.show()
        if save_path and self.save:
            self.plotter.screenshot(save_path, scale=40, window_size = [1000, 1000])



if __name__ == "__main__":
    pass