import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fminbound
from mpl_toolkits.mplot3d import Axes3D

from typing import Tuple



#This is done to make the import below work However should work by itself #TODO low priority
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

from gym_quad.utils.house_paths import house_paths

"""Path generation using Quadratic Piecewise Membership Functions (QPMI)"""
class QPMI():
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.wp_idx = 0
        self.us = self._calculate_us()
        self.length = self.us[-1]
        self.calculate_quadratic_params()
    
    def _calculate_us(self):
        diff = np.diff(self.waypoints, axis=0)
        seg_lengths = np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))
        return np.array([0,*seg_lengths[:]])
    
    def calculate_quadratic_params(self):
        self.x_params = []
        self.y_params = []
        self.z_params = []
        for n in range(1, len(self.waypoints)-1):
            wp_prev = self.waypoints[n-1]
            wp_n = self.waypoints[n]
            wp_next = self.waypoints[n+1]
            
            u_prev = self.us[n-1]
            u_n = self.us[n]
            u_next = self.us[n+1]

            U_n = np.vstack([np.hstack([u_prev**2, u_prev, 1]),
                           np.hstack([u_n**2, u_n, 1]),
                           np.hstack([u_next**2, u_next, 1])])
            x_params_n = np.linalg.inv(U_n).dot(np.array([wp_prev[0], wp_n[0], wp_next[0]]))
            y_params_n = np.linalg.inv(U_n).dot(np.array([wp_prev[1], wp_n[1], wp_next[1]]))
            z_params_n = np.linalg.inv(U_n).dot(np.array([wp_prev[2], wp_n[2], wp_next[2]]))

            self.x_params.append(x_params_n)
            self.y_params.append(y_params_n)
            self.z_params.append(z_params_n)


    def get_u_index(self, u):
        n = 0
        while n < len(self.us) - 1:
            if u <= self.us[n+1]:
                break
            else:
                n += 1
        return n


    def calculate_ur(self, u):
        n = self.get_u_index(u)
        # ur = (u - self.us[n]) / (self.us[n+1] - self.us[n])
        if n == len(self.us) - 1:  # if u is at the end
            return 1.0
        try:
            ur = (u - self.us[n]) / (self.us[n+1] - self.us[n])
        except IndexError:
            print("n:", n)
            print("u:", u)
            print("us:", self.us)
            ur = (u - self.us[n]) / (self.us[n+1] - self.us[n])
        return ur


    def calculate_uf(self, u):
        n = self.get_u_index(u)
        if n == len(self.us) - 1:  # if u is at the end
            return 0.0
        uf = (self.us[n+1]-u)/(self.us[n+1] - self.us[n])
        return uf        


    def __call__(self, u):
        if u >= self.length: #Ensure that the path does not go beyond the last waypoint
            u = self.length
        if u >= self.us[0] and u <= self.us[1]: # first stretch
            ax = self.x_params[0][0]
            ay = self.y_params[0][0]
            az = self.z_params[0][0]
            bx = self.x_params[0][1]
            by = self.y_params[0][1]
            bz = self.z_params[0][1]
            cx = self.x_params[0][2]
            cy = self.y_params[0][2]
            cz = self.z_params[0][2]
            
            x = ax*u**2 + bx*u + cx
            y = ay*u**2 + by*u + cy
            z = az*u**2 + bz*u + cz

        elif u >= self.us[-2]-0.001 and u <= self.us[-1]: # last stretch
            ax = self.x_params[-1][0]
            ay = self.y_params[-1][0]
            az = self.z_params[-1][0]
            bx = self.x_params[-1][1]
            by = self.y_params[-1][1]
            bz = self.z_params[-1][1]
            cx = self.x_params[-1][2]
            cy = self.y_params[-1][2]
            cz = self.z_params[-1][2]
            
            x = ax*u**2 + bx*u + cx
            y = ay*u**2 + by*u + cy
            z = az*u**2 + bz*u + cz

        else: # else we are in the intermediate waypoints and we use membership functions to calc polynomials
            n = self.get_u_index(u)
            ur = self.calculate_ur(u)
            uf = self.calculate_uf(u)
            
            ax1 = self.x_params[n-1][0]
            ay1 = self.y_params[n-1][0]
            az1 = self.z_params[n-1][0]
            bx1 = self.x_params[n-1][1]
            by1 = self.y_params[n-1][1]
            bz1 = self.z_params[n-1][1]
            cx1 = self.x_params[n-1][2]
            cy1 = self.y_params[n-1][2]
            cz1 = self.z_params[n-1][2]
            
            x1 = ax1*u**2 + bx1*u + cx1
            y1 = ay1*u**2 + by1*u + cy1
            z1 = az1*u**2 + bz1*u + cz1

            ax2 = self.x_params[n][0]
            ay2 = self.y_params[n][0]
            az2 = self.z_params[n][0]
            bx2 = self.x_params[n][1]
            by2 = self.y_params[n][1]
            bz2 = self.z_params[n][1]
            cx2 = self.x_params[n][2]
            cy2 = self.y_params[n][2]
            cz2 = self.z_params[n][2]
            
            x2 = ax2*u**2 + bx2*u + cx2
            y2 = ay2*u**2 + by2*u + cy2
            z2 = az2*u**2 + bz2*u + cz2

            x = ur*x2 + uf*x1
            y = ur*y2 + uf*y1
            z = ur*z2 + uf*z1

        return np.array([x,y,z])


    def calculate_gradient(self, u):
        if u >= self.us[0] and u <= self.us[1]: # first stretch
            ax = self.x_params[0][0]
            ay = self.y_params[0][0]
            az = self.z_params[0][0]
            bx = self.x_params[0][1]
            by = self.y_params[0][1]
            bz = self.z_params[0][1]
            
            dx = ax*u*2 + bx
            dy = ay*u*2 + by
            dz = az*u*2 + bz
        elif u >= self.us[-2]: # last stretch
            ax = self.x_params[-1][0]
            ay = self.y_params[-1][0]
            az = self.z_params[-1][0]
            bx = self.x_params[-1][1]
            by = self.y_params[-1][1]
            bz = self.z_params[-1][1]
            
            dx = ax*u*2 + bx
            dy = ay*u*2 + by
            dz = az*u*2 + bz
        else: # else we are in the intermediate waypoints and we use membership functions to calc polynomials
            n = self.get_u_index(u)
            ur = self.calculate_ur(u)
            uf = self.calculate_uf(u)
            
            ax1 = self.x_params[n-1][0]
            ay1 = self.y_params[n-1][0]
            az1 = self.z_params[n-1][0]
            bx1 = self.x_params[n-1][1]
            by1 = self.y_params[n-1][1]
            bz1 = self.z_params[n-1][1]
            
            dx1 = ax1*u*2 + bx1
            dy1 = ay1*u*2 + by1
            dz1 = az1*u*2 + bz1

            ax2 = self.x_params[n][0]
            ay2 = self.y_params[n][0]
            az2 = self.z_params[n][0]
            bx2 = self.x_params[n][1]
            by2 = self.y_params[n][1]
            bz2 = self.z_params[n][1]
            
            dx2 = ax2*u*2 + bx2
            dy2 = ay2*u*2 + by2
            dz2 = az2*u*2 + bz2

            dx = ur*dx2 + uf*dx1
            dy = ur*dy2 + uf*dy1
            dz = ur*dz2 + uf*dz1
        return np.array([dx,dy,dz])
    

    def calculate_acceleration(self, u):
        """
        Calculate the acceleration of the path. in other words, the derivative of the tangent vector.
        """
        if u >= self.us[0] and u <= self.us[1]: # first stretch
            ax = self.x_params[0][0]
            ay = self.y_params[0][0]
            az = self.z_params[0][0]
            
            ddx = ax*2
            ddy = ay*2
            ddz = az*2

        elif u >= self.us[-2]: # last stretch
            ax = self.x_params[-1][0]
            ay = self.y_params[-1][0]
            az = self.z_params[-1][0]
            
            ddx = ax*2
            ddy = ay*2
            ddz = az*2

        else: # else we are in the intermediate waypoints and we use membership functions to calc polynomials
            n = self.get_u_index(u)
            ur = self.calculate_ur(u)
            uf = self.calculate_uf(u)
            
            ax1 = self.x_params[n-1][0]
            ay1 = self.y_params[n-1][0]
            az1 = self.z_params[n-1][0]
            
            ddx1 = ax1*2
            ddy1 = ay1*2
            ddz1 = az1*2

            ax2 = self.x_params[n][0]
            ay2 = self.y_params[n][0]
            az2 = self.z_params[n][0]
            
            ddx2 = ax2*2
            ddy2 = ay2*2
            ddz2 = az2*2

            ddx = ur*ddx2 + uf*ddx1
            ddy = ur*ddy2 + uf*ddy1
            ddz = ur*ddz2 + uf*ddz1
        
        return np.array([ddx,ddy,ddz])
    

    def calculate_vectors(self, u: float) -> Tuple[np.array, np.array, np.array]:
        """
        Calculate path describing vectors at point u.

        Parameters:
        ----------
        u : float
            Distance along the path from the beginning of the path

        Returns:
        -------
        t_hat : np.array
            Unit tangent vector
        n_hat : np.array
            Unit normal vector - perpendicular to the tangent in the osculating plane
        b_hat : np.array
            Unit binormal vector - perpendicular to both tangent vector and normal vector
        """

        dp = self.calculate_gradient(u)
        ddp = self.calculate_acceleration(u)

        t_hat = dp / np.linalg.norm(dp)
        n_hat = ddp / np.linalg.norm(ddp)
        b_hat = np.cross(t_hat, n_hat) / np.linalg.norm(np.cross(t_hat, n_hat))
        n_hat = - np.cross(t_hat, b_hat) / np.linalg.norm(np.cross(t_hat, b_hat))

        return t_hat, n_hat, b_hat


    def get_direction_angles(self, u):
        dx, dy, dz = self.calculate_gradient(u)[:]
        azimuth = np.arctan2(dy, dx)
        elevation = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        return azimuth, elevation
    

    def get_closest_u(self, position, wp_idx, margin = 2.0):
        x1 = self.us[wp_idx] - margin
        x2 = self.us[wp_idx+1] + margin if wp_idx < len(self.us) - 2 else self.length
        output = fminbound(lambda u: np.linalg.norm(self(u) - position), 
                        full_output=0, x1=x1, x2=x2, xtol=1e-6, maxfun=500)
        return output


    def get_closest_position(self, position, wp_idx):
        return self(self.get_closest_u(position,wp_idx))


    def get_endpoint(self):
        return self(self.length)

    def get_lookahead_point(self, position, lookahead_distance,wp_idx):
        '''
        Calculate the position on the path that is lookahead_distance from the given position
        '''
        u = self.get_closest_u(position, wp_idx)
        if u + lookahead_distance > self.length:
            u_lookahead = self.length
        else:
            u_lookahead = u + lookahead_distance
        return self(u_lookahead)


    def plot_path(self, wps_on=True, leave_out_first_wp=True):
        u = np.linspace(self.us[0], self.us[-1], 10000)
        quadratic_path = []
        for du in u:
            quadratic_path.append(self(du))
            self.get_direction_angles(du)
        quadratic_path = np.array(quadratic_path)
        ax = plt.axes(projection='3d')
        ax.plot3D(xs=quadratic_path[:,0], ys=quadratic_path[:,1], zs=quadratic_path[:,2], color="#3388BB", label="Path", linestyle="dotted")
        if wps_on:
            for i, wp in enumerate(self.waypoints):
                if i == 0 and leave_out_first_wp: continue
                if i == 1: ax.scatter3D(*wp, color="#EE6666", label="Waypoints")
                else: ax.scatter3D(*wp, color="#EE6666")
        
        #Turn off ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # ax.set_xlabel(xlabel=r"$x_w$ [m]", fontsize=14)
        # ax.set_ylabel(ylabel=r"$y_w$ [m]", fontsize=14)
        # ax.set_zlabel(zlabel=r"$z_w$ [m]", fontsize=14)
        # ax.set_xlabel(xlabel="x [m]")
        # ax.set_ylabel(ylabel="y [m]")
        # ax.set_zlabel(zlabel="z [m]")
        return ax

#Helper fcn for the house scenario in generate_random_waypoints
def remove_first_if_duplicate(path, previous_path):
    if path and previous_path and np.array_equal(path[0], previous_path[-1]):
        return path[1:]
    return path

def concatenate_paths(*paths):
    concatenated_path = []
    for i, path in enumerate(paths):
        if i > 0:
            path = remove_first_if_duplicate(path, paths[i-1])
        concatenated_path.extend(path)
    return concatenated_path

def generate_random_waypoints(nwaypoints, scen, segmentlength=5, select_house_path = None, e_angle_range = (np.pi/3,np.pi/2)):
    '''
    Generate random waypoints for the path generation
    Input:
    nwaypoints: int, number of waypoints
    scen: str, the scenario for the waypoints
    segmentlength: float, the length of the segments
    select_house_path: int, the house path to select used in the house scenario
    e_angle_range: tuple, the range of the elevation angle used in the 3d_up and 3d_down scenarios
    '''
    
    
    waypoints = [np.array([0,0,0])]

    if scen == "house": 
        #waypoints.pop(0)  # remove the initial waypoint
        waypoints = []

        # Define the subpaths
        path_1 = house_paths["bedroom_to_kitchen"]          # Hard path

        path_2 = concatenate_paths(
            house_paths["leisure_room_to_corridor"],
            house_paths["corridor_to_living_room"],
            house_paths["living_room_to_entrance_hall"],
            house_paths["entrance_hall_to_guest_room"]
        )

        path_3 = concatenate_paths(
            house_paths["living_room_to_bedroom"],
            house_paths["bedroom_to_main_bedroom"],
            house_paths["main_bedroom_to_home"]
        )

        path_4 = concatenate_paths(
            house_paths["kitchen_to_home"],
            house_paths["home_to_leisure_room"],
            house_paths["leisure_room_to_corridor"]
        )

        path_5 = concatenate_paths(
            house_paths["entrance_hall_to_corridor"],
            house_paths["corridor_to_living_room"],
            house_paths["living_room_to_kitchen"]
        )

        path_6 = concatenate_paths(                         # Easy path
            house_paths["main_bedroom_to_corridor"],
            house_paths["corridor_to_guest_room"]
        )

        # Select one of the paths based on random choice or a specific selection
        paths = [path_1, path_2, path_3, path_4, path_5, path_6]
        selected_path = select_house_path - 1 if select_house_path is not None else np.random.randint(len(paths))
        waypoints.extend(map(np.array, paths[selected_path]))    
        
    elif scen =='squiggly_line_xy_plane':
        a_start_angle=np.random.uniform(-np.pi,np.pi)
        e_start_angle=np.random.uniform(-np.pi,np.pi) 
        distance = segmentlength
        for i in range(nwaypoints-1):
            azimuth = a_start_angle + np.random.uniform(-np.pi/4, np.pi/4)
            x = waypoints[i][0] + distance*np.cos(azimuth)
            y = waypoints[i][1] + distance*np.sin(azimuth)
            z = 0
            wp = np.array([x, y, z])
            waypoints.append(wp)

    elif scen=='line':
        #Line in x direction throws error if segmentlength is too small
        path_length = segmentlength*(nwaypoints-1)
        
        new_segmentlength = path_length/2
        nwaypoints = 3

        for i in range(nwaypoints-1):
            distance = new_segmentlength
            azimuth = np.random.uniform(-np.pi/4, np.pi/4)
            elevation = np.random.uniform(-np.pi/4, np.pi/4)
            x = waypoints[i][0] + distance*np.cos(azimuth)*np.cos(elevation)
            y = 0 
            z = 0 
            
            wp = np.array([x, y, z])
            waypoints.append(wp)

    elif scen == 'line_y':
        path_length = segmentlength*(nwaypoints-1)
        new_segmentlength = path_length/2
        nwaypoints = 3

        for i in range(nwaypoints-1):
            distance = new_segmentlength
            azimuth = np.random.uniform(-np.pi/4, np.pi/4)
            elevation = np.random.uniform(-np.pi/4, np.pi/4)
            x = 0 
            y = waypoints[i][1] + distance*np.sin(azimuth)*np.cos(elevation)
            z = 0 
            
            wp = np.array([x, y, z])
            waypoints.append(wp)

    elif scen=='line_up':
        path_length = segmentlength*(nwaypoints-1)
        new_segmentlength = path_length/2
        nwaypoints = 3

        for i in range(nwaypoints-1):
            distance = new_segmentlength
            azimuth = np.random.uniform(-np.pi/4, np.pi/4)
            elevation = np.random.uniform(-np.pi/4, np.pi/4)
            x = 0 
            y = 0 
            z = waypoints[i][0] + distance*np.cos(azimuth)*np.cos(elevation)
            
            wp = np.array([x, y, z])
            waypoints.append(wp)

    elif scen=='xy_line':
        a_start_angle=np.random.uniform(-np.pi,np.pi)
        azimuth = a_start_angle+np.random.uniform(-np.pi/4, np.pi/4)
        elevation = np.random.uniform(-np.pi/4, np.pi/4)
        distance = segmentlength
        for i in range(nwaypoints-1):
            x = waypoints[i][0] + distance*np.cos(azimuth)
            y = waypoints[i][1] + distance*np.sin(azimuth)
            z = 0
            wp = np.array([x, y, z])
            waypoints.append(wp)

    elif scen =='3d_new':
        a_start_angle=np.random.uniform(-np.pi,np.pi)
        e_start_angle=np.random.uniform(-np.pi,np.pi) 
        e_start_angle=0
        distance = segmentlength
        for i in range(nwaypoints-1):
            azimuth = a_start_angle + np.random.uniform(-np.pi/4, np.pi/4)
            elevation = e_start_angle + np.random.uniform(-np.pi/4, np.pi/4)
            x = waypoints[i][0] + distance*np.cos(azimuth)*np.cos(elevation)
            y = waypoints[i][1] + distance*np.sin(azimuth)*np.cos(elevation)
            z = waypoints[i][2] + distance*np.sin(elevation)
            wp = np.array([x, y, z])
            waypoints.append(wp)

    elif scen =='3d_up':
        e_angle_min = e_angle_range[0]
        e_angle_max = e_angle_range[1]
        a_start_angle=np.random.uniform(-np.pi,np.pi)
        e_start_angle=np.random.uniform(e_angle_min, e_angle_max) 
        distance = segmentlength
        for i in range(nwaypoints-1):
            azimuth = a_start_angle + np.random.uniform(-np.pi/4, np.pi/4)
            elevation = max(np.pi/4, min(e_angle_max, e_start_angle + np.random.uniform(-np.pi/8, np.pi/8)))  # Clamp the elevation
            x = waypoints[i][0] + distance*np.cos(azimuth)*np.cos(elevation)
            y = waypoints[i][1] + distance*np.sin(azimuth)*np.cos(elevation)
            z = waypoints[i][2] + distance*np.sin(elevation)
            wp = np.array([x, y, z])
            waypoints.append(wp)

    elif scen =='3d_down':
        e_angle_min = -e_angle_range[0]
        e_angle_max = -e_angle_range[1]
        a_start_angle=np.random.uniform(-np.pi,np.pi)
        e_start_angle = np.random.uniform(e_angle_min, e_angle_max) 
        distance = segmentlength
        for i in range(nwaypoints-1):
            azimuth = a_start_angle + np.random.uniform(-np.pi/4, np.pi/4)
            elevation = max(e_angle_max, min(-np.pi/4, e_start_angle + np.random.uniform(-np.pi/8, np.pi/8)))  # Clamp the elevation
            x = waypoints[i][0] + distance*np.cos(azimuth)*np.cos(elevation)
            y = waypoints[i][1] + distance*np.sin(azimuth)*np.cos(elevation)
            z = waypoints[i][2] + distance*np.sin(elevation)
            wp = np.array([x, y, z])
            waypoints.append(wp)

    elif scen=='helix':
        waypoints = []
        n_waypoints = 26
        angles = np.linspace(0, 4*np.pi, n_waypoints)
        radius = 11
        for i, angle in enumerate(angles):
            x = radius*np.cos(angle)
            y = radius*np.sin(angle)
            z = (i - n_waypoints) + n_waypoints/2
            wp = np.array([x, y, z])
            waypoints.append(wp)

    return np.array(waypoints)


if __name__ == "__main__":
    wps = np.array([np.array([0,0,0]), np.array([20,10,15]), np.array([50,20,20]), np.array([80,20,15]), np.array([90,50,20]), np.array([80,80,15]), np.array([50,80,20]), np.array([20,60,15]), np.array([20,40,10]), np.array([0,0,0])])
    wps = np.array([np.array([0,0,0]), np.array([20,10,15]), np.array([50,20,20]), np.array([80,20,40]), np.array([90,50,50]),
                    np.array([80,80,60]), np.array([50,80,20]), np.array([20,60,15]), np.array([20,40,10]), np.array([0,0,0])])
    #wps = np.array([np.array([0,0,0]), np.array([20,25,22]), np.array([50,40,30]), np.array([90,55,60]), np.array([130,95,110]), np.array([155,65,86])])
    #wps = generate_random_waypoints(10)
    wps = generate_random_waypoints(nwaypoints=6,segmentlength=5, scen='house',select_house_path=6)
    path = QPMI(wps)
   
    # point = path(20)
    # azi, ele = path.get_direction_angles(20)
    # vec_x = point[0] + 20*np.cos(azi)*np.cos(ele)
    # vec_y = point[1] + 20*np.sin(azi)*np.cos(ele)
    # vec_z = point[2] - 20*np.sin(ele)
    
    ax = path.plot_path()
    ax.plot3D(xs=wps[:,0], ys=wps[:,1], zs=wps[:,2], linestyle="dashed", color="#33bb5c")
    #ax.plot3D(xs=[point[0],vec_x], ys=[point[1],vec_y], zs=[point[2], vec_z])
    #ax.scatter3D(*point)
    for i, wp in enumerate(wps):
        if i == 0:
            ax.scatter3D(*wp, color="g")
        else:
            ax.scatter3D(*wp, color="r")
    ax.legend(["QPMI path", "Linear piece-wise path", "Waypoints"], fontsize=14)
    plt.rc('lines', linewidth=3)
    ax.set_xlabel(xlabel="East [m]", fontsize=14)
    ax.set_ylabel(ylabel="North [m]", fontsize=14)
    ax.set_zlabel(zlabel="Up [m]", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.show()
