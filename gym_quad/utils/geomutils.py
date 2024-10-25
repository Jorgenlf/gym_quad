import numpy as np
import torch
from trimesh.transformations import rotation_matrix

### Helper functions to transform between ENU and trimesh/pytorch3D coordinate systems
def enu_to_pytorch3d(enu_position: torch.Tensor,device) -> torch.Tensor:
    '''ENU is x-east, y-north, z-up. 
    Pytorch3D is x-left, y-up, z-forward. 
    This function converts from ENU to Pytorch3D coordinate system.'''
    return torch.tensor([enu_position[1], enu_position[2], enu_position[0]],device=device)

def pytorch3d_to_enu(pytorch3d_position: torch.Tensor,device) -> torch.Tensor:
    '''ENU is x-east, y-north, z-up. 
    Pytorch3D is x-left, y-up, z-forward. 
    This function converts from ENU to Pytorch3D coordinate system.'''
    return torch.tensor([pytorch3d_position[2], pytorch3d_position[0], pytorch3d_position[1]],device=device)
###

def enu_to_tri(enu_pos: np.ndarray):
    '''ENU is x-east, y-north, z-up.
    Trimesh is similar to pytroch3D, x-left, y-up, z-forward.'''
    return np.array([enu_pos[1], enu_pos[2], enu_pos[0]])

def tri_to_enu(tri_pos: np.ndarray):
    '''ENU is x-east, y-north, z-up.
    Trimesh is similar to pytroch3D, x-left, y-up, z-forward.'''
    return np.array([tri_pos[2], tri_pos[0], tri_pos[1]])

def tri_Rotmat(roll_enu,pitch_enu,yaw_enu):
    '''Creates a rotationmatrix for changing the attidude of a mesh given the roll, pitch, yaw in ENU coordinates.
    does yzx rotation as trimesh is x-left(north), y-up, z-forward(east). matches ENU zxy being x-east, y-north, z-up.'''
    body_y_tri = np.array([0, 1, 0]) 
    body_z_tri = np.array([0, 0, 1]) 
    body_x_tri = np.array([1, 0, 0]) 

    rot_about_tri_y = rotation_matrix(yaw_enu, body_y_tri)
    rot_about_tri_z = rotation_matrix(roll_enu, body_z_tri)
    rot_about_tri_x = rotation_matrix(pitch_enu, body_x_tri)
    
    R_tri = np.dot(rot_about_tri_y, np.dot(rot_about_tri_z, rot_about_tri_x))
    return R_tri


def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def R(x, y, z):
    """
    Rotation matrix to world frame.

    Parameters:
    ----------
    x : np.array
        x-axis of coordinate frame expressed in world frame.
    y : np.array
        y-axis of coordinate frame expressed in world frame.
    z : np.array
        z-axis of coordinate frame expressed in world frame.
    
    Returns:
    -------
    R : np.array
        Rotation matrix from the coordinate frame expressed by [x, y, x] to world frame.
    """
    return np.transpose(np.vstack((x, y, z)))
    # return np.vstack([
    #     np.hstack([x[0], y[0], z[0]]),
    #     np.hstack([x[1], y[1], z[1]]),
    #     np.hstack([x[2], y[2], z[2]])])

def R2Euler(R):
    phi = np.arctan2(R[2, 1], R[2, 2])
    theta = -np.arctan(R[2, 0] / np.sqrt(1 - R[2, 0]**2))
    psi = np.arctan2(R[1, 0], R[0, 0])

    return np.array([phi, theta, psi])



def ssa(angle):
    """ Returns the smallest signed angle in the range [-pi, pi]."""
    return ((angle + np.pi) % (2*np.pi)) - np.pi


def Rzyx(phi, theta, psi):
    '''
    input: phi, theta, psi of the body frame relative to the world frame
    Rotation matrix from the body frame to the world frame.
    '''
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth]),
        np.hstack([spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi]),
        np.hstack([-sth, cth*sphi, cth*cphi])])


def Tzyx(phi, theta, psi):
    sphi = np.sin(phi)
    tth = np.tan(theta)
    cphi = np.cos(phi)
    cth = np.cos(theta)

    return np.vstack([
        np.hstack([1, sphi*tth, cphi*tth]), 
        np.hstack([0, cphi, -sphi]),
        np.hstack([0, sphi/cth, cphi/cth])])


def J(eta):
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    R = Rzyx(phi, theta, psi)
    T = Tzyx(phi, theta, psi)
    zero = np.zeros((3,3))

    return np.vstack([
        np.hstack([R, zero]),
        np.hstack([zero, T])])


def S_skew(a):
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]

    return np.vstack([
        np.hstack([0, -a3, a2]),
        np.hstack([a3, 0, -a1]),
        np.hstack([-a2, a1, 0])])


def _H(r):
    I3 = np.identity(3)
    zero = np.zeros((3,3))

    return np.vstack([
        np.hstack([I3, np.transpose(S_skew(r))]),
        np.hstack([zero, I3])])


def move_to_CO(A_CG, r_g):
    H = _H(r_g)
    Ht = np.transpose(H)
    A_CO = Ht.dot(A_CG).dot(H)
    return A_CO

def vee_map(skew_matrix):
    #Maps a skew-symmetric matrix to a vector
    m21 = skew_matrix[2, 1]
    m10 = skew_matrix[1, 0]
    m02 = skew_matrix[0, 2]
    # Compute the vee map
    vee_map = np.array([[m21, m02, m10]])
    return vee_map    

def Rz(psi):
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    return np.array([[cpsi, -spsi, 0],
                     [spsi, cpsi, 0],
                     [0, 0, 1]])

# def vee_map2(skew_matrix):
#     #Maps a skew-symmetric matrix to a vector Kulkarni version
#     vm = np.array([[-skew_matrix[1, 2], skew_matrix[0, 2], -skew_matrix[0, 1]]])
#     return vm