"""
From https://github.com/ankurhanda/simkinect
Citations:
[1] Handa, Ankur and Whelan, Thomas and McDonald, John and Davison, Andrew J. "A benchmark for RGB-D visual odometry, 3D reconstruction and SLAM." ICRA (2014).
[2] Jonathan T. Barron and Jitendra Malik. "Intrinsic Scene Properties from a Single RGB-D Image." CVPR (2013).
[3] Bohg, Jeannette and Romero, Javier and Herzog, Alexander and Schaal, Stefan. "Robot arm pose estimation through pixel-wise part classification." ICRA (2014).
"""
import numpy as np 
import cv2 
from scipy.interpolate import griddata
from scipy import ndimage
import os
import matplotlib.pyplot as plt


def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp
    

def filterDisp(disp, dot_pattern_, invalid_disp_):

    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp


def add_noise(depth, dot_pattern_):
    scale_factor  = 100     # converting depth from mm to cm 
    focal_length  = 480.0   # focal length of the camera used 
    baseline_m    = 0.095   # baseline in m, Intel realsense D435i has 95mm baseline 
    invalid_disp_ = 99999999.9

    depth = depth.astype('float') / 1000.0
    depth_interp = add_gaussian_shifts(depth)
    disp_= focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0)/8.0
    out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)
    depth = focal_length * baseline_m / out_disp
    depth[out_disp == invalid_disp_] = 0 
    noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 
    noisy_depth = noisy_depth * 1000.0 
    noisy_depth = noisy_depth.astype('uint16')

    return noisy_depth



if __name__ == "__main__":

    depth_imgs = os.listdir("./depth_imgs")
    print(depth_imgs)

    # reading the image directly in gray with 0 as input 
    dot_pattern_ = cv2.imread("./cam_noise_data/kinect-pattern_3x3.png", 0)
    #dot_pattern_ = cv2.imread("./cam_noise_data/realsense_pattern.png", 0)

    scale_factor  = 100     # converting depth from mm to cm 
    focal_length  = 480.0   # focal length of the camera used 
    baseline_m    = 0.095   # baseline in m, Intel realsense D435i has 95mm baseline 
    invalid_disp_ = 99999999.9

    for depth_im in depth_imgs:

        depth_uint16 = cv2.imread(f'./depth_imgs/{depth_im}', cv2.IMREAD_UNCHANGED)
        print(depth_uint16)
        h, w = depth_uint16.shape 

        # dividing to get depth in meters 
        depth = depth_uint16.astype('float') / 1000.0
        print(depth)

        depth_interp = add_gaussian_shifts(depth)

        disp_= focal_length * baseline_m / (depth_interp + 1e-10)
        depth_f = np.round(disp_ * 8.0)/8.0

        out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

        depth = focal_length * baseline_m / out_disp
        depth[out_disp == invalid_disp_] = 0 
        
        # The depth here needs to converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 

        noisy_depth = noisy_depth * 1000.0 
        noisy_depth = noisy_depth.astype('uint16')

        comparison = np.hstack((depth_uint16, noisy_depth))

        # Save with matplotlib using the magma colormap
        plt.imshow(comparison, cmap='magma')
        plt.savefig(f'./{depth_im.split(".")[0]}_noisy.png')

