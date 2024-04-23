import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


folder = "data/realsense_data_v2/realsense_examples"
output_folder = "data/realsense_data_v2/realsense_examples/rgb_depth_comparison"

color_imgs = os.listdir(f"{folder}/color")
depth_imgs = os.listdir(f"{folder}/depth")

plt.style.use('ggplot')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)


# Plot rgb image and depth image with rgb resized to depth size for all image numbers
for i in range(len(color_imgs)):
    color_im = color_imgs[i]
    depth_im = depth_imgs[i]
        
    rgb_img = cv2.imread(f"{folder}/color/{color_im}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    depth_img = cv2.imread(f"{folder}/depth/{depth_im}", cv2.IMREAD_ANYDEPTH)
    depth_np = np.array(depth_img, dtype=np.float32)
    
        
    plt.figure()
    plt.subplots_adjust(left=0)
    plt.imshow(depth_np, cmap='magma')
    plt.colorbar(fraction = 0.05, shrink = 1.0, pad = 0.01)    
    plt.axis('off')
    plt.savefig(f"{output_folder}/depth_{depth_im}.pdf")#, bbox_inches='tight')
    
    # Saturate
    depth_np[depth_np > 10000] = 10000
    depth_np_sat = depth_np / 10000

    # Resize rgb to depth size
    rgb_resized = cv2.resize(rgb_img, (depth_np.shape[1], depth_np.shape[0]))
    
    # Save images
    plt.figure()
    plt.subplots_adjust(left=0)
    plt.subplots_adjust(right=1.06) 
    plt.imshow(rgb_resized)
    #plt.colorbar(fraction = 0.05, shrink = 1.0, pad = 0.01, alpha=0, visible=False)
    plt.axis('off')
    rgb_cb = plt.colorbar()
    rgb_cb.ax.set_visible(False)
    plt.savefig(f"{output_folder}/rgb_resized_{color_im}.pdf")#, bbox_inches='tight')
    
    plt.figure()
    plt.subplots_adjust(left=0)
    plt.imshow(depth_np_sat, cmap='magma')
    plt.colorbar(fraction = 0.05, shrink = 1.0, pad = 0.01)
    plt.axis('off')
    plt.savefig(f"{output_folder}/depth_{depth_im}_saturated.pdf")#, bbox_inches='tight')
    

