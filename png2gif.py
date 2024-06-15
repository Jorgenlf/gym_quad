import imageio.v2 as imageio
import imageio.v3 as iio
import glob
import re
from PIL import Image

# Define filetype of source images:
extension = '.png'

# Option known_n_imgs: N filenames have common structure and are numbered in the correct sequence:
#	    "image_name_0.png", "image_name_1.png", "image_name_2.png", ...
# 	    n_imgs: known number of images

# Option unknown_ordered: unknown filenames, but they are still ordered in ascending order by name:

#Choose option known_n_imgs or unknown_ordered:
###---###---###---### 
option = 'unknown_ordered' # 'known_n_imgs' or 'unknown_ordered
###---###---###---###

###---### Define if you want a gif or a webp animation: ###---###
mode = "gif"  # "gif" or "webp"
###---###---###---###

if option == 'known_n_imgs':
    n_imgs = 24
    path = "gym_quad/tests/test_img/depth_maps/"
    filenames = [f"{path}depth_map{i}{extension}" for i in range(n_imgs)]
    
elif option == 'unknown_ordered':
    #Choose which depth maps to use:    
    ###---###---###---###
    exp_id = 32
    test_id = 1
    scenario = "helix"
    path_pattern = f'log/LV_VAE_MESH-v0/Experiment {exp_id}/{scenario}/tests/test{test_id}/depth_maps/depth_map_*.png'
    ###---###---###---###
    
    #If depth and flight together choose agent aswell COMMENT THIS OUT IF ONLY DEPTH MAP POV WANTED
    agent = "locked_conv"
    scenario = "horizontal"
    path_pattern = f'plotting/replotting_results/depth_and_flight/{agent}/{scenario}/combined_img_*.png'

    ###---###---###---###

    # Regular expression to extract numbers from filenames:
    def sort_key(filename):
        numbers = re.findall(r'\d+', filename)
        return [int(num) for num in numbers]

    filenames = sorted(glob.glob(path_pattern), key=sort_key)


if mode == "gif":
    print("There are ", len(filenames), " images in the folder\nBeginning to create gif... ",end="")
    
    format = ".gif" 
    output_path = scenario + format
    iio.imwrite(output_path, [iio.imread(fp) for fp in filenames],
                    fps=30,  # Frames per second
                    loop=0,  # 0 = infinite loop, 1 = no loop
                    quantizer="nq",
                    )
    print("GIF created successfully!")

elif mode == "webp":
    print("There are ", len(filenames), " images in the folder\nBeginning to create webp animation... ",end="")

    format = ".webp" 
    output_path = scenario + format
    iio.imwrite(output_path, [iio.imread(fp) for fp in filenames],
                    fps=30,  # Frames per second
                    loop=0,  # 0 = infinite loop, 1 = no loop
                    quantizer="nq",
                    )

    print("WebP animation created successfully!")        
