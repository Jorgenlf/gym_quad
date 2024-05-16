import imageio.v2 as imageio
import glob
import re


# Define filetype of source images:
extension = '.png'

# Option A: N filenames have common structure and are numbered in the correct sequence:
#	    "image_name_0.png", "image_name_1.png", "image_name_2.png", ...
# 	    n_imgs: known number of images

# Option B: unknown filenames, but they are still ordered in ascending order by name:

#Choose option A or B:
###---###---###---### 
option = 'B'
###---###---###---###

if option == 'A':
    n_imgs = 24
    path = "gym_quad/tests/test_img/depth_maps/"
    filenames = [f"{path}depth_map{i}{extension}" for i in range(n_imgs)]
    
elif option == 'B':
    #Choose which depth maps to use:    
    ###---###---###---###
    exp_id = 19
    test_id = 16
    scenario = "house"
    ###---###---###---###

    # Regular expression to extract numbers from filenames:
    def sort_key(filename):
        numbers = re.findall(r'\d+', filename)
        return [int(num) for num in numbers]

    path_pattern = f'log/LV_VAE_MESH-v0/Experiment {exp_id}/{scenario}/tests/test{test_id}/depth_maps/depth_map_*.png'
    filenames = sorted(glob.glob(path_pattern), key=sort_key)

###---###---###---### Define if you want a gif or a webp animation:
create_gif = False #IF false creates a webp animation instead (better quality)
###---###---###---###

if create_gif:
    print("There are ", len(filenames), " images in the folder\nBeginning to create gif... ",end="")
    
    #Frame duration (max 50 for gif):
    frame_duration = 1 / 30

    # Read, compose and write images to .gif:
    with imageio.get_writer('my_image_animation.gif', mode='I', duration=frame_duration, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("GIF created successfully!")

elif not create_gif:
    print("There are ", len(filenames), " images in the folder\nBeginning to create webp animation... ",end="")

    # Frame duration set for 90 FPS:
    frame_duration = 1 / 90  # seconds per frame

    # Read, compose and write images to .webp:
    with imageio.get_writer('my_image_animation.webp', mode='I', duration=frame_duration, loop=0, quality=100) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print("WebP animation created successfully!")        

#TODO can make a call to this after calling run such that the gif is created automatically can then delete all the depthmaps if we want to save space