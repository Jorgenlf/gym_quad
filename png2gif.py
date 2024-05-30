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
mode = "webp" 
###---###---###---###

if option == 'known_n_imgs':
    n_imgs = 24
    path = "gym_quad/tests/test_img/depth_maps/"
    filenames = [f"{path}depth_map{i}{extension}" for i in range(n_imgs)]
    
elif option == 'unknown_ordered':
    #Choose which depth maps to use:    
    ###---###---###---###
    exp_id = 32
    test_id = 4
    scenario = "house_easy_obstacles"
    ###---###---###---###

    # Regular expression to extract numbers from filenames:
    def sort_key(filename):
        numbers = re.findall(r'\d+', filename)
        return [int(num) for num in numbers]

    path_pattern = f'log/LV_VAE_MESH-v0/Experiment {exp_id}/{scenario}/tests/test{test_id}/depth_maps/depth_map_*.png'
    filenames = sorted(glob.glob(path_pattern), key=sort_key)


if mode == "gif":
    print("There are ", len(filenames), " images in the folder\nBeginning to create gif... ",end="")
    
    #Frame duration (max 50 for gif):
    frame_duration = 1 / 50

    # Read, compose and write images to .gif:
    with imageio.get_writer('my_image_animation.gif', mode='I', duration=frame_duration, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("GIF created successfully!")

elif mode == "webp":
    print("There are ", len(filenames), " images in the folder\nBeginning to create webp animation... ",end="")

    # Frame duration set for 90 FPS:
    # frame_duration = 1 / 100  # seconds per frame #TODO changing duration seemingly does nothing...

    #IMAGEIOv2
    # Read, compose and write images to .webp:
    # with imageio.get_writer('my_image_animation.webp', mode='I', duration=frame_duration, loop=0, quality=100) as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    
    #IMAGEIOv3
    # # Load images
    # images = [iio.imread(filename) for filename in filenames]
    # # Set duration (10 ms per frame for 100 FPS)
    # duration_per_frame = 10
    # # Save as WebP animation
    # iio.imwrite('my_image_animation.webp', images, duration=duration_per_frame, loop=0)
    format = ".webp" 
    output_path = "animated" + format
    iio.imwrite(output_path, [iio.imread(fp) for fp in filenames],
                    fps=30,  # Frames per second
                    loop=0,  # 0 = infinite loop, 1 = no loop
                    quantizer="nq",
                    )

    #PILLOW
    # images = [Image.open(filename) for filename in filenames]

    # # Save as WebP animation
    # images[0].save('my_image_animation.webp',
    #             save_all=True, append_images=images[1:], duration=10, loop=0, quality=100)

    print("WebP animation created successfully!")        

#TODO can make a call to this after calling run such that the gif is created automatically can then delete all the depthmaps if we want to save space