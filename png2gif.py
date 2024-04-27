import imageio.v2 as imageio
import glob


# Define filetype of source images:
extension = '.png'
option = 'B'


# Option A: N filenames have common structure and are numbered in the correct sequence:
#	    "image_name_0.png", "image_name_1.png", "image_name_2.png", ...
# 	    n_imgs: known number of images

# Option B: unknown filenames, but they are still ordered in ascending order by name:
# import glob


if option == 'A':
    n_imgs = 24
    path = "gym_quad/tests/test_img/depth_maps/"
    filenames = [f"{path}depth_map{i}{extension}" for i in range(n_imgs)]
elif option == 'B':

    exp_id = 2
    test_id = 5
    scenario = "horizontal"

    filenames = sorted(glob.glob(f'log/LV_VAE_MESH-v0/Experiment {exp_id}/{scenario}/tests/test{test_id}/depth_maps/depth_map_*.png'))

print("there are ", len(filenames), " images in the folder\nBeginning to create gif... ",end="")


# Read, compose and write images to .gif:
with imageio.get_writer('my_image_animation.gif', mode='I', duration=0.01, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF created successfully!")
#TODO make resolution of gif better?
#TODO can make a call to this after calling run such that the gif is created automatically can then delete all the depthmaps if we want to save space