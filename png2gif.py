import imageio.v2 as imageio


# Define filetype of source images:
extension = '.png'

# Option A: N filenames have common structure and are numbered in the correct sequence:
#	    "image_name_0.png", "image_name_1.png", "image_name_2.png", ...
# 	    n_imgs: known number of images
n_imgs = 24
path = "C:/Users/admin/Desktop/EirikJorgenMasterCode/gym_quad/gym_quad/tests/test_img/depth_maps/"
filenames = [f"{path}depth_map{i}{extension}" for i in range(n_imgs)]

# Option B: unknown filenames, but they are still ordered in ascending order by name:
# import glob

# filenames = sorted(glob.glob('*{extension}'))

# Read, compose and write images to .gif:
with imageio.get_writer('my_image_animation.gif', mode='I', duration=0.01, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
