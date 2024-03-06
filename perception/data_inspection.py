import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import skew, kurtosis, entropy
from tqdm import tqdm


# Get all images from ./sunrgbd_stripped and create a histogram of the pixel intensities
path = "data/realsense_data/depth_imgs"
#path = "data/sunrgbd_images_depth"
images = os.listdir(path)

#all_pixels = np.array([])
shapes = {}
depth_stats = {
    'mean': [],
    'median': [],
    'std': [],
    'min': [],
    'max': [],
    'skewness': [],
    'kurtosis': [],
    'entropy': []
}
aspect_ratios = []

if path == "data/realsense_data/depth_imgs":
    savepath = "realsense_distributions/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    divide_by_to_get_meters = 1000

if path == "data/sunrgbd_images_depth":
    savepath = "sunrgbd_distributions/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    divide_by_to_get_meters = 10000

max_depth = 65535 / 10000

num_bins = 100
bin_edges = np.linspace(0, max_depth, num_bins+1)
hist_counts = np.zeros(num_bins, dtype=np.int64)

counter = 0
#pixels_batch = []
for image in tqdm(images, desc="Processing images"):
    counter += 1
    im = cv2.imread(f"{path}/{image}", cv2.IMREAD_ANYDEPTH)
    if path == "data/realsense_data/depth_imgs":
        im[im > 6500] = 6500
        #im = im / 6500
    im = im / divide_by_to_get_meters
    flattened_im = im.flatten()

    # Incrementally build histogram, needed bc. task is killed if too much memory is used
    indices = np.searchsorted(bin_edges, flattened_im, side='right') - 1
    indices[indices == num_bins] = num_bins - 1  # Adjust indices that are out of bounds

    np.add.at(hist_counts, indices, 1)

    # Collect stats
    #pixels_batch.extend(flattened_im)
    depth_stats['mean'].append(np.mean(flattened_im))
    depth_stats['median'].append(np.median(flattened_im))
    depth_stats['std'].append(np.std(flattened_im))
    depth_stats['min'].append(np.min(flattened_im))
    depth_stats['max'].append(np.max(flattened_im))
    depth_stats['skewness'].append(skew(flattened_im))
    depth_stats['kurtosis'].append(kurtosis(flattened_im))
    depth_stats['entropy'].append(entropy(flattened_im))

    # Image shapes and aspect ratios
    shape = str(im.shape)
    shapes[shape] = shapes.get(shape, 0) + 1
    aspect_ratios.append(im.shape[1] / im.shape[0])

    #if counter == 100: # To not overshoot memory capacity
    #    break

# Calculate bin centers from bin_edges for plotting
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])


# Save arrays to file
#np.save(f"{savepath}all_pixels.npy", all_pixels)
np.save(f"{savepath}depth_stats_mean.npy", depth_stats['mean'])
np.save(f"{savepath}depth_stats_median.npy", depth_stats['median'])
np.save(f"{savepath}depth_stats_std.npy", depth_stats['std'])
np.save(f"{savepath}depth_stats_min.npy", depth_stats['min'])
np.save(f"{savepath}depth_stats_max.npy", depth_stats['max'])
np.save(f"{savepath}depth_stats_skewness.npy", depth_stats['skewness'])
np.save(f"{savepath}depth_stats_kurtosis.npy", depth_stats['kurtosis'])
np.save(f"{savepath}depth_stats_entropy.npy", depth_stats['entropy'])
np.save(f"{savepath}aspect_ratios.npy", aspect_ratios)
np.save(f"{savepath}shapes.npy", shapes)


print("Done processing images")
print("Plotting...")

plt.style.use('ggplot')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

if path == "realsense_data/depth_imgs":
    ggplot_blue = '#377eb8'  # This is a common blue used in ggplot visualizations.
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[ggplot_blue])

plt.bar(bin_centers, hist_counts, width=np.diff(bin_edges))
plt.xlabel('Depth [m]')
plt.ylabel('Number of Pixels')
plt.savefig(f"{savepath}depth_histogram.pdf", bbox_inches='tight')

# Plot histogram of pixel intensities
#plt.figure()
#plt.hist(all_pixels, bins=100, range=(0, max(all_pixels)))
#plt.xlabel("Depth [m]")
#plt.ylabel("Frequency")
#plt.savefig(f"{savepath}depth_histogram.pdf", bbox_inches='tight')


# Plot image shapes
plt.figure()
plt.bar(shapes.keys(), shapes.values())
plt.xlabel("Image Shapes")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.savefig(f"{savepath}image_shapes.pdf", bbox_inches='tight')

# Plot additional statistics
# Mean Depth Value Distribution
plt.figure()
plt.hist(depth_stats['mean'], bins=100)
plt.xlabel("Mean Depth Value [m]")
plt.ylabel("Frequency")
plt.savefig(f"{savepath}mean_depth_distribution.pdf", bbox_inches='tight')

# Standard Deviation Distribution
plt.figure()
plt.hist(depth_stats['std'], bins=100)
plt.xlabel("Standard Deviation of Depth Values [m]")
plt.ylabel("Frequency")
plt.savefig(f"{savepath}std_dev_depth_distribution.pdf", bbox_inches='tight')

# Aspect Ratio Distribution
plt.figure()
plt.hist(aspect_ratios, bins=100)
plt.xlabel("Aspect Ratio (Width/Height)")
plt.ylabel("Frequency")
plt.savefig(f"{savepath}aspect_ratio_distribution.pdf", bbox_inches='tight')

# Min Depth Value Distribution
plt.figure()
plt.hist(depth_stats['min'], bins=100)
plt.xlabel("Minimum Depth Value [m]")
plt.ylabel("Frequency")
plt.savefig(f"{savepath}min_depth_distribution.pdf", bbox_inches='tight')

# Max Depth Value Distribution
plt.figure()
plt.hist(depth_stats['max'], bins=100)
plt.xlabel("Maximum Depth Value [m]")
plt.ylabel("Frequency")
plt.savefig(f"{savepath}max_depth_distribution.pdf", bbox_inches='tight')

# Skewness Distribution
plt.figure()
plt.hist(depth_stats['skewness'], bins=100)
plt.xlabel("Skewness of Depth Values")
plt.ylabel("Frequency")
plt.savefig(f"{savepath}skewness_distribution.pdf", bbox_inches='tight')

# Kurtosis Distribution
plt.figure()
plt.hist(depth_stats['kurtosis'], bins=100)
plt.xlabel("Kurtosis of Depth Values")
plt.ylabel("Frequency")
plt.savefig(f"{savepath}kurtosis_distribution.pdf", bbox_inches='tight')

# Entropy Distribution
plt.figure()
plt.hist(depth_stats['entropy'], bins=100)
plt.xlabel("Entropy of Depth Values")
plt.ylabel("Frequency")
plt.savefig(f"{savepath}entropy_distribution.pdf", bbox_inches='tight')
