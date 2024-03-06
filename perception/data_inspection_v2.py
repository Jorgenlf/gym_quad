import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import skew, kurtosis, entropy
from tqdm import tqdm

locations = ["office", "hallway", "stairs", "glassgaarden"]
basepath = "data/realsense_data_v2"

savepath = "data/realsense_distributions_v2/"

# Collect all images in this list
images = []
for location in locations:
    images.extend(os.listdir(f"{basepath}/{location}/depth_imgs"))

print(len(images))


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

divide_by_to_get_meters = 1000
thresh_depth = 10000 # 10 meters
max_depth_meters = 65535 / divide_by_to_get_meters

num_bins = 100
bin_edges = np.linspace(0, max_depth_meters, num_bins+1)
hist_counts = np.zeros(num_bins, dtype=np.int64)

num_bins = 100
bin_edges_trunc = np.linspace(0, thresh_depth/divide_by_to_get_meters, num_bins+1)
hist_counts_trunc = np.zeros(num_bins, dtype=np.int64)

for loc in locations:
    images = os.listdir(f"{basepath}/{loc}/depth_imgs")
    for image in tqdm(images, desc="Processing images"):
        im = cv2.imread(f"{basepath}/{loc}/depth_imgs/{image}", cv2.IMREAD_ANYDEPTH)

        #im[im > thresh_depth] = thresh_depth
        im = im / divide_by_to_get_meters
        flattened_im = im.flatten()

        # Incrementally build histogram, needed bc. task is killed if too much memory is used
        indices = np.searchsorted(bin_edges, flattened_im, side='right') - 1
        indices[indices == num_bins] = num_bins - 1  # Adjust indices that are out of bounds
        np.add.at(hist_counts, indices, 1)

        # Truncated histogram
        im_trunc = im.copy()
        im_trunc[im_trunc > thresh_depth/divide_by_to_get_meters] = thresh_depth/divide_by_to_get_meters
        flattened_im_trunc = im_trunc.flatten()
        indices_trunc = np.searchsorted(bin_edges_trunc, flattened_im_trunc, side='right') - 1
        indices_trunc[indices_trunc == num_bins] = num_bins - 1  # Adjust indices that are out of bounds
        np.add.at(hist_counts_trunc, indices_trunc, 1)

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

 

# Calculate bin centers from bin_edges for plotting
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_centers_trunc = 0.5 * (bin_edges_trunc[:-1] + bin_edges_trunc[1:])


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



print("Done processing images")
print(f"Image shape = {im.shape}")
print("Plotting...")

plt.style.use('ggplot')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

# Depth histograms
plt.figure()
plt.bar(bin_centers, hist_counts, width=np.diff(bin_edges))
plt.xlabel('Depth [m]')
plt.ylabel('Number of Pixels')
plt.savefig(f"{savepath}depth_histogram.pdf", bbox_inches='tight')

plt.figure()
bin_centers_trunc[-1] = bin_centers_trunc[-3]
hist_counts_trunc[-1] = hist_counts_trunc[-3]
plt.bar(bin_centers_trunc, hist_counts_trunc, width=np.diff(bin_edges_trunc))
plt.xlabel('Depth [m]')
plt.ylabel('Number of Pixels')
plt.savefig(f"{savepath}depth_histogram_trunc.pdf", bbox_inches='tight')


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
