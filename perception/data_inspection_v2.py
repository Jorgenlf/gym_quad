"""
Simple script to inspect the depth images and calculate some statistics for the report. 
Handles histograms of very large datasets by incrementally building them.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import skew, kurtosis, entropy
from tqdm import tqdm


if __name__ == "__main__":
    datatype = "synthetic" # "real" or "synthetic", litt jalla


    if datatype == "synthetic":
        readpath = "data/synthetic_depthmaps/"
        savepath = "data/synthetic_distributions_v2/"
        os.makedirs(savepath, exist_ok=True)

        # Collect all images in this list
        images = []
        images = os.listdir(readpath)

        images = images[:10000]

        depth_stats = {
            'mean': [],
            'median': [],
            'std': []
        }

        divide_by_to_get_meters = 1
        thresh_depth = 10 # 10 meters

        num_bins = 100
        bin_edges = np.linspace(0, thresh_depth/divide_by_to_get_meters, num_bins+1)
        hist_counts = np.zeros(num_bins, dtype=np.int64)

        for image in tqdm(images, desc="Processing images"):
            im = cv2.imread(f"{readpath}/{image}", cv2.IMREAD_ANYDEPTH)

            # Incrementally build histogram, needed bc. task is killed if too much memory is used
            indices = np.searchsorted(bin_edges, im.flatten(), side='right') - 1
            indices[indices == num_bins] = num_bins - 1
            np.add.at(hist_counts, indices, 1)

            # Collect stats
            depth_stats['mean'].append(np.mean(im.flatten()))
            depth_stats['median'].append(np.median(im.flatten()))
            depth_stats['std'].append(np.std(im.flatten()))


        # Calculate bin centers from bin_edges for plotting
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Save arrays to file   
        np.save(f"{savepath}depth_stats_mean.npy", depth_stats['mean'])
        np.save(f"{savepath}depth_stats_median.npy", depth_stats['median'])
        np.save(f"{savepath}depth_stats_std.npy", depth_stats['std'])

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

        # Plot additional statistics
        # Mean Depth Value Distribution
        plt.figure()
        plt.hist(depth_stats['mean'], bins=100)
        plt.xlabel("Mean Depth Value [m]")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}mean_depth_distribution.pdf", bbox_inches='tight')

        # Median Depth Value Distribution
        plt.figure()
        plt.hist(depth_stats['median'], bins=100)
        plt.xlabel("Median Depth Value [m]")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}median_depth_distribution.pdf", bbox_inches='tight')

        # Standard Deviation Distribution
        plt.figure()
        plt.hist(depth_stats['std'], bins=100)
        plt.xlabel("Standard Deviation of Depth Values [m]")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}std_dev_depth_distribution.pdf", bbox_inches='tight')




    if datatype == "real":
        locations = ["office", "hallway", "stairs", "glassgaarden", "el5", "elbygget_longdistances", "hallway2", "hsp_human", "hsp_trees", "ohma", "parkinglot", "stairs_2"]
        basepath = "data/synthetic_depthmaps/"

        savepath = "data/synthetic_distributions_v2/"
        os.makedirs(savepath, exist_ok=True)

        # Collect all images in this list
        images = []
        #for location in locations:
        #    images.extend(os.listdir(f"{basepath}/{location}/depth_imgs"))

        images = os.listdir(basepath)

        #truncate list for testing
        images = images[:10000]

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

        divide_by_to_get_meters = 1 # 1000
        thresh_depth = 10 #10000 # 10 meters
        max_depth_meters = 65535 / divide_by_to_get_meters

        num_bins = 100
        bin_edges = np.linspace(0, thresh_depth/divide_by_to_get_meters, num_bins+1)
        hist_counts = np.zeros(num_bins, dtype=np.int64)

        num_bins = 100
        bin_edges_trunc = np.linspace(0, thresh_depth/divide_by_to_get_meters, num_bins+1)
        hist_counts_trunc = np.zeros(num_bins, dtype=np.int64)

        #for loc in locations:
            #images = os.listdir(f"{basepath}/{loc}/depth_imgs")
        for image in tqdm(images, desc="Processing images"):
            #im = cv2.imread(f"{basepath}/{loc}/depth_imgs/{image}", cv2.IMREAD_ANYDEPTH)
            im = cv2.imread(f"{basepath}/{image}", cv2.IMREAD_ANYDEPTH)

            #im[im > thresh_depth] = thresh_depth
            #im = im / divide_by_to_get_meters
            flattened_im = im.flatten()

            # Incrementally build histogram, needed bc. task is killed if too much memory is used
            indices = np.searchsorted(bin_edges, flattened_im, side='right') - 1
            indices[indices == num_bins] = num_bins - 1  # Adjust indices that are out of bounds
            np.add.at(hist_counts, indices, 1)

            # Truncated histogram
            #im_trunc = im.copy()
            #im_trunc[im_trunc > thresh_depth/divide_by_to_get_meters] = thresh_depth/divide_by_to_get_meters
            #flattened_im_trunc = im_trunc.flatten()
            #indices_trunc = np.searchsorted(bin_edges_trunc, flattened_im_trunc, side='right') - 1
            #indices_trunc[indices_trunc == num_bins] = num_bins - 1  # Adjust indices that are out of bounds
            #np.add.at(hist_counts_trunc, indices_trunc, 1)

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
        #bin_centers_trunc = 0.5 * (bin_edges_trunc[:-1] + bin_edges_trunc[1:])


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
        plt.savefig(f"{savepath}depth_histogram_saturated.pdf", bbox_inches='tight')

        #plt.figure()
        #bin_centers_trunc[-1] = bin_centers_trunc[-3]
        #hist_counts_trunc[-1] = hist_counts_trunc[-3]
        #plt.bar(bin_centers_trunc, hist_counts_trunc, width=np.diff(bin_edges_trunc))
        #plt.xlabel('Depth [m]')
        #plt.ylabel('Number of Pixels')
        #plt.savefig(f"{savepath}depth_histogram_trunc.pdf", bbox_inches='tight')


        # Plot additional statistics
        # Mean Depth Value Distribution
        plt.figure()
        plt.hist(depth_stats['mean'], bins=100)
        plt.xlabel("Mean Depth Value [m]")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}mean_depth_distribution_saturated.pdf", bbox_inches='tight')

        # Standard Deviation Distribution
        plt.figure()
        plt.hist(depth_stats['std'], bins=100)
        plt.xlabel("Standard Deviation of Depth Values [m]")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}std_dev_depth_distribution_saturated.pdf", bbox_inches='tight')

        # Min Depth Value Distribution
        plt.figure()
        plt.hist(depth_stats['min'], bins=100)
        plt.xlabel("Minimum Depth Value [m]")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}min_depth_distribution_saturated.pdf", bbox_inches='tight')

        # Max Depth Value Distribution
        plt.figure()
        plt.hist(depth_stats['max'], bins=100)
        plt.xlabel("Maximum Depth Value [m]")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}max_depth_distribution_saturated.pdf", bbox_inches='tight')

        # Skewness Distribution
        plt.figure()
        plt.hist(depth_stats['skewness'], bins=100)
        plt.xlabel("Skewness of Depth Values")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}skewness_distribution_saturated.pdf", bbox_inches='tight')

        # Kurtosis Distribution
        plt.figure()
        plt.hist(depth_stats['kurtosis'], bins=100)
        plt.xlabel("Kurtosis of Depth Values")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}kurtosis_distribution_saturated.pdf", bbox_inches='tight')

        # Entropy Distribution
        plt.figure()
        plt.hist(depth_stats['entropy'], bins=100)
        plt.xlabel("Entropy of Depth Values")
        plt.ylabel("Frequency")
        plt.savefig(f"{savepath}entropy_distribution_saturated.pdf", bbox_inches='tight')
