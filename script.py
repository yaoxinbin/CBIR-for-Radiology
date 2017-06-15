from __future__ import print_function
# uses py 2.7

if __name__ == '__main__':

	import cv2
	import os
 	#os.chdir('/Users/Sriram/Desktop/DePaul/CBIR-for-Radiology/images_sample')
	os.chdir('C:/Users/SYARLAG1/Documents/CBIR-for-Radiology')
	from calc_image_association import *
	from read_images_gen_feats import *

	import matplotlib.pyplot as plt


	image_dict = read_images_from_folder('./images_sample/') # make sure '/' is included at end!

	image_feats_dict = add_image_features(image_dict, kind = 'sift', ellipse=True)

	query_image_arr = cv2.imread('./images_sample/3_12') # change as needed

	query_image_feats = image_descriptors.sift(query_image_arr, ellipse=True)

	#cv2.imshow('QUERY IMAGE', query_image_arr)

	image_dist_dict = calc_dist_sim(query_image_feats, image_feats_dict, method='bag_of_words', k=50)
	    
	result_image_id_list = return_images(image_dist_dict, image_dict, k=5, distance=True, show=False)

	for image_id in result_image_id_list:

		image_location = './images_sample/'+image_id

		# display images
		img = cv2.imread(image_location)

		plt.figure()

		plt.imshow(img, cmap='gray')

		plt.show()


### Visualize the images in a mds projection plot of the images
import numpy as np


# from the calc image association function:

# apply k-means to find the centroids
train_feats = np.concatenate(image_feats_dict.values())

# TODO: Kmeans calculation takes the largest amount of time, everything else is fast
k_means = KMeans(n_clusters=k, random_state=seed).fit(train_feats)

cluster_centers = k_means.cluster_centers_

# TODO: loops are slow --> replace with numpy matrix magic
# find closest center to each image keypoint and generate histogram
image_hist_dict = {}

for image_id, each_image in image_feats_dict.items():

	image_hist_dict[image_id] = [0] * k

	for keypoint in each_image:

		diff = cluster_centers - repmat(keypoint,len(cluster_centers), 1)

		euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff, ord=2)

		image_hist_dict[image_id][np.argmin(euclidean_dists)] += 1. # add to frequency of correponding center
    
X_to_project = np.array(image_hist_dict.values())

Y_for_color = 









