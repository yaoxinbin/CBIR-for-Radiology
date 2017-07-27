# This script performs the following:
# <------------------------The offine parts---------------------->
# PART A: Download images from server
# PART B: Extract SIFT keypoints from downloaded images
# PART C: Perform KMeans to gen a matrix of cluster centers
#         (each row is a cluster center and the cols are the features from SIFT)
# PART D: Generate the Bag of Words (BoW) vector for each database image
# <----------------------The online part------------------------->
# PART E: Read in the query image
# PART F: Generate SIFT keypoints for the query image
# PART G: Create BoW vector for query image using previously generated
#         cluster centers.
# PART H: Find the cosine distance between query BoW vector and database BoW vectors
# PART I: Find the 10 most similar images to query and display them along with
#         various attributes


import os
import matplotlib.pyplot as plt
from CBIR_functions import * # reads in all the necessary functions for CBIR

os.chdir('C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology')

images_folder = 'images_sample'
images_percent_for_kmeans = 0.1
cluster_count = 50
query_image_id = '180_4'

# PART A
download_images(only_dicom=False, folder_name= images_folder,
                images_loc= 'http://rasinsrv04.cstcis.cti.depaul.edu/all_images/all_tf/')

# PART B
database_dict = read_images_from_folder('./'+images_folder+'/')
database_SIFT_feats_dict = add_image_features(database_dict, kind='sift', ellipse=False)

# PART C (only use 10% of keypoints -- randomly selected)
cluster_centers = kmeans_centers(database_SIFT_feats_dict, cluster_count, True, images_percent_for_kmeans)

# PART D
database_BoW_dict =  bag_of_words(database_SIFT_feats_dict, cluster_centers, query=False)

# PART E
query_image_arr = plt.imread('./'+images_folder+'/'+query_image_id)

# PART F
query_SIFT_feats = sift(query_image_arr, ellipse=False)

# PART G
query_BoW_arr =  bag_of_words(query_SIFT_feats, cluster_centers, query=True)

# PART H (using cosine distance NOT similarity)
dist_dict = calc_dist_sim(query_BoW_arr, database_BoW_dict, method='cosine')

# PART I
closest_images = return_images(dist_dict, database_dict, k=10, distance=True, show=True)
