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
import pandas as pd
import pickle

os.chdir('C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/functions_for_database_CBIR')

from CBIR_functions import * # reads in all the necessary functions for CBIR

# change these params as needed
images_folder = 'all_images_sample'
images_percent_for_kmeans = 0.08
cluster_count = 50
query_image_id = '60_1'
save_files = True
ellipse = False
image_return_count = 20

# <------------------------The offine parts---------------------->
# PART A
download_images(only_dicom=False, folder_name = images_folder,
                images_loc= 'http://rasinsrv04.cstcis.cti.depaul.edu/all_images/all_tf/')

# PART B
database_dict = read_images_from_folder('./'+images_folder+'/')

database_SIFT_feats_dict = add_image_features(database_dict, kind='sift', ellipse=ellipse)


# remove invalid images -- sometimes the CV2 sift doesnt work.
wrong_img_store = []
for image_id in database_SIFT_feats_dict.keys():
    try:
        database_SIFT_feats_dict[image_id].shape
    except:
        wrong_img_store.append(image_id)
        del database_SIFT_feats_dict[image_id]
        del database_dict[image_id]

# wrong_img_store =  ['1990_2', '1988_2', '308_26', '308_25', '422_7', '1835_1']

# PART C (only use 8% of keypoints -- randomly selected)
cluster_centers = kmeans_centers(database_SIFT_feats_dict, cluster_count, True, images_percent_for_kmeans)

# save as csv
if save_files:
    cluster_centers_df = pd.DataFrame(cluster_centers)
    cluster_centers_df.to_csv('cluster_centers.csv')
    del cluster_centers_df

# PART D
database_BoW_dict =  bag_of_words(database_SIFT_feats_dict, cluster_centers, query=False)

# save as csv
if save_files:
    database_BoW_df = pd.DataFrame(database_BoW_dict.values(), index=database_BoW_dict.keys())
    database_BoW_df.to_csv('database_BoW.csv')
    del database_BoW_df

# <----------------------The online part------------------------->
# PART E
query_image_arr = plt.imread('./'+images_folder+'/'+query_image_id)

# PART F
query_SIFT_feats = sift(query_image_arr, ellipse=False)

# PART G
query_BoW_arr =  bag_of_words(query_SIFT_feats, cluster_centers, query=True)

# PART H (using cosine distance NOT similarity)
dist_dict = calc_dist_sim(query_BoW_arr, database_BoW_dict, method='cosine')

# PART I (k is the number of images to return)
closest_images = return_images(dist_dict, database_dict, k=image_return_count, distance=True, show=True)

# save result as csv
if save_files:
    closest_images = np.array(closest_images)
    database_result_df = pd.DataFrame(closest_images[:,1], index=closest_images[:,0])
    database_result_df.to_csv('closest_image_list.csv')
    del database_result_df

    
