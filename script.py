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








