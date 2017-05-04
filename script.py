from __future__ import print_function
# uses py 2.7

import cv2
import os
import dicom
import numpy as np
from numpy.matlib import repmat# for repmat

from scipy.spatial.distance import * # dist and sim metrics

start = True # set to True to run the functions

# for each image in dict, exi
class image_descriptors():

	# returns feature array obtained using sift algorithm
	@staticmethod
	def sift(pixel_array):

		# convert to grayscale
		gray= cv2.cvtColor(pixel_array,cv2.COLOR_BGR2GRAY)

		# use ORD. similar to SIFT + SURF - and is free to use (unlike the other 2)
		orb = cv2.ORB()

		# detector of the points in the image
		#detector = cv2.FeatureDetector_create("SIFT") 

		# extractor of the detected points 
		#descriptor = cv2.DescriptorExtractor_create("SIFT")
 
		#skp = detector.detect(pixel_array)

		_, feats = orb.detectAndCompute(gray,None)

		return feats

	# returns a relative histogram for each HSV channel
	@staticmethod
	def hist(pixel_array, bins = 10):
     # with thanks to http://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
     # and https://github.com/danuzclaudes/image-retrieval-OpenCV/		

          hsv = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2HSV)

          feats = []
          
          # get image dimensions
          x,y = hsv.shape[:2]

          dx, dy = int(x*0.5), int(y*0.5) # get halfway point to divde into segements
          
          # create 4 segements (top-left, bottom-left, top-right, bottom-right)
          regions = [(0,dx,0,dy), (0,dx,dy,y), (dx,x,0,dy), (dx,x,dy,y)]
          
          # elliptical mask for the center of the image
          # major and minor axis
          ex, ey = int(x*0.75)/2, int(y*0.75)/2
          
          ellipse_mask = np.zeros([x,y], dtype = "uint8")

          # make mask into an ellipse          
          cv2.ellipse(ellipse_mask, (dx, dy), (ex, ey), 0, 0, 360, 255, -1)
          
          # gen histogram
          hist = cv2.calcHist([hsv],[0,1,2],ellipse_mask,[bins],[0,180,0,256,0,256],True)
          
          # normalize
          hist = cv2.normalize(hist).flatten()
          
          feats.extend(hist)
          
          
          # loop through the segements and extract the histograms
          for area in regions:

              # a second mask is needed for each of the corners (everything other than the ellipse)
              corner_mask = np.zeros([x,y], dtype = "uint8")
              
              # draw rectangle mask on corner_mask object
              corner_mask[area[0]:area[1], area[2]:area[3]] = 255        

              corner_mask = cv2.subtract(corner_mask, ellipse_mask)
              
              # gen histogram like before
              hist = cv2.calcHist([hsv],[0,1,2],corner_mask,bins,[0,180,0,256,0,256])

              hist = cv2.normalize(hist).flatten()
              
              feats.extend(hist)
           
           
          return feats


	@staticmethod
	def geometric(pixel_array):
		pass

	@staticmethod
	def mixed(pixel_array):
		pass



# returns the pixel array of dicom image
def read_dicom_image(image_path):

	return dicom.read_file(image_path)


# returns a dictionary of images extracted from folder
def read_images_from_folder(location):
	
	image_dict = {}
	image_path_list = os.listdir(location)

	for image_path in image_path_list:

		if image_path[-3:] == 'dcm': # if dicom image

			image_dict[image_path]=read_dicom_image(image_path).pixel_array

		else:
		
			image_dict[image_path]=cv2.imread(image_path,1)

	return image_dict


# for each image in dict, extract image features and add to new dict
# this function will updated as more feature extraction techniques 
# are introduced
def add_image_features(image_dict, kind = 'sift'):

	image_feats_dict = {}

	for image in image_dict.keys():

		if image[0] == '.': continue # ignore non-image files

		print(str(image))

		if kind == 'sift':

			image_feats_dict[image] = image_descriptors.sift(image_dict[image])

		if kind == 'hist':

			image_feats_dict[image] = image_descriptors.hist(image_dict[image])

		if kind == 'geometric':

			image_feats_dict[image] = image_descriptors.geometric(image_dict[image])

		if kind == 'mixed':

			image_feats_dict[image] = image_descriptors.mixed(image_dict[image])


	return image_feats_dict


# returns keys of the top-k images
# euclidean and cosine used for histogram and sift has its own method using bag of words approach

def calc_dist_sim(query_image_arr, image_feats_dict, method='sift', k=10):

	image_sim_dist_dict = {}

	if method == 'cosine':

		# get the histogram features for the query image
		query_feats = image_descriptors.hist(query_image_arr)
		
		mat = image_feats_dict.values()

		# L-2 norms of the train image featuers
		row_norms = np.apply_along_axis(np.linalg.norm, 1, mat)

		query_norm = np.linalg.norm(query_feats)

		# quick matrix multiplication
		sim_mat = np.einsum('ji,i->j', mat, query_feats.T)

		# first divide by query norm
		sim_mat = sim_mat/query_norm

        # then by individual row norms
		sim_mat = sim_mat/row_norms[:,None]

        # add to the dictionary 
		image_sim_dist_dict = dict((key, val) for key,val in zip(image_feats_dict.keys(),sim_mat))


	if method == 'euclidean':

		# get the histogram features for the query image
		query_feats = image_descriptors.hist(query_image_arr)

		mat = image_feats_dict.values()

		diff = mat - repmat(query_feats,len(mat), 1)

		# L-2 norms of the train image featuers
		euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff)

		# add to the dictionary 
		image_sim_dist_dict = dict((key, val) for key,val in zip(image_feats_dict.keys(),euclidean_dists))
      

	if method == 'sift':

		# init the matching method
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# extract sift features from query image
		query_feats = image_descriptors.sift(query_image_arr)

		# find the match between query and each training image
		for image in image_feats_dict:

			matches = matcher.match(query_feats, image_feats_dict[image])

			# calculate the matching distance
			distances = [x.distance for x in matches]

			# assign query, image dist as the average
			image_sim_dist_dict[image] = sum(distances)/(len(distances)+1)


	return image_sim_dist_dict


# display retrieved images
def return_images(image_sim_dist_dict, image_dict, k=5, distance=True):

	result_image_id_list = []

	# sort based on whether sim or dist measure
	sorted_list = sorted(image_sim_dist_dict.items(), key=lambda x: x[1], reverse=distance)

	for i in range(k):

		image_id = sorted_list[i][0]

		image_name = 'result ' + str(i) + ': ' + image_id  

		cv2.imshow(image_name, image_dict[image_id])

		result_image_id_list.append(image_id)

	return result_image_id_list


############################################################################################

if start == True:
    
    os.chdir('C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/images_sample')
    
    image_dict = read_images_from_folder('./')
    
    image_feats_dict = add_image_features(image_dict, kind = 'sift')
    
    query_image_arr = cv2.imread('C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/images_sample/171_1',1) # change as needed
    
    cv2.imshow('QUERY IMAGE', query_image_arr)

    image_sim_dict = calc_sim(query_image_arr, image_feats_dict, method='sift', k=10)
        
    result_image_id_list = return_images(image_sim_dict, image_dict, k=5, distance=True)
    
    
    









