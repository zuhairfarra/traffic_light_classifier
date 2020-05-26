# Traffic Light Classifier

import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

get_ipython().run_line_magic('matplotlib', 'inline')


# Training and Testing Data
# 
# All 1484 of the traffic light images are separated into training and testing datasets. 
# 
# * 80% of these images are training images, for use to create a classifier.
# * 20% are test images, which will be used to test the accuracy of the classifier.
# * All images are pictures of 3-light traffic lights with one light illuminated.

# First, set some variables to keep track of some where images are stored:
# 
#     IMAGE_DIR_TRAINING: the directory where training image data is stored
#     IMAGE_DIR_TEST: the directory where test image data is stored

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


# Load the datasets

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


# Visualize the Data

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
img_num = 0

selected_image = IMAGE_LIST[img_num][0]
selected_label = IMAGE_LIST[img_num][1]

plt.imshow(selected_image)
print("Image Dimensions: " + str(selected_image.shape))
print("Label: " + str(selected_label))


# Pre-process the Data

# A red light should have the  label: [1, 0, 0]. Yellow should be: [0, 1, 0]. Green should be: [0, 0, 1]. These labels are called **one-hot encoded labels**.

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    # Resize image and pre-process so that all "standard" images are the same size  
    standard_im = cv2.resize(image,(32,32))
    
    return standard_im
    


# Standardize the output

# Given a label - "red", "green", or "yellow" - return a one-hot encoded label

def one_hot_encode(label):
    
    # Create a one-hot encoded label that works for all classes of traffic lights
    if (label == "red"):
        return [1, 0, 0]
    elif (label == "green"):
        return [0, 0, 1]
    else:
        return [0, 1, 0]

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


# Construct a `STANDARDIZED_LIST` of input images and output labels.

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


# Visualize the standardized data

img = 748
org_img = IMAGE_LIST[img][0]
std_img = STANDARDIZED_LIST[img][0]

# f, (ax1,ax2) = plt.subplots(1,2)
# ax1.imshow(org_img)
# ax2.imshow(std_img)

print("Label of original img: " + str(IMAGE_LIST[img][1]))
print("Label of standardized img: " + str(STANDARDIZED_LIST[img][1]))

r = std_img[:,:,0]
g = std_img[:,:,1]
b = std_img[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(std_img)
ax2.set_title('R channel')
ax2.imshow(r, cmap='gray')
ax3.set_title('G channel')
ax3.imshow(g, cmap='gray')
ax4.set_title('B channel')
ax4.imshow(b, cmap='gray')

# Convert an image to HSV colorspace
# Visualize the individual color channels

image_num = 78
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

# This feature uses HSV colorspace values
def create_feature(rgb_image):
    
    hsv = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)

    # Create and return a feature value and/or vector
    sat_weight = 0
    val_weight = 1
    
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    up_region = np.array(v[:12,10:25])
    mid_region = np.array(v[12:22,10:25])
    low_region = np.array(v[22:,10:25])
    
    up_mean = np.sum(up_region)/len(up_region)
    mid_mean = np.sum(mid_region)/len(mid_region)
    low_mean = np.sum(low_region)/len(low_region)
    
    up_sregion = np.array(s[:12,10:25])
    mid_sregion = np.array(s[12:22,10:25])
    low_sregion = np.array(s[22:,10:25])
    
    up_smean = np.sum(up_sregion)/len(up_sregion)
    mid_smean = np.sum(mid_sregion)/len(mid_sregion)
    low_smean = np.sum(low_sregion)/len(low_sregion)
    
    total_up_mean = val_weight*up_mean + sat_weight*up_smean
    total_mid_mean = val_weight*mid_mean + sat_weight*mid_smean
    total_low_mean = val_weight*low_mean + sat_weight*low_smean
    
    return [total_up_mean, total_mid_mean, total_low_mean]

ftre1 = create_feature(hsv)

print(ftre1)
print(np.argmax(ftre1))

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    # Extract feature(s) from the RGB image and use those features to
    # classify the image and output a one-hot encoded label
    sum_feature = create_feature(rgb_image)
    
    feature_max = np.argmax(sum_feature)
    
    if feature_max == 0:
        return one_hot_encode("red")
    elif feature_max == 2:
        return one_hot_encode("green")
    else:
        return one_hot_encode("yellow")
    


# Testing the classifier


# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from the classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))


# Visualize misclassified example(s)
# Display an image in the `MISCLASSIFIED` list 
# Print out its predicted label - to see what the image *was* incorrectly classified
imgNUM = 6

plt.imshow(MISCLASSIFIED[imgNUM][0])
print('Predicted label: ' + str(MISCLASSIFIED[imgNUM][1]))

hsv = cv2.cvtColor(MISCLASSIFIED[imgNUM][0], cv2.COLOR_RGB2HSV)

# # HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# # Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(hsv)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')


# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")