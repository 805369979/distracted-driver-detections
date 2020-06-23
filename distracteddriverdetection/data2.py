# *-**-*
from builtins import Exception

import numpy as np
import pandas as pd
import os
import argparse
import errno
import scipy.misc
import dlib
import cv2

from skimage.feature import hog

# initialization
image_height = 1080
image_width = 1920
window_size = 112
window_step = 32
ONE_HOT_ENCODING = True
SAVE_IMAGES = False
GET_LANDMARKS = True
GET_HOG_FEATURES = True
GET_HOG_IMAGES = False
GET_HOG_WINDOWS_FEATURES = True
SELECTED_LABELS = []
IMAGES_PER_LABEL = 500
OUTPUT_FOLDER_NAME = "driver_feature"
data_path = 'data/test/'
data_dir_list = os.listdir(data_path)
# parse arguments and initialize variables:
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jpg", default="no", help="save images as .jpg files")
parser.add_argument("-l", "--landmarks", default="no", help="extract Dlib Face landmarks")
parser.add_argument("-ho", "--hog", default="yes", help="extract HOG features")
parser.add_argument("-hw", "--hog_windows", default="yes", help="extract HOG features from a sliding window")
parser.add_argument("-hi", "--hog_images", default="no", help="extract HOG images")
parser.add_argument("-o", "--onehot", default="yes", help="one hot encoding")
#['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
parser.add_argument("-e", "--expressions", default="'a','b','c','d','e','f','g','h','i','j'", help="choose the faciale expression you want to use: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sad, 6=Surprise")
args = parser.parse_args()
if args.jpg == "yes":
    SAVE_IMAGES = True
if args.landmarks == "yes":
    GET_LANDMARKS = True
if args.hog == "yes":
    GET_HOG_FEATURES = True
if args.hog_windows == "yes":
    GET_HOG_WINDOWS_FEATURES = True
if args.hog_images == "yes":
    GET_HOG_IMAGES = True
if args.onehot == "yes":
    ONE_HOT_ENCODING = True
if SELECTED_LABELS == []:
    SELECTED_LABELS = ['a','b','c','d','e','f','g','h','i','j']
    # SELECTED_LABELS = [0,1,2,3,4,5,6,7,8,9]
print( str(len(SELECTED_LABELS)) + " expressions")

# loading Dlib predictor and preparing arrays:
print( "preparing")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')
original_labels = ['a','b','c','d','e','f','g','h','i','j']
new_labels= SELECTED_LABELS
print(new_labels)
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
try:
    os.makedirs(OUTPUT_FOLDER_NAME)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
        pass
    else:
        raise

def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)

def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, 224, window_step):
        for x in range(0, 224, window_step):
            # print(x,y)
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                            cells_per_block=(1, 1), visualize=False))
    return hog_windows

print( "importing csv file")
# data = pd.read_csv('fer2013.csv')

for category in ['train','test']:
    print( "converting set: " + category + "...")
    #create folder
    if not os.path.exists(category):
        try:
            os.makedirs(OUTPUT_FOLDER_NAME + '\\' + category)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
                pass
            else:
                raise

data_path = 'data/train/'
data_dir_list = os.listdir(data_path)

img_rows=256
img_cols=256
num_channel=3

num_epoch=10
img_data_list_train=[]
img_data_list_test=[]

labels_list_train = []
labels_list_test = []

landmarks_train = []
landmarks_test = []

hog_features_train = []
hog_features_test = []

hog_images_train = []
hog_images_test = []

num_classes = 10
num_of_samples = 213
labels = []

import random

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        print(data_path + '/'+ dataset + '/'+ img)
        print(input_img)
        input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(input_img, (224, 224))
        # img_data_list_train.append(image)
        img_data_list_test.append(image)
        labels_list_test.append(get_new_label(img[0], one_hot_encoding=ONE_HOT_ENCODING))
        # labels_list_train.append(get_new_label(img[0], one_hot_encoding=ONE_HOT_ENCODING))
        nb_images_per_label[get_new_label(img[0])] += 1
# print(len(img_data_list_train))
print(len(img_data_list_test))
print(len(labels_list_test))
# print(len(labels_list_train))
np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/images.npy', img_data_list_test)
# np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/images.npy', img_data_list_train)
# if ONE_HOT_ENCODING:
#     np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/labels.npy', labels_list_train)
# else:
#     np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/labels.npy', labels_list_train)
if ONE_HOT_ENCODING:
    np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/labels.npy', labels_list_test)
else:
    np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/labels.npy', labels_list_test)
del img_data_list_test
# del img_data_list_train
del labels_list_test
# del labels_list_train

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        # labels=[]
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        print(data_path + '/'+ dataset + '/'+ img)
        input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(input_img, (224, 224))
        # labels.append(img[0])
        try:
            if img[0] in SELECTED_LABELS:
                if GET_HOG_WINDOWS_FEATURES:
                    # features = sliding_hog_windows(image)
                    # hog_features_test.append(features)
                    f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                       cells_per_block=(1, 1), visualise=True)
                    hog_features_train.append(f)
                # print(get_new_label(img[0], one_hot_encoding=ONE_HOT_ENCODING))

        except Exception as e:
            # print( "error in image: " + str(i) + " - " + str(e))
             print(e)

print(len(hog_features_test))
# print(len(hog_features_train))

# if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
#     np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/hog_features.npy', hog_features_train)
if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
    np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/hog_features.npy', hog_features_test)
    # print(len(img_data_list_train))
    # print(len(labels_list_train))
    # print(len(hog_features_train))
    # np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/images.npy', img_data_list_train)
    # if ONE_HOT_ENCODING:
    #     np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/labels.npy', labels_list_train)
    # else:
    #     np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/labels.npy', labels_list_train)
    # if GET_LANDMARKS:
    #     np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/landmarks.npy', landmarks_train)
    # if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
    #     np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/hog_features.npy', hog_features_train)
    # if GET_HOG_IMAGES:
    #     np.save(OUTPUT_FOLDER_NAME + '/' + '/train' + '/hog_images.npy', hog_images)


# np.save(OUTPUT_FOLDER_NAME + '/' + '/test' + '/images.npy', img_data_list_test)
# if ONE_HOT_ENCODING:
#     np.save(OUTPUT_FOLDER_NAME + '/' + '/test' + '/labels.npy', labels_list_test)
# else:
#     np.save(OUTPUT_FOLDER_NAME + '/' + '/test' + '/labels.npy', labels_list_test)
# if GET_LANDMARKS:
#     np.save(OUTPUT_FOLDER_NAME + '/' + '/test' + '/landmarks.npy', landmarks_test)
# if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
#     np.save(OUTPUT_FOLDER_NAME + '/' + '/test' + '/hog_features.npy', hog_features_test)
#     if GET_HOG_IMAGES:
#         np.save(OUTPUT_FOLDER_NAME + '/' + '/test' + '/hog_images.npy', hog_images_test)

    
