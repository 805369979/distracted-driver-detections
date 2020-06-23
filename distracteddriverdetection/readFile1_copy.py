import pandas as pd
import cv2
import os
import numpy as np
from skimage.feature import hog
from PIL import Image
ONE_HOT_ENCODING = True
GET_HOG_WINDOWS_FEATURES = True
GET_HOG_FEATURES = True


OUTPUT_FOLDER_NAME = "driver_features_img_1"


data_path_abs = 'img/valid_temp/'


original_labels = [0,1,2,3,4]
SELECTED_LABELS = [0,1,2,3,4]

new_labels= SELECTED_LABELS


def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)


img_data_list_train=[]
img_data_list_test=[]

labels_list_train = []
labels_list_test = []

hog_features_train = []
hog_features_test = []

hog_images_train = []
hog_images_test = []



img_list_all = os.listdir(data_path_abs)
print(img_list_all)
sums = 1
for i in img_list_all:
    count = 1
    train_count = 1

    img_list_alls = os.listdir(data_path_abs+i)
    # s = int(len(img_list_alls)*0.2)
    # print(s)
    for j in img_list_alls:
        # print(sums)
        # if count<=s:
        #     print(data_path_abs+"/"+i+"/"+j+"")
        #     im = Image.open(data_path_abs+"/"+i+"/"+j+"")
        #
        #     out = im.resize((224, 224), Image.ANTIALIAS)
        #     out.save('img/test1/%s/%d.jpg'%(i,count))
        #     print('img/test1/%s/%d.jpg' % (i, count))
        #     input_img = cv2.imread('img/test1/%s/%d.jpg' % (i, count))
        #     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            # image = cv2.resize(input_img, (224, 224))
            # img_data_list_train.append(input_img)
            # print(get_new_label(int(i[-1]), one_hot_encoding=ONE_HOT_ENCODING))
            # labels_list_train.append(get_new_label(int(i[-1]), one_hot_encoding=ONE_HOT_ENCODING))

            # if GET_HOG_FEATURES:
            #     feature= hog(input_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
            #             block_norm='L1', visualise=False, transform_sqrt=False)
            #     hog_features_train.append(feature)
            # count+=1
        # else:
        print(sums)
        # im = Image.open(data_path_abs+"/"+i+"/"+j+"")
        # print(data_path_abs + "/" + i + "/" + j + "")
        # out = im.resize((96, 54), Image.ANTIALIAS)
        # im.save('img/train1/%s/%d.jpg' % (i, train_count))
        # print('img/train1/%s/%d.jpg' % (i, train_count))
        #
        # input_img = cv2.imread('img/train1/%s/%d.jpg' % (i, train_count))
        # input_img1 = cv2.imread('img/train1/%s/%d.jpg' % (i, train_count))

        input_img = cv2.imread(data_path_abs+"/"+i+"/"+j+"")

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(input_img, (112, 112))
        img_data_list_test.append(image)

        # image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (224, 224))
        # data_path_abs + "/" + i + "/" + j + ""
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(input_img, (96, 54))
        # img_data_list_test.append(input_img)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        print(get_new_label(int(i[-1]), one_hot_encoding=ONE_HOT_ENCODING))
        labels_list_test.append(get_new_label(int(i[-1]), one_hot_encoding=ONE_HOT_ENCODING))
        #
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        if GET_HOG_FEATURES:
            feature = hog(input_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                           visualise=False, transform_sqrt=False)
            hog_features_test.append(feature)

        train_count+=1
        sums+=1



print(len(img_data_list_test))
print(len(labels_list_test))
print(len(hog_features_test))

print(len(img_data_list_train))
print(len(labels_list_train))
print(len(hog_features_train))
print(labels_list_test)
np.random.seed(2019)
np.random.shuffle(img_data_list_test)
np.random.seed(2019)
np.random.shuffle(labels_list_test)
np.random.seed(2019)
np.random.shuffle(hog_features_test)

print(labels_list_test)
# np.save(OUTPUT_FOLDER_NAME + '/' + "train" + '/images.npy', img_data_list_train)
# img_data_list_train=[]
# if ONE_HOT_ENCODING:
#     np.save(OUTPUT_FOLDER_NAME + '/' + "train" + '/labels.npy', labels_list_train)
# else:
#     np.save(OUTPUT_FOLDER_NAME + '/' + "train" + '/labels.npy', labels_list_train)
# labels_list_train=[]
# np.save(OUTPUT_FOLDER_NAME + '/' + "train" + '/hog_features.npy', hog_features_train)
#
# hog_features_train=[]


np.save(OUTPUT_FOLDER_NAME + '/' + "test" + '/images.npy', img_data_list_test)

if ONE_HOT_ENCODING:
    print('one-hot')
    np.save(OUTPUT_FOLDER_NAME + '/' + "test" + '/labels.npy', labels_list_test)
else:
    np.save(OUTPUT_FOLDER_NAME + '/' + "test" + '/labels.npy', labels_list_test)
np.save(OUTPUT_FOLDER_NAME + '/' + "test" + '/hog_features.npy', hog_features_test)



img_data_list_test = []
labels_list_test = []
hog_features_test = []
#
