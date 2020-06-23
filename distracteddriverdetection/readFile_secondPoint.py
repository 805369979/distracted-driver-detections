import pandas as pd
import cv2
import os
import numpy as np
from skimage.feature import hog

ONE_HOT_ENCODING = True
GET_HOG_WINDOWS_FEATURES = True
GET_HOG_FEATURES = True


OUTPUT_FOLDER_NAME = "driver_feature_small"
data_path = 'Train_data_list.csv'

data_path_abs = 'distracted.driver/'


original_labels = [0,1,2,3,4,5,6,7,8,9]
SELECTED_LABELS = [0,1,2,3,4,5,6,7,8,9]

new_labels= SELECTED_LABELS
print(new_labels)


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

landmarks_train = []
landmarks_test = []

hog_features_train = []
hog_features_test = []

hog_images_train = []
hog_images_test = []


for path in ['Train_data_list.csv','Test_data_list.csv']:
    print(path)

    train = pd.read_csv(path)

    train_data = train['Image']
    train_label = train['Label']

    print(len(train_data))

    count = 0
    for k, i in enumerate(train_data):
        print(k, i)
        name = i.split("/")[-1]
        class_data = i.split("/")[-2]
        print(class_data, name, train_label[k])

        img_list_all = os.listdir(data_path_abs + '/' + class_data)

        for d in img_list_all:
            if name == d:
                # print('/'.join(i.split('/')[1:]))
                input_img = cv2.imread('/'.join(i.split('/')[1:]))
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(input_img, (224, 224))
                image = image/255.0
                # img_data_list_train.append(image)
                img_data_list_test.append(image)
                print(get_new_label(train_label[k], one_hot_encoding=ONE_HOT_ENCODING))
                labels_list_test.append(get_new_label(train_label[k], one_hot_encoding=ONE_HOT_ENCODING))

                if GET_HOG_FEATURES:
                    features, hog_image = hog(image, orientations=9, pixels_per_cell=(10, 10),
                                              cells_per_block=(1, 1), visualise=True)
                    hog_features_test.append(features)
                count+=1
                break
    print(count)
    if path == 'Train_data_list.csv':
        save_n = '/train'
    else:
        save_n = '/test'
    np.random.seed(2019)
    np.random.shuffle(img_data_list_test)
    np.random.seed(2019)
    np.random.shuffle(labels_list_test)
    np.random.seed(2019)
    np.random.shuffle(hog_features_test)

    print(len(img_data_list_test))
    print(len(labels_list_test))
    print(len(hog_features_test))
    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/images.npy', img_data_list_test)

    if ONE_HOT_ENCODING:
        np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
    else:
        np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/hog_features.npy', hog_features_test)


    img_data_list_test = []
    labels_list_test = []
    hog_features_test = []

