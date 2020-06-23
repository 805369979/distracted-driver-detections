import pandas as pd
import cv2
import os
import numpy as np
from PIL import Image
from skimage.feature import hog

ONE_HOT_ENCODING = True
GET_HOG_WINDOWS_FEATURES = True
GET_HOG_FEATURES = True


OUTPUT_FOLDER_NAME = "driver_feature_tranform"
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


img_data_list_train1=[]
img_data_list_train2=[]
img_data_list_train3=[]
img_data_list_test=[]

labels_list_train = []
labels_list_test = []

landmarks_train = []
landmarks_test = []

hog_features_train = []
hog_features_test = []

hog_images_train = []
hog_images_test = []

from keras.models import Model
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
model1 = Model(inputs=model.input,outputs=model.layers[5].output)
model2 = Model(inputs=model.input,outputs=model.layers[9].output)
model3 = Model(inputs=model.input,outputs=model.layers[13].output)



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

                im = Image.open('/'.join(i.split('/')[1:]))

                out = im.resize((224, 224), Image.ANTIALIAS)
                out.save('data_test/1.jpg')

                # input_img = cv2.imread('/'.join(i.split('/')[1:]))
                input_img = cv2.imread('data_test/1.jpg')

                input_imgs =np.expand_dims(input_img,axis=0)
                model1_pred = model1.predict(input_imgs)



                # model2_pred = model2.predict(input_imgs)
                # model3_pred = model3.predict(input_imgs)
                #
                img_data_list_train1.append(model1_pred)
                # img_data_list_train2.append(model2_pred)
                # img_data_list_train3.append(model3_pred)

                # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

                # image = cv2.resize(input_img, (224, 224))
                # img_data_list_train.append(image)
                # img_data_list_test.append(image)
                # print(get_new_label(train_label[k], one_hot_encoding=ONE_HOT_ENCODING))
                # labels_list_test.append(get_new_label(train_label[k], one_hot_encoding=ONE_HOT_ENCODING))
                #
                # if GET_HOG_FEATURES:
                #     features, hog_image = hog(input_img, orientations=8, pixels_per_cell=(64, 64),
                #                               cells_per_block=(1, 1), visualise=True)
                #     hog_features_test.append(features)
                count+=1
                break


    print(count)
    if path == 'Train_data_list.csv':
        save_n = '/train'
    else:
        save_n = '/test'

    # print(len(img_data_list_test))
    print(len(img_data_list_train1))
    print(len(img_data_list_train2))
    print(len(img_data_list_train3))
    print(len(labels_list_test))
    print(len(hog_features_test))
    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/images1.npy', img_data_list_train1)
    # np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/images2.npy', img_data_list_train2)
    # np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/images3.npy', img_data_list_train3)

    # if ONE_HOT_ENCODING:
    #     np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
    # else:
    #     np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
    # np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/hog_features.npy', hog_features_test)

    img_data_list_train1=[]
    img_data_list_train2=[]
    img_data_list_train3=[]
    labels_list_test = []
    hog_features_test = []

