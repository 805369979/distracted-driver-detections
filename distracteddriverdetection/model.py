"""
@AmineHorseman
Sep, 1st, 2016
"""
import tensorflow as tf
import pandas as pd



import keras.layers.advanced_activations


from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
import tflearn.layers.merge_ops
from tflearn.layers.merge_ops import merge_outputs, merge
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum, Adam
import numpy as np
import copy

# from parameters import NETWORK, HYPERPARAMS

from parameters import NETWORK, HYPERPARAMS
# from parametersJaffe import NETWORK, HYPERPARAMS
import sklearn
def build_model(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
    learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
    learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):

    if NETWORK.model == 'A':
        return build_modelA(optimizer, optimizer_param, learning_rate, keep_prob, learning_rate_decay, decay_step)
    elif NETWORK.model == 'B':
        return build_modelB(optimizer, optimizer_param, learning_rate, keep_prob, learning_rate_decay, decay_step)
    elif NETWORK.model == 'C':
        return build_modelC(optimizer, optimizer_param, learning_rate, keep_prob, learning_rate_decay, decay_step)
    elif NETWORK.model == 'D':
        return build_modelD(optimizer, optimizer_param, learning_rate, keep_prob, learning_rate_decay, decay_step)
    else:
        print( "ERROR: no model " + str(NETWORK.model))
        exit()

def build_modelB(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param,
    learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
    learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):
    img_prep = tflearn.ImagePreprocessing()

    # img_prep.add_featurewise_zero_center(per_channel=True)
    # img_prep.add_featurewise_stdnorm()
    img_prep.add_samplewise_zero_center()

    img_prep.add_image_normalization()
    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()

    # img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=5.0)
    # img_aug.add_random_blur(sigma_max=3.)

    img_aug.add_random_crop([224, 224], padding=4)
    images_network = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size1,1],data_preprocessing=img_prep,
                             data_augmentation=img_aug ,name='input1')

    images_network = conv_2d(images_network, 64, 3, activation=NETWORK.activation, regularizer="L2")
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)


    images_network4 = images_network
    # images_network4 = conv_2d(images_network4, 64, 1, activation=NETWORK.activation, regularizer="L2")
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network4 = batch_normalization(images_network4)
    #
    images_network4 = global_avg_pool(images_network4)

    images_network = max_pool_2d(images_network, 2, strides = 2)
    images_network = dropout(images_network, keep_prob=0.9)

    # images_network = conv_2d(images_network, 128, 3, activation=NETWORK.activation, regularizer="L2",weights_init='truncated_normal')
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 128, 3, activation=NETWORK.activation, regularizer="L2")
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    # images_network = conv_2d(images_network, 128,  3, activation=NETWORK.activation, regularizer="L2")
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)



    images_network2 = images_network
    # images_network2 = conv_2d(images_network2, 128, 1, activation=NETWORK.activation, regularizer="L2",weights_init='truncated_normal')
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network2 = batch_normalization(images_network2)

    images_network2 = global_avg_pool(images_network2)

    images_network = max_pool_2d(images_network, 2, strides = 2)
    images_network = dropout(images_network, keep_prob=0.8)


    # images_network = conv_2d(images_network, 256, 3, activation=NETWORK.activation,regularizer="L2",weights_init='truncated_normal')
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    # images_network = conv_2d(images_network, 256, 3, activation=NETWORK.activation, regularizer="L2",weights_init='truncated_normal')
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 256,  3, activation=NETWORK.activation, regularizer="L2")
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)


    images_network3 = images_network
    # images_network3 = conv_2d(images_network3, 256, 1, activation=NETWORK.activation, regularizer="L2",weights_init='truncated_normal')
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network3 = batch_normalization(images_network3)
    #
    images_network3 = global_avg_pool(images_network3)

    images_network = max_pool_2d(images_network, 2, strides=2)

    images_network = dropout(images_network, keep_prob=0.7)


    # images_network = conv_2d(images_network, 512, 3, activation=NETWORK.activation,regularizer="L2",weights_init='truncated_normal')
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    # images_network = conv_2d(images_network, 512, 3, activation=NETWORK.activation, regularizer="L2",weights_init='truncated_normal')
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 512,  3, activation=NETWORK.activation, regularizer="L2")
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)

    # images_network = conv_2d(images_network, 512, 1, activation=NETWORK.activation, regularizer="L2",weights_init='truncated_normal')
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)

    images_network = dropout(images_network, keep_prob=0.6)

    images_network = global_avg_pool(images_network)

    # images_network = global_avg_pool(images_network)
    # images_network = dropout(images_network, keep_prob=0.5)


    #========================================================================================
    #========================================================================================
    #========================================================================================
    # print(type(images_network))
    # # images_network2 = copy.deepcopy(images_network)
    # # print(id(images_network),id(images_network2))
    # images_network2 = images_network
    #
    # if images_network == images_network2:
    #     print(True)
    # else:
    #     print(False)
    # images_network2 = global_avg_pool(images_network)
    #
    # if images_network == images_network2:
    #     print(True)
    # else:
    #     print(False)
    #
    # print(id(images_network), id(images_network2))
    # images_network = dropout(images_network, keep_prob=0.5)
    # print(id(images_network), id(images_network2))
    # images_network = conv_2d(images_network, 512, 3, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    # images_network = conv_2d(images_network, 512, 3, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    # images_network = conv_2d(images_network, 512, 3, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    #
    # # ========================================================================================
    # # ========================================================================================
    #  # ========================================================================================
    # images_network3 = images_network
    # images_network3 = global_avg_pool(images_network3)
    # print(id(images_network), id(images_network3))
    # images_network = dropout(images_network, keep_prob=0.5)
    # print(id(images_network), id(images_network3))
    #
    # images_network = conv_2d(images_network, 512, 3, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    # images_network = conv_2d(images_network, 512, 3, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    # images_network = conv_2d(images_network, 512, 3, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network = batch_normalization(images_network)
    #
    # # images_network = dropout(images_network, keep_prob=0.5)
    # images_network = global_avg_pool(images_network)

    if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
        if NETWORK.use_hog_sliding_window_and_landmarks:
            landmarks_network = input_data(shape=[None, 2916], name='input2')
            if NETWORK.use_batchnorm_after_fully_connected_layers:
                landmarks_network = batch_normalization(landmarks_network)
            print("进来了")

            # landmarks_network = input_data(shape=[None, 119208], name='input2')
        elif NETWORK.use_hog_and_landmarks:
            landmarks_network = input_data(shape=[None, 2916], name='input2')
            print("=============================================================")
        else:
            landmarks_network = input_data(shape=[None, 68, 2], name='input2')
        landmarks_network = fully_connected(landmarks_network, 1024, activation=NETWORK.activation,regularizer="L2")
        # landmarks_network = fully_connected(landmarks_network, 1024, activation=NETWORK.activation)
        if NETWORK.use_batchnorm_after_fully_connected_layers:
            landmarks_network = batch_normalization(landmarks_network)


        network = merge([images_network,images_network2,images_network3,images_network4, landmarks_network], 'concat', axis=1)
    else:
        network = images_network
    # network = fully_connected(network, 1024, activation=NETWORK.activation,regularizer="L2")
    network = fully_connected(network, 1024, activation=NETWORK.activation,regularizer="L2")
    if NETWORK.use_batchnorm_after_fully_connected_layers:
        network = batch_normalization(network)
    # network = dropout(network, keep_prob=0.5)
    network = fully_connected(network, NETWORK.output_size, activation='softmax')

    if optimizer == 'momentum':
        optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param,
                    lr_decay=learning_rate_decay, decay_step=decay_step)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
    else:
        print( "Unknown optimizer: {}".format(optimizer))
    network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')

    return network

# def build_modelB(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param,
#     learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
#     learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):
#
#     images_network = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size,1], name='input1')
#     # images_network = conv_2d(images_network, 64, 3, activation=NETWORK.activation,regularizer='L2', weight_decay=0.0001)
#     # if NETWORK.use_batchnorm_after_conv_layers:
#     #     images_network = batch_normalization(images_network)
#     images_network = conv_2d(images_network, 64, 3, activation=NETWORK.activation)
#     images_network = max_pool_2d(images_network, 3, strides = 2)
#     if NETWORK.use_batchnorm_after_conv_layers:
#         images_network = batch_normalization(images_network)
#
#     # images_network = conv_2d(images_network, 128, 3, activation=NETWORK.activation,regularizer='L2', weight_decay=0.0001)
#     # if NETWORK.use_batchnorm_after_conv_layers:
#     #     images_network = batch_normalization(images_network)
#     images_network = conv_2d(images_network, 128, 3, activation=NETWORK.activation)
#
#     images_network = max_pool_2d(images_network, 3, strides = 2)
#     if NETWORK.use_batchnorm_after_conv_layers:
#         images_network = batch_normalization(images_network)
#
#     images_network = dropout(images_network, keep_prob=0.5)
#
#     images_network = conv_2d(images_network, 256, 3, activation=NETWORK.activation)
#     # if NETWORK.use_batchnorm_after_conv_layers:
#     #     images_network = batch_normalization(images_network)
#     # images_network = conv_2d(images_network, 256, 3, activation=NETWORK.activation)
#     # if NETWORK.use_batchnorm_after_conv_layers:
#     #     images_network = batch_normalization(images_network)
#     # images_network = conv_2d(images_network, 256, 1, activation=NETWORK.activation)
#     # if NETWORK.use_batchnorm_after_conv_layers:
#     #     images_network = batch_normalization(images_network)
#     # images_network = max_pool_2d(images_network, 3, strides = 2)
#     # images_network = dropout(images_network, keep_prob=0.75)
#
#     # images_network = fully_connected(images_network, 4096, activation=NETWORK.activation)
#     # images_network = dropout(images_network, keep_prob=0.75)
#     # images_network = fully_connected(images_network, 1024, activation=NETWORK.activation)
#     # if NETWORK.use_batchnorm_after_fully_connected_layers:
#     #     images_network = batch_normalization(images_network)
#     # images_network = dropout(images_network, keep_prob=0.85)
#
#     if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
#         if NETWORK.use_hog_sliding_window_and_landmarks:
#             landmarks_network = input_data(shape=[None, 51200], name='input2')
#             print("进来了")
#             # landmarks_network = input_data(shape=[None, 119208], name='input2')
#         elif NETWORK.use_hog_and_landmarks:
#             landmarks_network = input_data(shape=[None, 51200], name='input2')
#             print("=============================================================")
#         else:
#             landmarks_network = input_data(shape=[None, 68, 2], name='input2')
#         landmarks_network = fully_connected(landmarks_network, 2048, activation=NETWORK.activation)
#         # landmarks_network = fully_connected(landmarks_network, 512, activation=NETWORK.activation)
#         # if NETWORK.use_batchnorm_after_fully_connected_layers:
#         #     landmarks_network = batch_normalization(landmarks_network)
#         landmarks_network = fully_connected(landmarks_network, 128, activation=NETWORK.activation)
#         if NETWORK.use_batchnorm_after_fully_connected_layers:
#             landmarks_network = batch_normalization(landmarks_network)
#
#         # landmarks_network =global_avg_pool(landmarks_network)
#         images_network =global_avg_pool(images_network)
#         images_network = fully_connected(images_network, 128, activation=NETWORK.activation)
#         network = merge([images_network, landmarks_network], 'concat', axis=1)
#     # else:
#     #     network = images_network
#     network = fully_connected(network, NETWORK.output_size, activation='softmax')
#
#     if optimizer == 'momentum':
#         optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param,
#                     lr_decay=learning_rate_decay, decay_step=decay_step)
#     elif optimizer == 'adam':
#         optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
#     else:
#         print( "Unknown optimizer: {}".format(optimizer))
#     network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')
#
#     return network


def build_modelA(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param,
                 learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
                 learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_image_normalization()
    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()

    # img_aug.add_random_flip_leftright()
    # img_aug.add_random_rotation(max_angle=10.0)
    # img_aug.add_random_blur(sigma_max=3.)
    # data_preprocessing
    #     img_aug.add_random_crop([224, 224],8)
    images_network = input_data(shape=(NETWORK.input_size, NETWORK.input_size1, 1),
                                data_augmentation=img_aug, data_preprocessing=img_prep, name='input1')


    images_network = conv_2d(images_network, 32, 12, regularizer="L2",weights_init='truncated_normal', activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides = 2)
    # images_network = dropout(images_network, keep_prob=0.9)
    #
    images_network = conv_2d(images_network, 64, 9, regularizer="L2",weights_init='truncated_normal',activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides = 2)
    # images_network = dropout(images_network, keep_prob=0.8)

    images_network = conv_2d(images_network, 128, 6,regularizer="L2",weights_init='truncated_normal', activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides = 2)
    # images_network = dropout(images_network, keep_prob=0.7)


    images_network = conv_2d(images_network, 256, 3,regularizer="L2",weights_init='truncated_normal', activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides = 2)
    images_network = dropout(images_network, keep_prob=0.5)


    images_network = global_avg_pool(images_network)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    # images_network = fully_connected(images_network, 4096, activation=NETWORK.activation)
    # images_network = dropout(images_network, keep_prob=0.5)
    # images_network = fully_connected(images_network, 1024, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_fully_connected_layers:
    #     images_network = batch_normalization(images_network)
    # images_network = dropout(images_network, keep_prob=0.9)
    #
    # network = fully_connected(images_network, 1024, activation=NETWORK.activation, regularizer="L2")
    network = fully_connected(images_network, NETWORK.output_size,regularizer="L2",weights_init='truncated_normal',activation='softmax')

    if optimizer == 'momentum':
        optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param,
                             lr_decay=learning_rate_decay, decay_step=decay_step)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
    else:
        print("Unknown optimizer: {}".format(optimizer))
    network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')

    return network


def build_modelC(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param,
    learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
    learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):

    images_network = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size, 1], name='input1')
    images_network2 = images_network
    images_network5 = images_network


    images_network = conv_2d(images_network, 64, 3, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 64, 3, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides = 2)

    images_network = conv_2d(images_network, 128, 3, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 128, 3, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides = 2)
    images_network = dropout(images_network, keep_prob=0.7)

    images_network = conv_2d(images_network, 256, 3, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 256, 3, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides = 2)
    images_network = dropout(images_network, keep_prob=0.6)

    images_network = fully_connected(images_network, 4096, activation=NETWORK.activation)
    images_network = dropout(images_network, keep_prob=0.5)
    images_network = fully_connected(images_network, 1024, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_fully_connected_layers:
        images_network = batch_normalization(images_network)
    images_network = dropout(images_network, keep_prob=0.8)
#"================================5*5卷积核============================================"
    # images_network5 = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size, 1], name='input2')
    images_network5 = conv_2d(images_network5, 64, 2, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network5 = batch_normalization(images_network5)
    images_network5 = conv_2d(images_network5, 64, 2, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network5 = batch_normalization(images_network5)
    images_network5 = max_pool_2d(images_network5, 2, strides = 2)

    images_network5 = conv_2d(images_network5, 128, 2, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network5 = batch_normalization(images_network5)
    images_network5 = conv_2d(images_network5, 128, 2, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network5 = batch_normalization(images_network5)
    images_network5 = max_pool_2d(images_network5, 2, strides = 2)
    images_network5 = dropout(images_network5, keep_prob=0.7)

    images_network5 = conv_2d(images_network5, 256, 2, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network5 = batch_normalization(images_network5)
    images_network5 = conv_2d(images_network5, 256, 2, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network5 = batch_normalization(images_network5)
    images_network5 = max_pool_2d(images_network5, 2, strides = 2)
    images_network5 = dropout(images_network5, keep_prob=0.6)

    images_network5 = fully_connected(images_network5, 4096, activation=NETWORK.activation)
    images_network5 = dropout(images_network5, keep_prob=0.5)
    images_network5= fully_connected(images_network5, 1024, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_fully_connected_layers:
        images_network5 = batch_normalization(images_network5)
    images_network5 = dropout(images_network5, keep_prob=0.8)
#==============================2*2==========================================="
    # images_network2 = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size, 1], name='input3')
    images_network2 = conv_2d(images_network2, 64, 5, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network2 = batch_normalization(images_network2)
    images_network2 = conv_2d(images_network2, 64, 5, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network2 = batch_normalization(images_network2)
    images_network2 = max_pool_2d(images_network2, 2, strides=2)

    images_network2 = conv_2d(images_network2, 128, 5, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network2 = batch_normalization(images_network2)
    images_network2 = conv_2d(images_network2, 128, 5, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network2 = batch_normalization(images_network2)
    images_network2 = max_pool_2d(images_network2, 2, strides=2)
    # images_network2 = conv_2d(images_network2, 256, 5, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network2 = batch_normalization(images_network2)
    # images_network2 = conv_2d(images_network2, 256, 5, activation=NETWORK.activation)
    # if NETWORK.use_batchnorm_after_conv_layers:
    #     images_network2 = batch_normalization(images_network2)
    # images_network2 = max_pool_2d(images_network2, 2, strides=2)
    images_network2 = dropout(images_network2, keep_prob=0.65)

    images_network2 = fully_connected(images_network2, 4096, activation=NETWORK.activation)
    images_network2 = dropout(images_network2, keep_prob=0.5)
    images_network2 = fully_connected(images_network2, 1024, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_fully_connected_layers:
        images_network2 = batch_normalization(images_network2)
    images_network2 = dropout(images_network2, keep_prob=0.5)
#=======================================================================================
    if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
        if NETWORK.use_hog_sliding_window_and_landmarks:
            landmarks_network = input_data(shape=[None, 2728], name='input5')
        elif NETWORK.use_hog_and_landmarks:
            landmarks_network = input_data(shape=[None, 208], name='input5')
        else:
            landmarks_network = input_data(shape=[None, 68, 2], name='input5')
        landmarks_network = fully_connected(landmarks_network, 1024, activation=NETWORK.activation)
        if NETWORK.use_batchnorm_after_fully_connected_layers:
            landmarks_network = batch_normalization(landmarks_network)
        landmarks_network = fully_connected(landmarks_network, 128, activation=NETWORK.activation)
        if NETWORK.use_batchnorm_after_fully_connected_layers:
            landmarks_network = batch_normalization(landmarks_network)
        images_network = fully_connected(images_network, 128, activation=NETWORK.activation)
        images_network2 = fully_connected(images_network2, 128, activation=NETWORK.activation)
        images_network5 = fully_connected(images_network5, 128, activation=NETWORK.activation)
        network = merge([images_network, landmarks_network,images_network5,images_network2], 'concat', axis=1)
    else:
        network = images_network
    network = fully_connected(network, NETWORK.output_size, activation='softmax')

    if optimizer == 'momentum':
        optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param,
                    lr_decay=learning_rate_decay, decay_step=decay_step)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
    else:
        print( "Unknown optimizer: {}".format(optimizer))
    network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')

    return network


def build_modelD(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param,
    learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
    learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):
    n = 5
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)
    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([224, 224], padding=4)

    # Building Residual Network
    net = tflearn.input_data(shape=[None, 224,224, 1],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug,name='input1')
    net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, n-1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, n-1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    # net = tflearn.fully_connected(net, numClass, activation='softmax')
    # mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    # net = tflearn.regression(net, optimizer=mom,
    #                          loss='categorical_crossentropy')
    # # Training
    # model = tflearn.DNN(net, checkpoint_path='model_resnet_mymodel',
    #                     max_checkpoints=10, tensorboard_verbose=0,
    #                     clip_gradients=0.)

    # model.fit(X, Y, n_epoch=200, validation_set=(validationX, validationY),
    #           snapshot_epoch=False, snapshot_step=500,
    #           show_metric=True, batch_size=128, shuffle=True,
    #           run_id='resnet_mymodel')
    net = fully_connected(net, NETWORK.output_size, activation='softmax')
    mom = tflearn.Momentum(0.01, lr_decay=0.9, decay_step=32000, staircase=True)
    net = tflearn.regression(net, optimizer=mom,
                             loss='categorical_crossentropy')

    # if optimizer == 'momentum':
    #     optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param,
    #                          lr_decay=learning_rate_decay, decay_step=decay_step)
    # elif optimizer == 'adam':
    #     optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
    # else:
    #     print("Unknown optimizer: {}".format(optimizer))
    # network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')

    return net
