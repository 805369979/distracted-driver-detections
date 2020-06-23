# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
# # # y1_b = [0.245,0.245,0.245,0.245,0.245,0.245,0.690,0.690,1.118,1.118,1.118,0.690,0.690,0.690,0.690,0.690,0.690,0.690,0.690,1.118,1.118,1.118,0.690,0.245,0.245]
# # # y1_bs = [0.245,0.245,0.245,0.245,0.245,0.245,0.690,0.690,1.118,1.118,1.118,0.690,0.690,1.118,1.118,0.690,0.690,0.690,1.118,1.118,1.118,0.690,0.245,0.245]
# # # y2_ss = [0.214,0.214,0.214,0.214,0.214,0.214,0.607,0.607,0.995,0.995,0.995,0.607,0.607,0.995,0.995,0.607,0.607,0.607,0.995,0.995,0.995,0.607,0.214,0.214]
# # #
# # # y2_s =  [0.214,0.214,0.214,0.214,0.214,0.214,0.607,0.607,0.995,0.995,0.995,0.607,0.607,0.607,0.607,0.607,0.607,0.607,0.607,0.995,0.995,0.995,0.607,0.214,0.214]
# # #
# # # print(len(y1_bs))
# # # print(len(x))
# # # plt.figure(num=1, figsize=(11, 6))
# # # ax = plt.subplot(111)
# # # plt.grid()
# # # plt.xticks(range(0,26,1))
# # # plt.yticks(np.arange(0, 1.6, step=0.1))
# # # plt.rcParams['font.sans-serif'] = ['SimHei']#可以解释中文无法显示的问题
# # # for i in range(24):
# # #     print(i)
# # #     c = [i,i+1]
# # #     yc = [y1_bs[i],y1_bs[i]]
# # #     line = ax.plot(c,yc,color="blue")
# # #     if i >0:
# # #         cx = [i, i]
# # #         ycx = [y1_bs[i-1], y1_bs[i]]
# # #         ax.plot(cx,ycx,color="blue")
# # #     y=[]
# # # # for j in range(24):
# # # #     print(j)
# # # #     c = [j,j+1]
# # # #     yc = [y1_b[j],y1_b[j]]
# # # #     line = ax.plot(c,yc,color="blue",label="sdhvusf")
# # # #     if j >0:
# # # #         cx = [j, j]
# # # #         ycx = [y1_b[j-1], y1_b[j]]
# # # #         plt.plot(cx,ycx,color="blue")
# # # #     y=[]
# # # for i in range(24):
# # #     print(i)
# # #     c = [i,i+1]
# # #     yc = [y2_ss[i],y2_ss[i]]
# # #     line1 = ax.plot(c,yc,color="black")
# # #     if i >0:
# # #         cx = [i, i]
# # #         ycx = [y2_ss[i-1], y2_ss[i]]
# # #         ax.plot(cx,ycx,color="black")
# # #     y=[]
# # # # for x in range(24):
# # # #     print(x)
# # # #     c = [x,x+1]
# # # #     yc = [y2_s[x],y2_s[x]]
# # # #     line = ax.plot(c,yc,color="green",label="sdhvusf")
# # # #     if x >0:
# # # #         cx = [x, x]
# # # #         ycx = [y2_s[x-1], y2_s[x]]
# # # #         plt.plot(cx,ycx,color="green")
# # # #     y=[]
# # # # y2 = [0.995,0.607,0.214]
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # #
# # # # markes = ['-o', '-s', '-^', '-p', '-^', '-v', '-p', '-d', '-h', '-2', '-8', '-6']
# # # # plt.bar(x,y1,label = '大工业用电',color="w",edgecolor="k")
# # # # plt.plot(x,y2,markes[1],label = '一般工商业及其他用电')
# # # plt.legend(["大工业非夏季用电","一般工商业及其他非夏季用电(绿色线)"],loc='upper left')
# # # # plt.legend(["大工业夏季用电","一般工商业及其他夏季用电(黑色线)"],loc='upper left')
# # #
# # # # ax.legend(["夏季","非夏季"])
# # # -*- coding: utf-8 -*-
# # import numpy as np
# # np.random.seed(1337) #for reproducibility再现性
# # from keras.datasets import mnist
# # from keras.utils import np_utils
# # from keras.models import Sequential#按层
# # from keras.layers import Dense, Activation#全连接层
# # import matplotlib.pyplot as plt
# # from keras.optimizers import RMSprop
# #
# # #dowmload the mnisst the path '~/.keras/datasets/' if it is the first time to be called
# # #x shape (60000 28*28),y shape(10000,)
# # (x_train,y_train),(x_test,y_test) = mnist.load_data()#0-9的图片数据集
# #
# # #data pre-processing
# # x_train = x_train.reshape(x_train.shape[0],-1)/255 #normalize 到【0,1】
# # x_test = x_test.reshape(x_test.shape[0],-1)/255
# # y_train = np_utils.to_categorical(y_train, num_classes=10) #把标签变为10个长度，若为1，则在1处为1，剩下的都标为0
# # y_test = np_utils.to_categorical(y_test,num_classes=10)
# #
# # #Another way to build neural net
# # model = Sequential([
# #         Dense(32,input_dim=784),#传出32
# #         Activation('relu'),
# #
# #
# #         Dense(10),
# #         Activation('softmax')
# #         ])
# #
# # #Another way to define optimizer
# # rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
# #
# # # We add metrics to get more results you want to see
# # model.compile( #编译
# #         optimizer = rmsprop,
# #         loss = 'categorical_crossentropy',
# #         metrics=['accuracy'], #在更新时同时计算一下accuracy
# #         )
# #
# # print("Training~~~~~~~~")
# # #Another way to train the model
# # model.fit(x_train,y_train, epochs=2, batch_size=32) #训练2大批，每批32个
# #
# # print("\nTesting~~~~~~~~~~")
# # #Evalute the model with the  metrics we define earlier
# # loss,accuracy = model.evaluate(x_test,y_test)
# #
# # print('test loss:',loss)
# # print('test accuracy:', accuracy)
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(-10,10)
# y_sigmoid = 1/(1+np.exp(-x))
#
# y3 = y_sigmoid*x
#
# y = 2 *(1/(1+np.exp(-x)))-1
#
# y_tanh =(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
# y_sigmoid = y_sigmoid*(x+0.5)
# fig = plt.figure()
#
#
#
# # plot sigmoid
# ax = fig.add_subplot(231)
# ax.plot(x,y_sigmoid)
# ax.grid()
# ax.set_title('(a) Sigmoid')
#
# ax = fig.add_subplot(232)
# ax.plot(x,y)
# ax.grid()
# ax.set_title('(a) Sigmoid')
# # plot tanh
# ax = fig.add_subplot(233)
# ax.plot(x,y_tanh)
# ax.grid()
# ax.set_title('(b) Tanh')
#
# # plot relu
# ax = fig.add_subplot(234)
# y_relu = np.array([0*item  if item<0 else item for item in x ])
# ax.plot(x,y_relu)
# ax.grid()
# ax.set_title('(c) ReLu')
#
# #plot leaky relu
# ax = fig.add_subplot(235)
# y_relu = np.array([0.2*item  if item<0 else item for item in x ])
# ax.plot(x,y_relu)
# ax.grid()
# ax.set_title('(d) Leaky ReLu')
#
# ax = fig.add_subplot(236)
# ax.plot(x,y3)
# ax.grid()
# ax.set_title('(d) Leaky ReLu')
#
# plt.tight_layout()
# plt.show()

import os
# path = 'img/test1'
# print(len(os.listdir(path)))
# sums = 0
# for i in os.listdir(path):
#     sums += len(os.listdir(path+'/%s' % i))
#     print(i)
#     print(len(os.listdir(path+'/%s' % i)))
# print(sums)

for i in range(5):
    print("sdfdsfdsfdsfds")
    os.mkdir('img/test1/' + "/c%d" % i)
import numpy as np
# from parameters import DATASET, NETWORK
# import numpy as np
#
# x = np.load(DATASET.train_folder + '/hog_features.npy')
# x.reshape([-1,224,224,1])
# print(len(x))