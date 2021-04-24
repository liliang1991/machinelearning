#导入库
import tensorflow  as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from learning.mnist_learing.vgg16 import VGG16
from learning.mnist_learing.model import *
#获取数据集属性函数
def get_data_info(mnist_images,mnist_labels):
    image_shape=mnist_images.shape
    labels_shape=mnist_labels.shape
    image_len=len(mnist_images)
    labels_len=len(mnist_labels)
    first_data=mnist_images[0]
    first_lable=mnist_labels[0]
    return image_shape,labels_shape,image_len,labels_len,first_data,first_lable
if __name__ == '__main__':
    mnist=keras.datasets.mnist
    (train_images,train_labels),(test_images,test_labels)=mnist.load_data()
    tr_image_shape,tr_label_shape,tr_image_len,tr_label_len,tr_image_data,tr_label_data=get_data_info(train_images,train_labels)
    #打印训练数据信息
    print("训练图像数据集的尺寸：{}".format(tr_image_shape))
    print("训练图像标签尺寸：{}".format(tr_label_shape))
    print("训练图像数量：{}".format(tr_image_len))
    print("训练图像标签数量：{}".format(tr_label_len))
    print("第一个训练图像数据：{}".format(tr_image_data))
    print("第一个训练图像标签数据：{}".format(tr_label_data))
    #显示示例图像
    plt.figure(figsize=(2,2))
    for i in range(4):
       train_data=train_images[i]
       train_label=train_labels[i]
       plt.subplot(2,2,i+1)
       plt.subplots_adjust(wspace=0.5,hspace=0.8)
       plt.imshow(train_data)
       plt.title("label {}".format(train_label),)

    plt.show()

    #测试数据集
    test_image_shape,test_label_shape,test_image_len,test_label_len,test_image_data,test_label_data=get_data_info(test_images,test_labels)
    #打印测试集信息
    print("测试图像数据集的尺寸：{}".format(test_image_shape))
    print("测试图像标签尺寸：{}".format(test_label_shape))
    print("测试图像数量：{}".format(test_image_len))
    print("测试图像标签数量：{}".format(test_label_len))
    print("第一个测试图像数据：{}".format(test_image_data))
    print("第一个测试图像标签数据：{}".format(test_label_data))
    #显示示例图像
    plt.figure(figsize=(2,2))
    for i in range(4):
       test_data=test_images[i]
       test_label=test_labels[i]
       plt.subplot(2,2,i+1)
       plt.subplots_adjust(wspace=0.5,hspace=0.8)
       plt.imshow(test_data)
       plt.title("label {}".format(test_label),)
    plt.show()


    ###卷积网络搭建
    print("###卷积网络搭建")
    #shape: 输出张量的形状，必选
    #mean: 正态分布的均值，默认为0
    #stddev: 正态分布的标准差，默认为1.0
    #dtype: 输出的类型，默认为tf.float32
    #seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    #name: 操作的名称
    ##https://zhuanlan.zhihu.com/p/26139876(tf.nn.conv2d()的工作方法)

    x = tf.random.normal([2,5,5,3]) # 模拟输入，3通道，高宽为5
    w = tf.random.normal([3,3,3,4]) # 需要根据[k,k,cin,cout]格式创建，4个3x3大小卷积核
    out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[0,0],[0,0],[0,0]]) # 步长为1, padding为0,
    print(x)
    print('-'*50)

    print(w)
    print('-'*50)
    print(out)


    #训练集预处理
    train_images=train_images[:1000].reshape(-1,28,28,1)/255.0
    train_label=train_labels[:1000]
    #测试集预处理
    test_imagees=test_images[:100].reshape(-1,28,28,1)/255.0
    test_labels=test_labels[:100]

    model=VGG16()#创建网络
    model.summary()
    base_name=str(datetime.now().strftime("%Y%m%d_%H_%M_%S"))
    print(base_name)
    #模型保存路径
    model_path1 = "mnist_cnn"+base_name+".h5"
    print(model_path1)
    #启动训练
    train_modell(model,train_images,train_label,model_path1)

    ###模型推理
    model=load_model(model_path1)#加载网络
    test_image = tf.convert_to_tensor([test_imagees[2]])#数据转为tensor
    pre = model.predict(test_image)#模型预测
    pre = tf.math.argmax(pre, 1)#获取预测结果  提示one_hot的应用
    #打印预测结果和标签
    print("prediction:{}".format(pre))
    plt.imshow(test_imagees[2])
    plt.title("label {}".format(test_labels[2]),)
    plt.show()