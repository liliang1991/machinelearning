##lenet-5 网络 demo
import tensorboard
import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
import datetime
if __name__ == '__main__':
    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6,kernel_size=3,strides=1), # 第一个卷积层, 6个3x3卷积核
        tf.keras.layers.MaxPooling2D(pool_size=2,strides=2),# 高宽各减半的池化层
        tf.keras.layers.ReLU(),# 激活函数
        tf.keras.layers.Conv2D(16,kernel_size=3,strides=1), # 第二个卷积层, 16个3x3卷积核
        tf.keras.layers.MaxPooling2D(pool_size=2,strides=2),# 高宽各减半的池化层
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),# 打平层，方便全连接层处理
        tf.keras.layers.Dense(120,activation='relu'),# 全连接层，120个节点
        tf.keras.layers.Dense(84,activation='relu'),# 全连接层，84节点 layers.Dense(10) # 全连接层，10个节点

    ])
    # build一次网络模型，给输入X的形状，其中4为随意给的batchsz
    model.build(input_shape=(4,28,28,1))
    model.summary()

    ###处理数据
    # 读取数据集,28*28像素1通道mnist图片，标签为10类
    (train_images,train_lables),(test_images,test_lables)=tf.keras.datasets.mnist.load_data()
    train_images=tf.reshape(train_images,(train_images.shape[0],train_images.shape[1],train_images.shape[2],1))
    print(train_images.shape)
    test_images=tf.reshape(test_images,(test_images.shape[0],test_images.shape[1],test_images.shape[2],1))
    train_images=train_images / 255
    train_lables=train_lables / 255

    ##模型训练
    model.compile(optimizer=tf.keras.optimizers
                  .SGD(learning_rate=0.01),loss="sparse_categorical_crossentropy",metrics=['acc'])
    history=model.fit(train_images,train_lables,batch_size=5,epochs=5,validation_split=0.1,verbose=1)
    ##模型评估
    model.evaluate(test_images,test_lables)