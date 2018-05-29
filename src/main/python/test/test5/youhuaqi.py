#coding=utf-8
#非线性回归
import tensorflow as tf
#python 生成二维数组
import  numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
##载入数据集
mnist=input_data.read_data_sets("/home/moon/work/shendu/mnist_data",one_hot=True)
#每个批次的大小
batch_size=100
#计算有多少批次
n_batch=mnist.train.num_examples
print (n_batch)
##定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])
#0-9  标签十个
y=tf.placeholder(tf.float32,[None,10])

##创建神经网络
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(x,w)+b)
#二次代价函数
#loss=tf.reduce_mean(tf.square(y-prediction))


#交叉熵
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

##使用梯度下降法
train_step=tf.train.AdamOptimizer(1e-2).minimize(loss)
init=tf.global_variables_initializer()
#argmax 返回结果到一维张量中
corrct_pred=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
#true 转化为1.0 false转化为 0
accuracy=tf.reduce_mean(tf.cast(corrct_pred,tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print ("lter==="+str(epoch)+"accuracy==="+str(acc))

