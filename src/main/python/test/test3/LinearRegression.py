#coding=utf-8
#非线性回归
import tensorflow as tf
#python 生成二维数组
import  numpy as np
import matplotlib.pyplot as plt
#使用 numpy 生成 200个随机点
#np.linspace 在指定的间隔内返回均匀间隔的数字。
#1param start 2param stop 3param  生成的个数   np.newaxis 列的别名
xdata=np.linspace(-0.5,0.5,200)[:,np.newaxis]

noise=np.random.normal(0,0.02,xdata.shape)
#print noise
ydata=np.square(xdata)+noise
##定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义神经网络的中间层
weights=tf.Variable(tf.random_normal([1,10]))
biases=tf.Variable(tf.zeros([1,10]))
wx_plus=tf.matmul(x,weights)+biases
l1=tf.nn.tanh(wx_plus)
##定义神经网络输出层
weights_out=tf.Variable(tf.random_normal([10,1]))
biases_out=tf.Variable(tf.zeros([1,1]))
wx_plus_out=tf.matmul(l1,weights_out)+biases_out
prediction=tf.nn.tanh(wx_plus_out)
#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
##使用梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:xdata,y:ydata})
        #获取预测值
    prediction_value=sess.run(prediction,feed_dict={x:xdata,y:ydata})
    #画图
    plt.figure()
    plt.scatter(xdata,ydata)
    plt.plot(xdata,prediction_value,'r-',lw=5)
    plt.show()


