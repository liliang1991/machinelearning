#coding=utf-8
#一个小程序
import tensorflow as tf
import  numpy as np
#生成 100 个 随机点
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2
#构建一个线性模型
a=tf.Variable(0.)
b=tf.Variable(0.)
y=b*x_data+a


#二次代价函数
#1:求出一个误差的平方
#2:求出一个平均值

loss=tf.reduce_mean(tf.square(y_data-y))

#定义 一个梯度下降法进行训练的优化器
optimizer=tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数(loss)
train=optimizer.minimize(loss)
#初始化变量
init=tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)
     for step in range(201):
         sess.run(train)
         if step%20==0:
             print (step,sess.run([b,a]))