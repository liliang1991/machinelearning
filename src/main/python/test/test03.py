#coding=utf-8
#fetch and feed
#fetch进行多个 op
#feed 在 进行 op  的时候传参
import  tensorflow as tf
input1=tf.constant(2)
input2=tf.constant(3)
input3=tf.constant(4)
add=tf.add(input1,input2)
mul=tf.multiply(add,input3)
with tf.Session() as sess:
    result=sess.run([mul,add])
    print (result);
