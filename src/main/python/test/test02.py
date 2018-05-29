#coding=utf-8
#变量
import tensorflow as tf;
a=tf.Variable([1,2]);
b=tf.constant([2,2])
#减法op
sub=tf.subtract(a,b);
init=tf.global_variables_initializer();
with tf.Session() as sess:
    sess.run(init)
    print sess.run(sub)