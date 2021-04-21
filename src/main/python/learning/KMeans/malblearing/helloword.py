import matplotlib.pyplot as plt
import numpy as np
##张量学习,数据类型学习
import tensorflow as  tf
if __name__ == '__main__':

   print(tf.__version__)
   #0维
   scalar_tf=tf.constant(3.14)
   type(scalar_tf),tf.is_tensor(scalar_tf) #查看张量类型及是否是张量
   print(scalar_tf)
   #2维
   Max_tf=tf.constant([[3.14,0,1],[6.28,0,2]])
   type(Max_tf),tf.is_tensor(Max_tf)
   print(Max_tf)
   #3维
   Tensor_tf=tf.constant([[[3.14,0,1],[6.28,0,2]],[[1,2,3],[4,5,6]]])
   type(Tensor_tf),tf.is_tensor(Tensor_tf)
   print(Tensor_tf)
   # 创建一个全为1元素的张量。
   c = tf.ones([2,3])
   print(c)
   # 改变张量中元素的数据类型，只能由低精度向高精度转化。
   d = tf.cast(c,float)
   print(d)
   # tf.file(dims,value)创建维度为dims，值全为value的张量
   e = tf.fill([2,3],3)
   print(e)
   # 创建一个标准正态分布,shape:形状，mean: 均值，默认是0,stddev: 标准差，默认是1,dtype: 数据类型默认是float32
   f = tf.random.normal(shape=(3,2),mean=2,stddev=1,dtype=tf.float32)
   print(f)
   # 创建一个截断正态分布,返回一个截断的正态分布，截断标准是标准差的二倍
   g = tf.random.truncated_normal(shape=(3,2),mean=2,stddev=1,dtype=tf.float32)
   print(g)
   # 创建均匀分布张量，minval为最小值，maxval为最大值
   h = tf.random.uniform(shape=(2,3),minval=1,maxval=4,dtype=tf.float32)
   print(h)
   # 创建start为起点，delta为变化量，不超过limit的等差张量。
   j = tf.range(start=1,limit=5,delta=1,dtype=tf.float32)
   print(j)
   #字符串创建
   String_tf=tf.constant('tensorflow string type')
   print(String_tf)
   #bool类型创建
   bool_tf = tf.constant(False)
   print(bool_tf)
   #查看数值精度
   data_tf=tf.constant(3.1415926)
   print(data_tf.dtype)
   #转换其数值精度
   data_tf=tf.constant(3.1415926)
   data_tf=tf.cast(data_tf,tf.float64)
   print(data_tf.dtype)