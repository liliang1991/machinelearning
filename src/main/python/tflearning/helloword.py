import tensorflow as tf
import matplotlib.pyplot as plt
if __name__ == '__main__':
   scalar_tf=tf.constant(3.14)#创建张量
   m=tf.add(scalar_tf,scalar_tf)#执行操作
   print(m)#输出操作结果