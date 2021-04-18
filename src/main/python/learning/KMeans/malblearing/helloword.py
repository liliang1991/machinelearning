import matplotlib.pyplot as plt
import numpy as np
##张量学习
import tensorflow as  tf
if __name__ == '__main__':
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