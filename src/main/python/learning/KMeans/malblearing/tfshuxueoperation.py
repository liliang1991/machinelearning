#tf 数学计算相关操作
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
    ###########加减乘除
    print("###########加减乘除")
    ###加法：
    num=tf.math.add( 2, 3, name=None )
    print(num)
    ###减法：
    num=tf.math.subtract( 3, 2, name=None )
    print(num)
    ###乘法：
    num=tf.math.multiply( 2, 2, name=None )
    print(num)
    ###除法：
    num=tf.math.divide( 4, 2, name=None )
    print(num)


    #######指数平方运算
    print("#######指数平方运算")
    a = tf.constant([[1.,2.],[3.,4.]])
    # 指数为2
    b = tf.pow(a,2)
    # 开方
    c = tf.sqrt(b)
    # 自然指数运算
    d = tf.exp(a)
    # 对数运算,以自然常数e为底的对数
    e = tf.math.log(a)
    # 对各个元素求平方
    f = tf.square(a)
    print(b)
    print('-'*50)
    print(c)
    print('-'*50)
    print(d)
    print('-'*50)
    print(e)
    print('-'*50)
    print(f)



    ###########矩阵相乘(https://www.cnblogs.com/liuzhongchao/p/9303170.html)
    print("###########矩阵相乘")
    matrix1 = tf.constant([[1,2],[3,4]])
    matrix2 = tf.constant([[5,6],[7,8]])
    result = tf.matmul(matrix1,matrix2)
    print(matrix1)
    print(result)