import tensorflow as tf
#计算图测试类
@tf.function(autograph=True)
def myadd(a,b):
    for i in tf.range(3):
        tf.print(i)
    c = a+b
    print("tracing")
    print(c)
    return c
if __name__ == '__main__':
   # 动态计算图在每个算子处都进行构建，构建后立即执行

   # x = tf.constant("hello")
   # y = tf.constant("world")
   # z = tf.strings.join([x,y],separator=" ")
   #
   # tf.print(z)
   print(myadd("ab","cd"))
