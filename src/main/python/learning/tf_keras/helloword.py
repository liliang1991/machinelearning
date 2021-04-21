import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
if __name__ == '__main__':
    ##Softmax 在机器学习和深度学习中有着非常广泛的应用。尤其在处理多分类（C > 2）问题，分类器最后的输出单元需要Softmax 函数进行数值处理。关于Softmax 函数的定义如下所示：
    x = tf.constant([3.,4.,0.1])
    layer = layers.Softmax(axis=-1) # 创建Softmax层
    print(layer(x)) # 调用softmax前向计算

    model = tf.keras.models.Sequential([ # 封装为一个网络
        tf.keras.layers.Dense(128, activation='relu'), # 全连接层
        tf.keras.layers.Dense(64, activation='relu'), # 全连接层
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    x = tf.random.normal([6,7])
    print(x)