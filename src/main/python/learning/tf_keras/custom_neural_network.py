

#使用keras内置的数据集
import tensorflow.keras as keras
####自定义网络层(需继承自 Layer 基类)
from tensorflow.keras import layers
#导入tensorflow库
import tensorflow as tf
class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
         super(MyDense,self).__init__()
         self.kernel=self.add_variable('w', [inp_dim, outp_dim],trainable=True)
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)
    #实现自定义网络的前向运算逻辑:
    def call(self,inputs,training=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
if __name__ == '__main__':

      net=MyDense(4,3)
      #print(net.variables)

      network=tf.keras.models.Sequential([MyDense(784, 256), # 使用自定义的层 MyDense(256, 128),
                                         MyDense(128, 64),
                                         MyDense(64, 32),
                                         MyDense(32, 10)])
      network.build(input_shape=(None,28*28))
      print(network.summary())