#使用keras内置的数据集
import tensorflow.keras as keras
####自定义网络层(需继承自 Layer 基类)
from tensorflow.keras import layers
#导入tensorflow库
import tensorflow as tf
if __name__ == '__main__':
   # 加载 ImageNet 预训练网络模型，并去掉最后一层
   resnet = keras.applications.ResNet50(weights='imagenet',include_top=False)
   resnet.summary()
   x = tf.random.normal([4,224,224,3])
   out = resnet(x)
   out.shape
