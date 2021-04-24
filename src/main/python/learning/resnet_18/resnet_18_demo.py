###18 层的深度残差网络 ResNet18
from tensorflow.keras import Sequential,layers
import tensorflow as tf
from tensorflow import keras
#构建残差模块
class BasicBlock(layers.Layer):

    def __init__(self,filter_num, stride=1):
        super( BasicBlock,self).__init__()
        #第一个卷积
        self.conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        #第二个卷积
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample=lambda  x:x
        def call(self, inputs, training=None):
            #d第一个卷积
            out = self.conv1(inputs)
            out = self.bn1(out)
            out = self.relu(out)
            #第二个卷积
            out = self.conv2(out)
            out = self.bn2(out)
            #identity模块
            identity = self.downsample(inputs)
            #两个输出直接相加
            output = layers.add([out, identity])
            output = tf.nn.relu(output) # 激活函数
            return  output
###1次完成多个残差模块的新建
def build_resblock(self,filter_num, blocks, stride=1):
    res_blocks = Sequential()
    #只有第一个BasicBlock 的步长可能不为 1，实现下采样
    res_blocks.add(BasicBlock(filter_num, stride))

    for _ in range(1, blocks):#其他 BasicBlock 步长都为 1
        res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks

#ResNet实现类
class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=10): # [2, 2, 2, 2]
        super(ResNet, self).__init__()
        # 根网络，预处理
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 堆叠 4 个 Block，每个 block 包含了多个 BasicBlock,设置步长不一样
        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # 通过 Pooling 层将高宽降低为 1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(num_classes)

    def build_resblock(self,filter_num, blocks, stride=1):
        res_blocks = Sequential()
        #只有第一个BasicBlock 的步长可能不为 1，实现下采样
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):#其他 BasicBlock 步长都为 1
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks

    def call(self, inputs, training=None):
        # 通过根网络
        x = self.stem(inputs)
        # 一次通过 4 个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 通过池化层
        x = self.avgpool(x)
        # 通过全连接层
        x = self.fc(x)

        return x

#搭建ResNet18
def resnet18():
    # 通过调整模块内部 BasicBlock 的数量和配置实现不同的 ResNet
    return ResNet([2, 2, 2, 2])
if __name__ == '__main__':
    model=resnet18()
    inputs = tf.random.normal([4,32,32,3])
    print(model(inputs))