## 使用keras训练MNIST数据集
##kears 相关api 说明 https://tensorflow.google.cn/guide/keras
import tensorflow as tf #导入tensorflow库
import tensorflow.keras as keras  #使用keras内置的数据集
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() #加载MNIST数据集
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #网络设计
    ###Flatten即把多维的输入一维化，常用在从卷积层到全连接层的过渡
    ###activation 激活函数(没有办法画出一条直线来将数据区分开 这种问题使用激活函数)
    ###Dropout层用于防止过拟合,Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
])

    ##梯度下降（Gradient Descent）就好比一个人想从高山上奔跑到山谷最低点，用最快的方式（steepest）奔向最低的位置（minimum）。adam 是对梯度下降的一种优化
    ###适应性梯度算法（AdaGrad）为每一个参数保留一个学习率以提升在稀疏梯度（即自然语言和计算机视觉问题）上的性能。
    ###均方根传播（RMSProp）基于权重梯度最近量级的均值为每一个参数适应性地保留学习率。这意味着算法在非稳态和在线问题上有很有优秀的性能
    ###adam算法同时获得了 AdaGrad 和 RMSProp 算法的优点


    ##损失函数loss 介绍 https://cloud.tencent.com/developer/article/1390578?from=information.detail.%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E7%9A%84%E6%84%8F%E4%B9%89%E5%92%8C%E4%BD%9C%E7%94%A8
    ##损失函数作用是  衡量我们预测的公式与实际值好坏

    ##评价函数 和损失函数 相似，只不过评价函数的结果不会用于训练过程中
    model.compile(optimizer='adam', #选择adam优化器
                  loss='sparse_categorical_crossentropy',#选择sparse_categorical_crossentrop损失函数
                  metrics=['accuracy']) #选择sparse_categorical_accuracy评测指标

    a=model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
    print(a.history)
    test_scores = model.evaluate(x_test, y_test)
    print('test loss:', test_scores[0])
    print('test acc:', test_scores[1])
    model.save('model.h5')
    net = tf.keras.models.load_model('model.h5')
    net.summary()