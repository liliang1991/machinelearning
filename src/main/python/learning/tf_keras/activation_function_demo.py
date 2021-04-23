import tensorflow as tf
from tensorflow.keras import activations
import matplotlib.pyplot as plt
##激活函数deno

if __name__ == '__main__':
    x = tf.linspace(-5., 5., 100)   # 构造一段连续的数据
    x_ndarray = x.numpy()   # 转换为 ndarray 的类型
    ###sigmoid函数也叫Logistic函数,可以用来做二分类。在特征相差比较复杂或是相差不是特别大时效果比较好
    y_relu = activations.sigmoid(x)    # 使用 Relu 函数运算
    plt.plot(x, y_relu, c='red', label='sigmoid')    # 画折线图
    plt.ylim((-1, 1))  #y的范围
    plt.legend(loc='best')
    plt.show()


    ##Tanh是可以通过sigmoid平移等操作变化而来。但它的收敛速度要比sigmoid收敛的要快。其他的优缺点和sigmoid函数类似。它就是为了克服Sigmoid的不对原点对称的坏毛病。可惜的是它在两边还是有梯度饱和（也就是梯度趋近于0）的问题。
    x = tf.linspace(-5., 5., 100)   # 构造一段连续的数据
    x_ndarray = x.numpy()   # 转换为 ndarray 的类型
    y = activations.tanh(x)
    plt.plot(x, y, c='red', label='tanh')    # 画折线图
    plt.ylim((-1.2, 1.2))
    plt.legend(loc='best')
    plt.show()



    ###relu 比 sigmoid 和 tanh 快；（梯度不会饱和，解决了梯度消失问题
    ##缺点L:训练的时候很”脆弱”，因为当取负号的时候 ，它的导数为零，预示着后半段就没什么作用了。
    x = tf.linspace(-5., 5., 100)	# 构造一段连续的数据
    x_ndarray = x.numpy()	# 转换为 ndarray 的类型
    y_relu = activations.relu(x)

    plt.plot(x, y_relu, c='red', label='relu')    # 画折线图
    plt.ylim((-0.5, 1.2))
    plt.legend(loc='best')
    plt.show()

    ##elu为解决ReLU存在的问题而提出。Elu激活函数有优点：ReLU的基本所有优点、不会有Dead ReLU问题，输出的均值接近0、零中心点问题。Elu激活函数有缺点：计算量稍大，原点不可导。
    x = tf.linspace(-5., 5., 100)	# 构造一段连续的数据
    x_ndarray = x.numpy()	# 转换为 ndarray 的类型
    y_relu = activations.elu(x)

    plt.plot(x, y_relu, c='red', label='elu')    # 画折线图
    plt.ylim((-2, 5))
    plt.legend(loc='best')
    plt.show()


    ###selu
    ###其实就是ELU乘了个lambda，关键在于这个lambda是大于1的。以前relu，prelu，elu这些激活函数，都是在负半轴坡度平缓，这样在activation的方差过大的时候可以让它减小，防止了梯度爆炸，但是正半轴坡度简单的设成了1。而selu的正半轴大于1，在方差过小的的时候可以让它增大，同时防止了梯度消失。这样激活函数就有一个不动点，网络深了以后每一层的输出都是均值为0方差为1。
    x = tf.linspace(-5., 5., 100)	# 构造一段连续的数据
    x_ndarray = x.numpy()	# 转换为 ndarray 的类型
    y_relu = activations.selu(x)

    plt.plot(x, y_relu, c='red', label='selu')    # 画折线图
    plt.ylim((-2, 5))
    plt.legend(loc='best')
    plt.show()

    ###gelu 一般常在bert中使用的激活函数，作者经过实验证明比relu等要好。原点可导，不会有Dead ReLU问题。
    # x = tf.linspace(-5., 5., 100)	# 构造一段连续的数据
    # x_ndarray = x.numpy()	# 转换为 ndarray 的类型
    # y_relu = activations.gelu(x)
    # plt.plot(x, y_relu, c='red', label='gelu')    # 画折线图
    # plt.ylim((-2, 5))
    # plt.legend(loc='best')
    # plt.show()


    ###swish函数

    ##swish函数可以叫作自门控激活函数，它近期由谷歌的研究者发布
    x = tf.linspace(-5., 5., 100)	# 构造一段连续的数据
    x_ndarray = x.numpy()	# 转换为 ndarray 的类型
    y_relu = activations.swish(x)
    plt.plot(x, y_relu, c='red', label='swish')    # 画折线图
    plt.ylim((-2, 5))
    plt.legend(loc='best')
    plt.show()