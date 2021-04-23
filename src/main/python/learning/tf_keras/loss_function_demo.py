###损失函数demo
import tensorflow as tf

if __name__ == '__main__':
    o = tf.random.normal([2,10]) # 构造网络输出
    print(o)
    y_onehot = tf.constant([1,3]) # 构造真实值
    y_onehot = tf.one_hot(y_onehot, depth=10)
    print(y_onehot)


    ###均方差函数(mean_squared_error)越小越精准
    loss = tf.keras.losses.mse(y_onehot, o) # 计算均方差  平方差误差损失，用于回归，简写为 mse, 类实现形式为 MeanSquaredError 和 MSE
    print(loss)

    ###交叉熵函数
    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    print(loss)#可以自己试着计算，看结果 是否相等。