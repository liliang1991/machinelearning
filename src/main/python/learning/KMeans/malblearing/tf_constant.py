# tf张量操作
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
    # 创建2维张量
    t = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    ##########索引

    # 取0维度上的第一组数据
    a = t[0]
    # 即x[0]中的第二组数据
    b = t[0][1]
    print(a)
    print('-' * 50)
    print(b)

    #########切片
    # [start : end : step]
    print(t[1, 0:4:2])

    #######维度变换
    ##########改变视图的tf.reshape
    t = tf.range(24)
    t1 = tf.reshape(t, [2, 4, 3])
    print(t)
    print('-' * 50)
    print(t1)

    ######插入新维度tf.expand_dims
    print("######插入新维度tf.expand_dims")
    t = [[1, 2, 3], [4, 5, 6]]  # shape [2,3]
    t1 = tf.expand_dims(t, axis=0)  # sahpe [1,2,3]
    t2 = tf.expand_dims(t, axis=1)  # shape [2,,1,3]
    t3 = tf.expand_dims(t, axis=2)  # shape [2,3,1]
    t4 = tf.expand_dims(t, axis=-1)  # shape [2,3,1]
    print(t1)
    print('-' * 50)
    print(t2)
    print('-' * 50)
    print(t3)
    print('-' * 50)
    print(t4)

    ####删除维度tf.squeeze axis参数为待删除维度的索引号，删除维度只能删除长度为1的维度
    print("####删除维度tf.squeeze")
    t = tf.range(6)  # [0 1 2 3 4 5]
    t1 = tf.reshape(t, [1, 2, 1, 3, 1])
    t2 = tf.squeeze(t1)
    print(t1)
    print('-' * 50)
    print(t2)

     ######交换维度tf.transpose(交换数据位置)
    print("############交换维度tf.transpose")
    t = tf.random.normal([2, 5, 5, 3])   #sahpe [2, 5, 5, 3]
    t1 = tf.transpose(t, perm = [0, 3, 1, 2])
    print(t1.shape)

    ########合并(https://blog.csdn.net/leviopku/article/details/82380118)
    ###说明
    #########对于[ [ ], [ ]]和[[ ], [ ]]，低维拼接等于拿掉最外面括号，高维拼接是拿掉里面的括号(保证其他维度不变)。注意：tf.concat()拼接的张量只会改变一个维度，其他维度是保存不变的。比如两个shape为[2,3]的矩阵拼接，要么通过axis=0变成[4,3]，要么通过axis=1变成[2,6]。改变的维度索引对应axis的值。
    print("########合并")
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[7, 8, 9], [10, 11, 12]])
    c1 = tf.concat([a, b], 0)
    c2 = tf.concat([a, b], 1)
    print(c1)   # 未产生新维度
    print('-'*50)
    print(c2)   # 未产生新维度


    ##########分割(按形状分割)

    print("##########分割tf.unstack")
    a = tf.constant([[3,2,4,5,6],[1,6,7,8,0]])
    b = tf.constant([[3,1],[2,6],[4,7],[5,8],[6,0]])
    c = tf.unstack(a,axis=0)
    d = tf.unstack(a,axis=1)
    print(a.shape)
    print(c)
    print('-'*50)
    print(d)

    print("##########分割tf.split")
    ##tf.split(tensor, num_or_size_splits, axis),tensor为待分割张量，num_or_size_splits控制分割后对应维度上元素的个数，axis为指定分割的维度索引号
    x = [[1,2,3],[4,5,6]]
    print(tf.split(x, 3, 1))


    #########范数
    print("#########范数")
    x=tf.ones([3,3])
    # L1 范数
    L1=tf.norm(x,ord=1)
    print("L1 norm is: {}".format(L1))
    # L2范数
    L2=tf.norm(x,ord=2)
    print("L2 norm is: {}".format(L2))
    # ∞范数
    L_inf=tf.norm(x,ord=np.inf)
    print("∞ norm is: {}".format(L_inf))



    ######最值、均值(当不指定 axis 参数时，tf.reduce_*函数会求解出全局元素的最大、最小、均值。)
    print("######最值、均值")
    #创建张量
    x = tf.random.normal([3,3])
    print(x)
    #统计最大值
    max=tf.reduce_max(x,axis=1)
    print("max is: {}".format(max))
    #统计最小值
    min=tf.reduce_min(x,axis=1)
    print("min is: {}".format(min))
    # 统计均值
    mean=tf.reduce_mean(x,axis=1)
    print("mean is: {}".format(mean))

    ########张量比较
    print("########张量比较")
    pre_out=tf.convert_to_tensor(np.array([[1,2,3,4,2,3,5,6,7,8,9,2,3,4,5,4]]))
    label=tf.convert_to_tensor(np.array([[1,7,3,4,2,3,4,6,7,3,9,2,7,4,2,4]]))
    # 预测值与真实值比较
    acc=tf.equal(pre_out,label)
    print("acc is: {}".format(acc))
    # 布尔型转 int 型
    acc = tf.cast(acc, dtype=tf.float32)
    # 统计 True 的个数
    correct = tf.reduce_sum(acc)
    print("correct is: {}".format(correct))


    #######填充(默认维度33)
    print("#######填充")
    a= tf.fill([3,3],2)
    # 第一维度，前面补一维度，后面补一维度；第二维度，前面补两维度，后面补两维度。
    print(tf.pad(a,[[1,1],[2,2]]))

    ######复制
    print("######复制")
    a = tf.constant([[1,2],[3,4]])
    # 在axis=0维度复制一次，在axis=1维度不复制。
    c=tf.tile(a, multiples=[2, 1])
    print(c)
    #######限幅
    print("#######限幅")
    x=tf.range(100)
    #下限幅 50
    x=tf.maximum(x,50)
    #上限幅80
    s=tf.minimum(x,80)
    print(x)
    print(s)

    # 上下限幅为
    x = tf.range(10)
    z=tf.minimum(tf.maximum(x,3),8)
    print(z)