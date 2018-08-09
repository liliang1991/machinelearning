import numpy as np
def testArray():


    a = np.array([1,2,3])
    print(a.shape)
    a = np.array([[1,2,3],[4,5,6]])
    #一共有多少个，每组有多少个

    print(a.shape)
    a = np.array([[],[1,2,3],[]])
    print(a.shape)
    a = np.array([[1,2,3],[4,5,6]])
    a.shape =  (3,2)
    print(a)

# 多于一个维度
    a = np.array([[1,  2],  [3,  4]])
    print(a)
    a = np.array([1,  2,  3,4,5], ndmin =  3)
    print(a)
    a = np.array([1,  2,  3], dtype = complex)
    print(a)
    dt = np.dtype(np.int32)
    print(dt)

if __name__ == '__main__':
    testArray()