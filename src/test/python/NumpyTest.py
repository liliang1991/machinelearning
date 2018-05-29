#coding=utf-8
#numpy测试
import  numpy as np
import matplotlib.pyplot as plt
xdata=np.linspace(1.0,3.0,5)[:,np.newaxis]
noise=np.random.normal(0,0.02,xdata.shape)
plt.figure()
plt.scatter(xdata,None)
plt.plot(xdata,None,'r-',lw=5)
plt.show()
print(noise)