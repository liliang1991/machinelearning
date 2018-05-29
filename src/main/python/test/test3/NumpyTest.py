#coding=utf-8
import  numpy as np
import matplotlib.pyplot as plt
xdata=np.linspace(1,3,5)[:,np.newaxis]
noise=np.random.normal(1,1,xdata.shape)
print  xdata
#plt.figure()
#plt.scatter(xdata,None)
#关键句,前两个参数是X、Y轴数据,其他参数指定曲线属性，如标签label，颜色color,线宽linewidth或lw,点标记marker
#np.square xdata的平方
plt.plot(noise,noise,'r-',lw=5)
plt.show()

