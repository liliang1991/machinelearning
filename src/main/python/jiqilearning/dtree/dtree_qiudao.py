##决策树求导
##https://blog.csdn.net/qq_42442369/article/details/86625591
import numpy as np

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
if __name__ == '__main__':

  mpl.rcParams["font.family"] = "SimHei"
  mpl.rcParams["axes.unicode_minus"] = False

  # 计算概率值，概率值由0到1逐渐增大。
  p = np.linspace(0.01, 0.99, 100)
  # 计算概率逐渐改变的时候，信息熵的变化情况。
  h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
  plt.plot(p, h)
  plt.xlabel("概率1取值")
  plt.ylabel("信息熵")
  plt.title("两个随机随机变量不同取值-信息熵对比")
  plt.show()



