####决策树概率 信息熵 基尼系数值 错误率求导
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False

# 生成概率的范围
p = np.linspace(0.01, 0.99, 100)

# 计算基尼系数值。
def gini(p):
    return 1 - p ** 2 - (1 - p) ** 2

# 计算信息熵。
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)

# 计算错误率
def error(p):
    return 1 - np.max([p, 1 - p], axis=0)

x = np.linspace(0.01, 0.99, 200)
# 计算信息熵
en = entropy(x)
# 将信息熵进行缩放。因为信息熵的取值范围为[0, 1],而基尼系数与错误率的取值范围为[0, 0.5]，为了能够统一区间，
# 所以进行缩放，以便于可视化观察效果。
en2 = en * 0.5
# 计算错误率。
err = error(x)
# 计算基尼系数
g = gini(x)
fig = plt.figure()
for i, lab, ls, c, in zip([en, en2, g, err], ["信息熵", "信息熵（缩放）", "基尼系数", "错误率"],
                          ["-", ":", "--", "-."], ["r", "g", "b", "y"]):
    # 分别绘制信息熵，基尼系数与错误率的曲线，随着概率的变化而发生改变。
    plt.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
               ncol=4, fancybox=True, shadow=False)
    plt.axhline(y=0.5, linewidth=1, color='k', linestyle="--")
    plt.axhline(y=1.0, linewidth=1, color='k', linestyle="--")
    plt.ylim([0, 1.1])
    plt.xlabel("p(i=1)")
    plt.ylabel("纯度系数")
plt.show()
