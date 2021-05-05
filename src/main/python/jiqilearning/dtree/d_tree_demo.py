##决策树demo
##  基础函数库
import numpy as np

## 导入画图库
import matplotlib.pyplot as plt
import seaborn as sns

## 导入决策树模型函数
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

if __name__ == '__main__':
    ##Demo演示LogisticRegression分类

    ## 构造数据集
    x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
    y_label = np.array([0, 1, 0, 1, 0, 1])

    ## 调用决策树回归模型
    tree_clf = DecisionTreeClassifier()

    ## 调用决策树模型拟合构造的数据集
    tree_clf = tree_clf.fit(x_fearures, y_label)
    ## 可视化构造的数据样本点
    plt.figure()
    plt.scatter(x_fearures[:, 0], x_fearures[:, 1], c=y_label, s=50, cmap='viridis')
    plt.title('Dataset')
    plt.show()
