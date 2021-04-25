## 基于鸢尾花（iris）数据集的逻辑回归分类

import matplotlib.pyplot as plt
import seaborn as sns

# 变量 	描述
# sepal length 	花萼长度(cm)
# sepal width 	花萼宽度(cm)
# petal length 	花瓣长度(cm)
# petal width 	花瓣宽度(cm)
# target 	鸢尾的三个亚属类别,'setosa'(0), 'versicolor'(1), 'virginica'(2)
##  基础函数库
import numpy as np
import pandas as pd
## 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
if __name__ == '__main__':
    data = load_iris() #得到数据特征
    iris_target = data.target #得到数据对应的标签
    iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) #利用Pandas转化为DataFrame格式
    ## 利用.info()查看数据的整体信息
    iris_features.info()
    print(iris_features.head())
    print(iris_target)
    print(iris_features.describe())

    ## 合并标签和特征信息
    iris_all = iris_features.copy() ##进行浅拷贝，防止对于原始数据的修改
    iris_all['target'] = iris_target
    ## 特征与标签组合的散点可视化
    sns.pairplot(data=iris_all,diag_kind='hist', hue= 'target')
    plt.show()