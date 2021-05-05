##决策树企鹅数据demo
##  基础函数库
import numpy as np
import pandas as pd

## 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns
def trans(x):
    if x == data['Species'].unique()[0]:
        return 0
    if x == data['Species'].unique()[1]:
        return 1
    if x == data['Species'].unique()[2]:
        return 2
if __name__ == '__main__':
    data = pd.read_csv('./penguins_raw.csv')
    data = data[['Species','Culmen Length (mm)','Culmen Depth (mm)',
                 'Flipper Length (mm)','Body Mass (g)']]
    data.info()
    ##处理NaN
    data = data.fillna(-1)
    ##label列
    print(data['Species'].unique())
    ## 利用value_counts函数查看每个类别数量
    pd.Series(data['Species']).value_counts()
    ## 特征与标签组合的散点可视化
    sns.pairplot(data=data, diag_kind='hist', hue= 'Species')
    #plt.show()


    data['Species'] = data['Species'].apply(trans)



    ## 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
    from sklearn.model_selection import train_test_split

    ## 选择其类别为0和1的样本 （不包括类别为2的样本）
    data_target_part = data[data['Species'].isin([0,1])][['Species']]
    data_features_part = data[data['Species'].isin([0,1])][['Culmen Length (mm)','Culmen Depth (mm)',
                                                        'Flipper Length (mm)','Body Mass (g)']]
    print(data_features_part)
    print(data_target_part)
    ## 测试集大小为20%， 80%/20%分
    x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.2, random_state = 2020)

    ## 从sklearn中导入决策树模型
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    ## 定义 决策树模型
    clf = DecisionTreeClassifier(criterion='entropy')
    # 在训练集上训练决策树模型
    print("==")
    print(clf.fit(x_train, y_train))

    ## 可视化
    import graphviz
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("penguins")




    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)
    from sklearn import metrics

    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
    print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

    ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
    print('The confusion matrix result:\n',confusion_matrix_result)

    # 利用热力图对于结果进行可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()



    ##利用 决策树模型 在三分类(多分类)上 进行训练和预测
    ## 测试集大小为20%， 80%/20%分
    x_train, x_test, y_train, y_test = train_test_split(data[['Culmen Length (mm)','Culmen Depth (mm)',
                                                          'Flipper Length (mm)','Body Mass (g)']], data[['Species']], test_size = 0.2, random_state = 2020)
    ## 定义 决策树模型
    clf = DecisionTreeClassifier()
    # 在训练集上训练决策树模型
    clf.fit(x_train, y_train)



    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)

    ## 由于决策树模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所有我们可以利用 predict_proba 函数预测其概率
    train_predict_proba = clf.predict_proba(x_train)
    test_predict_proba = clf.predict_proba(x_test)

    print('The test predict Probability of each class:\n',test_predict_proba)
    ## 其中第一列代表预测为0类的概率，第二列代表预测为1类的概率，第三列代表预测为2类的概率。

    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
    print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

    ## 查看混淆矩阵
    confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
    print('The confusion matrix result:\n',confusion_matrix_result)

    # 利用热力图对于结果进行可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()