import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('./data/happiness_train_abbr.csv')
    data.info()
    data = data.fillna(-1)

    fig, ax = plt.subplots(1,2,figsize=(15,5))
    numerical_features = [x for x in data.columns if data[x].dtype == np.float]

    # sns.pairplot(data=data[['province',
    #                         'income',
    #                         'house'] + ['happiness']], diag_kind='hist', hue= 'happiness')
    # data = pd.melt(data, id_vars='happiness', var_name='Features', value_name='Values')
    #
    # sns.violinplot(x='Features', y='Values', hue='happiness',
    #                data=data, split=True, inner='quart', ax=ax[1], palette='happiness')
    # plt.show()
    x_train, x_test, y_train, y_test = train_test_split(data[['province','income',
                                                              'house','marital','status_3_before','health','depression']],data['happiness'], test_size = 0.2, random_state = 2020)
    ## 定义 XGBoost模型
    model = XGBClassifier(colsample_bytree = 0.6, learning_rate = 0.3, max_depth= 8, subsample = 0.9)

    # 在训练集上训练XGBoost模型
    model.fit(x_train, y_train)


    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    from sklearn import metrics

    # ## 从sklearn库中导入网格调参函数
    # from sklearn.model_selection import GridSearchCV
    #
    # ## 定义参数取值范围
    # learning_rate = [0.1, 0.3, 0.6]
    # subsample = [0.8, 0.9]
    # colsample_bytree = [0.6, 0.8]
    # max_depth = [3,5,8]
    #
    # parameters = { 'learning_rate': learning_rate,
    #                'subsample': subsample,
    #                'colsample_bytree':colsample_bytree,
    #                'max_depth': max_depth}
    # model = XGBClassifier(n_estimators = 50)

    ## 进行网格搜索
    # clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=1,n_jobs=-1)
    # clf = clf.fit(x_train, y_train)
    # print(clf)
    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
    print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

    ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    # confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
    # print('The confusion matrix result:\n',confusion_matrix_result)
    #
    # # 利用热力图对于结果进行可视化
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.show()

    data_test=pd.read_csv('./data/happiness_test_abbr.csv')
    data_test.info()

    test_predict = model.predict(data_test)
    print(test_predict)







