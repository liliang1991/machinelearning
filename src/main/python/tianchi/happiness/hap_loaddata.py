import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import csv
if __name__ == '__main__':
    data = pd.read_csv('./data/happiness_train_abbr.csv')
    data = data.loc[data['happiness'] != -8]
    data = data.fillna(-1)
    data["survey_month"] = data['survey_time'].str.split(' ').str[0].str.split('/').str[1].astype('int')
    data["survey_day"]=data['survey_time'].str.split(' ').str[0].str.split('/').str[2].astype('int')
    data["survey_hour"]=data['survey_time'].str.split(' ').str[1].str.split(':').str[0].astype('int')

# data['time']=data['survey_time'].str.split(' ').str[0].str.split('/').str[0]+str()+data['survey_time'].str.split(' ').str[0].str.split('/').str[1]
    # data['time']=data['time'].astype('float')



    data['location']=data['province'].map(str)+data['city'].map(str)+data['county'].map(str)
    data['location']=data['location'].astype('float')

    drop_cols = ['province','city','county','survey_time']
    data.drop(drop_cols, axis=1, inplace=True)


    #print(data['location'].head(10)).astype("int")
   # print(data['happiness','year'].value_counts())
    #print(data.groupby(['happiness','year','month']).size().reset_index(name='count'))
    fig, ax = plt.subplots(1,2,figsize=(25,5))
    print(data.columns)
    print(data.corr()['happiness'])
    #numerical_features = (data.corr()['happiness'][abs(data.corr()['happiness'])>0.05]).index
    numerical_features = [x for x in data.columns if  x!='happiness' and x!='id' ]
    print(len(numerical_features))
# 探究性别和幸福感的分布
    # sns.countplot('year',hue='happiness',data=data)
    # ax[0].set_title('Sex:happiness')
    # plt.show()
    ## 特征与标签组合的散点可视化
    # sns.pairplot(data=data,diag_kind='hist', hue= 'happiness')
    # plt.show()
    # for col in data.columns:
    #     sns.boxplot(x='happiness', y=col, saturation=0.5,palette='pastel', data=data)
    #     plt.title(col)
    #     plt.show()
    # sns.pairplot(data=data[['province',
    #                         'income',
    #                         'house'] + ['happiness']], diag_kind='hist', hue= 'happiness')
    # data = pd.melt(data, id_vars='happiness', var_name='Features', value_name='Values')
    #
    # sns.violinplot(x='Features', y='Values', hue='happiness',
    #                data=data, split=True, inner='quart', ax=ax[1], palette='happiness')
    # plt.show()num_class
    #numerical_features = numerical_features.values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(data[numerical_features],data['happiness'], test_size = 0.2, random_state = 12)
    ## 定义 XGBoost模型
    model = XGBClassifier(eta=0.1,min_child_weight=2,n_estimators=20,colsample_bytree = 0.6, learning_rate = 0.08, max_depth= 16, subsample = 0.8,objective='multi:softmax',num_class=5)
    #model = XGBClassifier(n_estimators=20,colsample_bytree = 0.6, learning_rate = 0.051, max_depth= 16, subsample = 0.8,objective='reg:linear')
    # model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.1,
    #                      colsample_bytree=0.971, gamma=0.11, learning_rate=0.069, max_delta_step=0,
    #                      max_depth=3, min_child_weight=1, missing=None, n_estimators=499,
    #                      n_jobs=-1, nthread=50, objective='reg:linear', random_state=0,
    #                      reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, seed=None,
    #                      silent=True, subsample=1.0)
    # 在训练集上训练XGBoost模型

    # model = LGBMRegressor(n_jobs=-1,learning_rate=0.051,
    #                       n_estimators=400,
    #                       num_leaves=11,
    #                       reg_alpha=2.0,
    #                       reg_lambda=2.1,
    #                       min_child_samples=6,
    #                       min_split_gain=0.5,
    #                       colsample_bytree=0.2
    #                       )
    model.fit(x_train, y_train)


    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    print(numerical_features)
    print(numerical_features)
    sns.barplot(y=data[numerical_features].columns, x=model.feature_importances_)
    plt.show()
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
    ## 定义参数取值范围
    # learning_rate = [0.1, 0.3, 0.6]
    # subsample = [0.8, 0.9]
    # colsample_bytree = [0.6, 0.8]
    # max_depth = [3,5,8]
    #
    # parameters = { 'learning_rate': learning_rate,
    #            'subsample': subsample,
    #            'colsample_bytree':colsample_bytree,
    #            'max_depth': max_depth}
    # model = XGBClassifier(n_estimators = 10)
    #
    # ## 进行网格搜索
    # clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=1,n_jobs=-1)
    # clf = clf.fit(x_train, y_train)
    # print(clf.best_params_)
    data_test=pd.read_csv('./data/happiness_test_abbr.csv')
    data_test["survey_month"] = data_test['survey_time'].str.split(' ').str[0].str.split('/').str[1].astype('int')
    data_test["survey_day"]=data_test['survey_time'].str.split(' ').str[0].str.split('/').str[2].astype('int')
    data_test["survey_hour"]=data_test['survey_time'].str.split(' ').str[1].str.split(':').str[0].astype('int')

    data_test['location']=data_test['province'].map(str)+data_test['city'].map(str)+data_test['county'].map(str)
    data_test['location']=data_test['location'].astype('int')
    data_test.drop(drop_cols, axis=1, inplace=True)

    print(numerical_features)
    print(data_test['id'].dtype)
    test_predict = model.predict(data_test[numerical_features])

    f = open('./data/res.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id','happiness'])

    for i in range(len(test_predict)):
        #f.write(str(data_test.iloc[i]['id'])+'\t'+str(test_predict[i])+'\n')
        csv_writer.writerow([data_test.iloc[i]['id'].astype(int),test_predict[i]])







