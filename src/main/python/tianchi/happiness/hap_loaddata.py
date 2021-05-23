import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, RepeatedKFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import csv
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
from xgboost import plot_importance
import graphviz
#import os
from xgboost import XGBRegressor as XGBR
from sklearn.linear_model import LinearRegression
#os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
def  nationality_handle(x):
    if x == -8:
        return 1
    else:
        return x
def fun(x):
    if x <= 0:
        return data['family_income'].mean()
    else:
        return x
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
def estimate(model,data):

    #sns.barplot(data.columns,model.feature_importances_)
    ax1=plot_importance(model,importance_type="gain")
    ax1.set_title('gain')
    ax2=plot_importance(model, importance_type="weight")
    ax2.set_title('weight')
    ax3 = plot_importance(model, importance_type="cover")
    ax3.set_title('cover')
    plt.show()
def classes(data,label,test):
    model=XGBClassifier()
    model.fit(data,label)
    ans=model.predict(test)
    estimate(model, data)
    return ans

def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    """绘制学习曲线：只需要传入算法(或实例对象)、X_train、X_test、y_train、y_test"""
    """当使用该函数时传入算法，该算法的变量要进行实例化，如：PolynomialRegression(degree=2)，变量 degree 要进行实例化"""
    train_score = []
    test_score = []
    for i in range(1, len(X_train)+1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train)+1)],
             train_score, label="train")
    plt.plot([i for i in range(1, len(X_train)+1)],
             test_score, label="test")

    plt.legend()
    plt.axis([0, len(X_train)+1, 0, 1])
    plt.show()


# def plot_learning_curve(estimator,title,X,y,ax = None,#选择子图
#                         ylim = None,#设置纵坐标取值范围
#                         cv = None,#交叉验证
#                         n_jobs = None#设定所要使用的线程
#                         ):
#     from sklearn.model_selection import learning_curve
#     train_sizes,train_scores,test_scores = learning_curve(estimator,X,y,shuffle = True,cv = cv,random_state=2020,n_jobs = n_jobs)
#     if ax == None:
#         ax = plt.gca()
#     else:
#         ax = plt.figure()
#     ax.set_title(title)
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     ax.set_xlabel('Training examples')
#     ax.set_ylabel('Score')
#     ax.grid()#绘制表格，不是必须
#     ax.plot(train_sizes,np.mean(train_scores,axis = 1),'o-',color='r',label='Training score')
#     ax.plot(train_sizes,np.mean(test_scores,axis = 1),'o-',color='g',label='Test score')
#     ax.legend(loc = 'best')
#     return ax

if __name__ == '__main__':

    data = pd.read_csv('./data/happiness_train_abbr.csv')
    data_test = pd.read_csv('./data/happiness_test_abbr.csv')

    # data = data.loc[data['happiness'] != -8]
    data['happiness'] = data['happiness'].replace(-8, 3)
    pd.set_option('display.max_columns', None)
    #print(data.isnull().sum(axis=0))


    # data = data.fillna(-2)
    data["survey_year"] = data['survey_time'].str.split(' ').str[0].str.split('/').str[0].astype('int')

    data["survey_month"] = data['survey_time'].str.split(' ').str[0].str.split('/').str[1].astype('int')
    data["survey_day"] = data['survey_time'].str.split(' ').str[0].str.split('/').str[2].astype('int')
    data["survey_hour"] = data['survey_time'].str.split(' ').str[1].str.split(':').str[0].astype('int')

    # data['time']=data['survey_time'].str.split(' ').str[0].str.split('/').str[0]+str()+data['survey_time'].str.split(' ').str[0].str.split('/').str[1]
    # data['time']=data['time'].astype('float')

    # data['location'] = data['province'].map(str) + data['city'].map(str) + data['county'].map(str)
    # data['location'] = data['location'].astype('float')

    ##删除确实值多的列
    data['work_type'].fillna(0, inplace=True)
    data['work_status'].fillna(9, inplace=True)
    data['work_manage'].fillna(2, inplace=True),
    data['work_yr'].fillna(0, inplace=True)

    # drop_cols = ['work_status','work_yr','work_type','work_manage']
    # data.drop(drop_cols, axis=1, inplace=True)

    # print(data.head(1).T)
    data["survey_age"] = data["survey_year"] - data["birth"]
    # print(data["survey_age"])
    data.sort_values("survey_age", inplace=True)
    res = data.sort_values(by='survey_age', ascending=False)
    # print(res)
    #data['nationality'] = data['nationality'].apply(lambda x: nationality_handle(x))
    #data.loc[data.product_inner_type == -8 , 'nationality' ] = 1

    #drop_cols = ['province', 'city', 'county', 'survey_time', 'birth']
    ##'floor_area','height_cm','weight_jin'
    drop_cols = [ 'survey_time', 'birth']

    data.drop(drop_cols, axis=1, inplace=True)
    #print(data['family_income'].head(10).T)
    print("========")
    #print(pd.Series(data['work_manage']).value_counts())
    for (columnName, columnData) in data.iteritems():
        if(columnName!='family_income'):
            #data[columnName].fillna(pd.Series(data[columnName]).value_counts().index[0], inplace=True)
            data.loc[data[columnName] < 0,columnName] = pd.Series(data[columnName]).value_counts().index[0]
        #print('Colunm Name : ' + str(columnName) + "\t" + str(pd.Series(data[columnName]).value_counts()))
    #print("family_income===" + str(data['family_income'].mean()))
    data['family_income'].fillna(data['family_income'].mean(), inplace=True)
    data['family_income'] = data['family_income'].apply(lambda x: fun(x))



    # print(data['family_income'][data['family_income'] < 0].mean())


    # print(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    # print(data.isnull().sum(axis=0))
    # print(data['location'].head(10)).astype("int")
    print(data['happiness'].value_counts())
    # print(data.groupby(['happiness','year','month']).size().reset_index(name='count'))
    # fig, ax = plt.subplots(1,2,figsize=(25,5))
    # numerical_features = (data.corr()['happiness'][abs(data.corr()['happiness'])>0.05]).index
    numerical_features = [x for x in data.columns if x != 'happiness' and x != 'id']

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
    # numerical_features = numerical_features.values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(data[numerical_features], data['happiness'], test_size=0.2,
                                                        random_state=12)
    ## 定义 XGBoost模型
    # model = XGBClassifier(eta=0.1,min_child_weight=2,n_estimators=2000,colsample_bytree = 0.6, learning_rate = 0.08, max_depth= 16, subsample = 0.8,objective='multi:softmax',num_class=5)
    # xgb_params = {"booster":'gbtree','eta': 0.005, 'max_depth': 5, 'subsample': 0.7,
    #               'colsample_bytree': 0.8, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}
    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test, y_test)
    #https://blog.51cto.com/u_15127527/2688437
    param = {'booster': 'gbtree', 'max_depth': 4, 'eta': 0.1, 'objective': 'reg:squarederror', 'learning_rate': 0.05,
             'colsample_bytree': 0.8, "subsample": 0.7,"verbosity":1,"max_delta_step":4,"tree_method":"exact","eval_metric":"mae"}
    model = xgb.train(param, dtrain, num_boost_round=200)
    pred_test = model.predict(dtest)
    pred_train = model.predict(dtrain)
    # for i in range(len(pred_test)):
    #     pred_test[i] = pred_test[i]
        # print(pred_test[i])
    print(mean_squared_error(dtrain.get_label(), pred_train))
    print(mean_squared_error(dtest.get_label(), pred_test))
    xgb.plot_importance(model)
    # xgb.plot_importance(model)
    # plt.rcParams['figure.figsize'] = [3, 3]
    plt.show()
    #cv = KFold(n_splits=5, shuffle = True, random_state=42)
    #plot_learning_curve(XGBR(n_estimators = 100,random_state = 2020),'XGB',x_train,y_train,ax = None,cv = cv)
    #plt.show()
    #plot_learning_curve(LinearRegression(),x_train, x_test, y_train, y_test)
    ######tree
    # ceate_feature_map(numerical_features)
    # plot_tree(model)
    # fig = plt.gcf()
    # fig.set_size_inches(150, 100)
    # fig.savefig('tree.png')


    #sns.barplot(y=data[numerical_features].columns, x=model.feature_importances_)



    # ans=classes(x_train,y_train,x_test)
    # pre=accuracy_score(y_test, ans)
    # ##模型训练的准确率百分比
    # print('acc=',accuracy_score(y_test,ans))
    #plt.show()

## 定义参数取值范围
    # learning_rate = [0.1, 0.3, 0.6]
    # subsample = [0.8, 0.9]
    # colsample_bytree = [0.6, 0.8]
    # max_depth = [3,5,8]
    # parameters = { 'learning_rate': learning_rate,
    #                'subsample': subsample,
    #                'colsample_bytree':colsample_bytree,
    #                'max_depth': max_depth}
    # model = XGBClassifier(n_estimators = 50)
    #
    # ## 进行网格搜索
    # clf = GridSearchCV(model, parameters, cv=3, scoring='neg_mean_squared_error',verbose=1,n_jobs=-1)
    # clf = clf.fit(x_train, y_train)
    # print(clf.best_params_)
    # #X_train_ = data[].X_train_ = data[:train.shape[0]]
    # for fold_, (trn_idx, val_idx) in enumerate(folds.split(data[numerical_features].shape[0], data['happiness'].shape[0])):
    #     #print("fold n°{}".format(fold_+1))
    #
    #     print(trn_idx[0])
    #     print("bbb"+str(trn_idx[1]))
    #     print(val_idx)
    #     trn_data = xgb.DMatrix(x_train[trn_idx], y_train[trn_idx])
    #     val_data = xgb.DMatrix(x_train[val_idx], y_train[val_idx])
    #     #model = XGBClassifier(max_depth=16, learning_rate=0.08, colsample_bytree=0.7, subsample=0.7,objective='reg:linear',eval_metric='rmse')
    #     watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    #     model = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
    #     oof_xgb[val_idx] = model.predict(xgb.DMatrix(x_train[val_idx]), ntree_limit=model.best_ntree_limit)
    #     predictions_xgb += model.predict(xgb.DMatrix(x_test), ntree_limit=model.best_ntree_limit) / folds.n_splits
    #
    #     print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))

    data_test["survey_month"] = data_test['survey_time'].str.split(' ').str[0].str.split('/').str[1].astype('int')
    data_test["survey_day"] = data_test['survey_time'].str.split(' ').str[0].str.split('/').str[2].astype('int')
    data_test["survey_hour"] = data_test['survey_time'].str.split(' ').str[1].str.split(':').str[0].astype('int')
    data_test["survey_year"] = data_test['survey_time'].str.split(' ').str[0].str.split('/').str[0].astype('int')

    # data_test['location'] = data_test['province'].map(str) + data_test['city'].map(str) + data_test['county'].map(str)
    # data_test['location'] = data_test['location'].astype('int')
    data_test["survey_age"] = data_test["survey_year"] - data_test["birth"]

    data_test.drop(drop_cols, axis=1, inplace=True)
    predictions = model.predict(xgb.DMatrix(data_test[numerical_features]))
    result = np.c_[data_test.id, predictions]
    np.savetxt('./data/res.csv', result, fmt="%d,%f", header='id,happiness', delimiter=',', comments='')

    # test_predict = model.predict(data_test[numerical_features])
    # xgtest = xgb.DMatrix(data_test[numerical_features])
    # test_predict = model.predict(xgtest,ntree_limit=model.best_iteration)
    # result=list(predictions)
    # result=list(map(lambda x: x + 1, result))
    # test_sub["happiness"]=result
    # test_sub.to_csv("submit_20190515.csv", index=False)

    # f = open('./data/res.csv','w',encoding='utf-8')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(['id','happiness'])
    #
    # for i in range(len(test_predict)):
    #     #f.write(str(data_test.iloc[i]['id'])+'\t'+str(test_predict[i])+'\n')
    #     csv_writer.writerow([data_test.iloc[i]['id'].astype(int),test_predict[i]])
