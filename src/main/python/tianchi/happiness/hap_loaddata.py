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
if __name__ == '__main__':

    data = pd.read_csv('./data/happiness_train_abbr.csv')
    data_test = pd.read_csv('./data/happiness_test_abbr.csv')

    # data = data.loc[data['happiness'] != -8]
    data['happiness'] = data['happiness'].replace(-8, 3)
    pd.set_option('display.max_columns', None)
    # print(data.isnull().sum(axis=0))

    # data = data.fillna(-2)
    data["survey_year"] = data['survey_time'].str.split(' ').str[0].str.split('/').str[0].astype('int')

    data["survey_month"] = data['survey_time'].str.split(' ').str[0].str.split('/').str[1].astype('int')
    data["survey_day"] = data['survey_time'].str.split(' ').str[0].str.split('/').str[2].astype('int')
    data["survey_hour"] = data['survey_time'].str.split(' ').str[1].str.split(':').str[0].astype('int')

    # data['time']=data['survey_time'].str.split(' ').str[0].str.split('/').str[0]+str()+data['survey_time'].str.split(' ').str[0].str.split('/').str[1]
    # data['time']=data['time'].astype('float')

    data['location'] = data['province'].map(str) + data['city'].map(str) + data['county'].map(str)
    data['location'] = data['location'].astype('float')

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

    data['family_income'].fillna(data['family_income'].mean(), inplace=True)
    data.loc[data['nationality'] == -8,'nationality'] = 1
    data.loc[data['religion'] == -8,'religion'] = 1
    data.loc[data['religion_freq'] == -8,'religion_freq'] = 1
    #data['nationality'] = data['nationality'].apply(lambda x: nationality_handle(x))
    #data.loc[data.product_inner_type == -8 , 'nationality' ] = 1

    drop_cols = ['province', 'city', 'county', 'survey_time', 'birth']
    data.drop(drop_cols, axis=1, inplace=True)
    #print(data['family_income'].head(10).T)

    data['family_income'] = data['family_income'].apply(lambda x: fun(x))
    for (columnName, columnData) in data.iteritems():
        data.loc[data[columnName] < 0,columnName] = pd.Series(data[columnName]).value_counts().index[0]
        #print(pd.Series(data[columnName]).value_counts())
        print('Colunm Name : ' + str(columnName) + "\t" + str(pd.Series(data[columnName]).value_counts()))
    #print("family_income===" + str(data['family_income'].mean()))



    # print(data['family_income'][data['family_income'] < 0].mean())


    # print(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    # print(data.isnull().sum(axis=0))
    # print(data['location'].head(10)).astype("int")
    # print(data['happiness','year'].value_counts())
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
    param = {'booster': 'gbtree', 'max_depth': 2, 'eta': 0.1, 'objective': 'reg:squarederror', 'learning_rate': 0.3,
             'colsample_bytree': 0.6, "subsample": 0.9}
    model = xgb.train(param, dtrain, num_boost_round=50)
    pred_test = model.predict(dtest)
    pred_train = model.predict(dtrain)
    for i in range(len(pred_test)):
        pred_test[i] = pred_test[i]
        # print(pred_test[i])
    print(mean_squared_error(dtrain.get_label(), pred_train))
    print(mean_squared_error(dtest.get_label(), pred_test))
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

    data_test['location'] = data_test['province'].map(str) + data_test['city'].map(str) + data_test['county'].map(str)
    data_test['location'] = data_test['location'].astype('int')
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
