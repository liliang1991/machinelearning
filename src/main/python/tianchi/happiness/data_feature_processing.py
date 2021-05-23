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
if __name__ == '__main__':
    data = pd.read_csv('./data/happiness_train_abbr.csv', parse_dates=['survey_time'])
    data_test = pd.read_csv('./data/happiness_test_abbr.csv', parse_dates=['survey_time'])
    ##使用.dt.year将survey_time转换成year的时间'''
    data['survey_time'] = data['survey_time'].dt.year
    data_test['survey_time'] = data_test['survey_time'].dt.year
    print(data.isnull().sum())
    data = data.loc[data['happiness'] != -8]
    f,ax=plt.subplots(1,2,figsize=(18,8))
    data['happiness'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title('happiness')
    ax[0].set_ylabel('')
    data['happiness'].value_counts().plot.bar(ax=ax[1])
    ax[1].set_title('happiness')

    # 探究性别和幸福感的分布
    sns.countplot('gender',hue='happiness',data=data)
    ax[1].set_title('Sex:happiness')
    plt.show()
    # # 探究年龄和幸福感的关系
    # train['survey_time'] = train['survey_time'].dt.year
    # test['survey_time'] = test['survey_time'].dt.year
    data['Age'] = data['survey_time']-data['birth']
    # test['Age'] = test['survey_time']-test['birth']
    # del_list=['survey_time','birth']
    # figure,ax = plt.subplots(1,1)
    # train['Age'].plot.hist(ax=ax,color='blue')

    '''
    绘制以下数据的热度图，
    .corr()方法表示的是计算dataframe多个指标的相关系数矩阵，默认使用pearson计算方法
    train[]表示传入热度图的数据，
     annot（布尔类型），用于控制是否在个字中间标明数字，
    cmap表示控制热度的渐变色，
    linewidths表示每个单元格的线的宽度'''
    sns.heatmap(data[['happiness','Age','inc_ability','gender','status_peer','family_status','health','equity','class','work_exper','health_problem','family_m','house','depression','learn','relax','edu']].corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
    fig=plt.gcf()  #获取当前的图表和子图
    fig.set_size_inches(15,15)  #设置图像的密集度：设置图像的长和宽
    plt.show()