import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import os, sys, pickle
import math
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
dfon=pd.read_csv("/home/moon/work/tianchi/ccf/data/ccf_online_stage1_train.csv",keep_default_na=False)
dfoff=pd.read_csv("/home/moon/work/tianchi/ccf/data/ccf_offline_stage1_train.csv",keep_default_na=False)
dftest=pd.read_csv("/home/moon/work/tianchi/ccf/data/ccf_offline_stage1_test_revised.csv",keep_default_na=False)
#统计各个指标
print('有优惠券但没有购买商品',dfoff[(dfoff["Date"]=='null')&(dfoff["Date_received"]!='null')].shape[0])
print('无优惠券，购买商品',dfoff[(dfoff["Date"]!='null')&(dfoff["Date_received"]=='null')].shape[0])
print("用优惠券购买商品",dfoff[(dfoff["Date"]!='null')&(dfoff["Date_received"]!='null')].shape[0])
print('无优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0])
#取差集
print("在测试集中出现的用户但训练集没有出现",set(dftest['User_id'])-set(dfoff['User_id']))
print("在测试集中出现的用户但训练集没有出现",set(dftest['Merchant_id'])-set(dfoff['Merchant_id']))
#优惠率＇150:20＇满１５０减２０
print('Discount_rate 类型:',dfoff['Discount_rate'].unique())
#user经常活动的地点离该merchant的最近门店距离是x*500米（如果是连锁店，则取最近的一家门店），x\in[0,10]；null表示无此信息，0表示低于500米，10表示大于5公里；
print("Distance ",dfoff['Distance'].unique())

#取特征列
#将满xx减yy类型(xx:yy)的券变成折扣率 : 1 - yy/xx
def getDiscountRate(row):
    if row=='null':
        return 1.0
    elif ':' in row:
        rows=row.split(':')
        return 1-float(rows[1])/float(rows[0])
    else:
        return row

def getDiscountMan(row):
    if ":" in row:
        rows=row.split(':')
        return rows[0]
    else:
        return 0

def getDiscountJian(row):
    if ":" in row:
        rows=row.split(':')
        return rows[0]
    else:
        return 0

def getDiscountType(row):
    if ':' in row:
        return 1
    else:
        return 0
#处理数据
def processData(df):
    df['discount_rate']=df['Discount_rate'].apply(getDiscountRate)
    df['discount_man']=df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian']=df['Discount_rate'].apply(getDiscountJian)
    df['discount_type']=df['Discount_rate'].apply(getDiscountType)
    #距离为null 的转化为-1
    df['distance'] = df['Distance'].replace('null', -1).astype(int)
    return df
dfoff = processData(dfoff)
dftest = processData(dftest)

#优惠时间和消费时间去重排序
#领取优惠券时间
date_received=dfoff['Date_received'].unique()
date_received=sorted(date_received[date_received!='null'])
#消费时间
date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[date_buy != 'null'])
#-1 返回数组倒数第一个数
date_buy = sorted(dfoff[dfoff['Date'] != 'null']['Date'])
print('优惠券收到日期从',date_received[0],'到', date_received[-1])
print('消费日期从', date_buy[0], '到', date_buy[-1])


#每天的顾客消费数量
couponbydate=dfoff[dfoff['Date_received']!='null'][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_received','count']
#用消费券消费的数量
buybydate=dfoff[(dfoff['Date_received']!='null')&(dfoff['Date']!='null')][['Date_received', 'Date']].groupby(['Date_received'],as_index=False).count()
buybydate.columns = ['Date_received','count']

#output
sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.4)
plt.figure(figsize=(12,8))
date_received_dt=pd.to_datetime(date_received,format='%Y%m%d')

plt.subplot(211)
plt.bar(date_received_dt,couponbydate['count'],label = 'number of coupon received' )
plt.bar(date_received_dt,buybydate['count'], label = 'number of coupon used')
plt.yscale('log')
plt.ylabel('Count')
plt.legend()

plt.subplot(212)
plt.bar(date_received_dt,buybydate['count']/couponbydate['count'])
plt.ylabel('Ratio(coupon used/coupon received)')
#plt.show()
plt.tight_layout()
print('输出完成')

#新建星期特征
def getWeekday(row):
    if row=='null':
        return 'null'
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1
dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)
# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )

#将工作日更改为一个热编码
#['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weekday_7']
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
#取离散值，是星期几就取下标几
tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))

tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

#测试集
tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf

#数据标注
def label(row):
    if row['Date_received']=='null':
        return -1
    elif row['Date']!='null':
        td=pd.to_datetime(row['Date'], format='%Y%m%d')-pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td<=pd.Timedelta(15,'D'):
            return 1
    return 0
dfoff['label'] = dfoff.apply(label, axis = 1)
print(dfoff['label'].value_counts())



#切分数据(领取优惠券)
df = dfoff[dfoff['label'] != -1].copy()
train=df[df['Date_received']<'20160516'].copy()
valid=df[(df['Date_received']>='20160516')&(df['Date_received']<='20160615')].copy()
print(train['label'].value_counts())
print(valid['label'].value_counts())

original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols

# model
predictors = original_feature
def check_model(data,predictors):
    #，该损失函数同样可以通过梯度下降算法来求解参数。下面是SGDClassifier的基本使用方法
    #loss	损失函数选择项，字符串型；默认为’hinge’即SVM；’log’为逻辑回归
    #penalty	惩罚方式,字符串型；默认为’l2’;其余有’none’,’l1’,’elasticnet’
    #alpha	惩罚参数,浮点型；默认值为0.0001
    #n_iter	迭代次数，整数型；默认值为5
    #learning_rate	学习速率，字符串型；默认值为’optimal’，根据alpha计算得到
    classifier=lambda:SGDClassifier(loss='log',penalty='elasticnet',fit_intercept=True,max_iter=100,shuffle=True,n_jobs=1,class_weight=None)
    model=Pipeline(steps=[('ss',StandardScaler()),('en',classifier())])
    params={
        'en__alpha': [ 0.001, 0.01, 0.1],
        'en__l1_ratio': [ 0.001, 0.01, 0.1]
    }
    folder=StratifiedKFold(n_splits=3,shuffle=True)
    grid_search=GridSearchCV(model,params,cv=folder,n_jobs=-1,verbose=1)
    grid_search=grid_search.fit(data[predictors],data['label'])
    return grid_search
if not os.path.isfile('1_model.pk1'):
    train.dropna(inplace=True)
    model=check_model(train,predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('1_model.pk1','wb') as f:
        pickle.dump(model,f)
else:
    with open('1_model.pk1','rb') as f:
        model=pickle.load(f)

#预测以及结果评价
y_valid_pred = model.predict_proba(valid[predictors])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:, 1]
valid1.head(2)

# avgAUC calculation
vg = valid1.groupby(['Coupon_id'])
aucs = []
for i in vg:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

# test prediction
y_test_pred=model.predict_proba(dftest[predictors])
dftest1=dftest[['User_id','Coupon_id','Date_received']].copy()
#获取第一位
dftest1['label']=y_test_pred[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)
dftest1.head()