import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
plt.show()
plt.tight_layout()
print('输出完成')