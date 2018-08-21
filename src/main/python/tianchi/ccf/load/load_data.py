import pandas as pd
dfon=pd.read_csv("/home/moon/work/tianchi/ccf/data/ccf_online_stage1_train.csv")
dfoff=pd.read_csv("/home/moon/work/tianchi/ccf/data/ccf_offline_stage1_train.csv")
dftest=pd.read_csv("/home/moon/work/tianchi/ccf/data/ccf_offline_stage1_test_revised.csv")
print('领取优惠券但没有使用，即负样本',dfoff["Date"]=='null'&dfoff["Coupon_id"]!='null')
print('普通消费日期',dfoff["Date"]!='null'&dfoff["Coupon_id"]=='null')
print("用优惠券消费日期，即正样本",dfoff["Date"]!='null'&dfoff["Coupon_id"]!='null')


dfoff.head(10)


