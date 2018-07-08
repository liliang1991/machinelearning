import pandas as pd
from sklearn import  linear_model
from sklearn.model_selection import StratifiedKFold

data=pd.read_csv(".csv").fillna(0,axis=1)
x=data.drop('label',axis=1)
y=data['label']
X_train,X_test,y_train,y_test=StratifiedKFold(x,y,test_size=0.1,random_state=0)
reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred)
reg=linear_model.Ridge(alpha=0.5)
reg=linear_model.Lasso(alpha=2,max_iter=10)

#逻辑回归只能做分类
data2=pd.read_csv("").fillna(0,axis=1)
x=data2.drop('label',axis=1)
y=data['label']
X_train2,X_test2,y_train2,y_test2=StratifiedKFold(x,y,test_size=0.1,random_state=0)
cls=linear_model.LogisticRegression()
cls.fit(X_train2,y_train2)
y_pred=cls.predict(X_test2)
metrics.roc_auc_score(y_test2,y_pred)

#svm 分类(不是和数据多)
from sklearn.svm import SVR,SVC
reg=SVR()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
metrics.mean_squared_error(y_test,y_pred)

cls=SVC(probability=True,kernel='rbf',C=0.1,max_iter=10)
cls.fit(X_train,y_train)[:,1]
y_pred=cls.predict_proba(X_test)
metrics.roc_auc_score(y_test,y_pred)


#神经网络
from sklearn.neural_network import MLPClassifier,MLPRegressor
reg=MLPRegressor(hidden_layer_sizes=(10,10,10),learning_rate=0.1)



#决策树
from sklearn.tree import DecisionTreeClassifier
cls=DecisionTreeClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0)
cls.fit(X_train,y_train)[:,1]

#随机森林
from sklearn.ensemble import RandomForestClassifier
cls=RandomForestClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0)
cls.fit(X_train,y_train)[:,1]

#GBDT
from sklearn.ensemble import GradientBoostingClassifier
cls=GradientBoostingClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0)
cls.fit(X_train,y_train)[:,1]

#xgboost n_jobs
import xgboost as XGbClassifier
cls=XGbClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0)
cls.fit(X_train,y_train)[:,1]

#lightgbm
from lightgbm import LGBMClassifier,LGBMRegressor
cls=LGBMClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0)
cls.fit(X_train,y_train)[:,1]


gbm = LGBMClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0)

cls=LGBMRegressor(random_state=0,reg_alpha=0,num_leaves=40,max_depth=7,n_estimators=200,subsample=0.75,colsample_bytree=0.75,reg_lambda=0.5)
cls.fit(X_train,y_train)[:,1]
#kNN(样本多，特征少)




#聚类
from sklearn.cluster import AgglomerativeClustering

data=pd.read_csv(".csv").fillna(0,axis=1)
y=data.label
feature=data.drop('label',axis=1)
