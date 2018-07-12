from nltk.corpus import webtext as wn

import pandas as pd

import numpy as np

from scipy import stats
from sklearn.preprocessing import MinMaxScaler, Imputer
#import nltk
#nltk.download()
# x = wordnet.synsets('recommended')[0]
#
# y = wordnet.synsets('suggested')[0]
#
# print(x.path_similarity(y))

data=pd.read_csv("combined.csv")
wordsList=np.array(data.iloc[:,[0,1]])
simScore=np.array(data.iloc[:,[2]])

predScoreList=np.zeros( (len(simScore),1) )
for i, (word1, word2) in enumerate(wordsList):
    count=0
    synsets1=wn.synsets(word1)
    synsets2=wn.synsets(word2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            score=synset1.path_similarity(synset2)
            if score is not None:
                predScoreList[i,0]+=score
                count+=1
            else:
                print (synset1,"path_similarity", synset2, "is None", "=="*10)
    predScoreList[i,0]=predScoreList[i,0]*1.0/count
#max(predScoreList)=array([ 1.])    min(predScoreList)=array([ 0.04166667])


imp=Imputer(missing_values='NaN', strategy='mean', axis=0)
impList=imp.fit_transform(predScoreList)
mms=MinMaxScaler(feature_range=(0.0,10.0))
impMmsList=mms.fit_transform(impList)
#max(impMmsList)=array([ 10.])    min(impMmsList)=array([ 0.])
#impMmsList.mean()=0.74249450173161169

(coef1, pvalue)=stats.spearmanr(simScore, impMmsList)
#(correlation=0.3136469783526708, pvalue=1.6943792485183932e-09)


