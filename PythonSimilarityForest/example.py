from simforest._simforest import SimilarityForest
import time
import random
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
#修改路径，特征数
if __name__ == '__main__':
  file=open("pbreast-cancer.txt")
  res=[]
  data=[]
  accuracy=0.0
  count=0
  while 1:
    line=file.readline()
    if line=="":
      break
    a=line.split(',')
    res.append(float(a[0]))
    x=[]
    for i in range(1,10):
      x.append(float(a[i]))
    data.append(x)
  for i in range(100):
    rand_state=random.randint(1,5000)
    x_train, x_test, y_train, y_test = train_test_split(data, res, random_state=rand_state, train_size=0.8,test_size=0.2)
    sf = SimilarityForest(n_estimators=100, n_axes=1)
    #进行训练
    sf.fit(np.array(x_train), np.array(y_train))
    #预测
    y_pred = sf.predict(x_test)
    accuracy+=accuracy_score(y_test,y_pred,True)
    count=count+1
print ("SimilarityForest accuracy:",end='')
print(accuracy/count)
    #构建随机森林
    

