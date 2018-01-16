from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import random
from sklearn.metrics import accuracy_score
#需要修改的部分：1.文件名  2.特征数  3.svm中的gamma值，通常为1/n，n为特征数量 
file=open("pa2a.txt")
res=[]
data=[]
accuracy=0.0
count=0
while 1:
  line=file.readline()
  if line=="":
    break
  a=line.split(',')
  res.append(a[0])
  x=a[1:123]
  data.append(x)

for i in range(100):
  rand_state=random.randint(1,5000)
  x_train, x_test, y_train, y_test = train_test_split(data, res, random_state=rand_state, train_size=0.8,test_size=0.2)
  clf = svm.SVC(C=1.0, kernel='rbf', gamma=0.01, decision_function_shape='ovr')
  clf.fit(x_train,y_train)
  y_pred=clf.predict(x_test)
  accuracy+=accuracy_score(y_test,y_pred,True)
  count=count+1
print ("SVM accuracy:",end='')
print(accuracy/count)
'''
accuracy=0.0
count=0
for i in range(100):
  rand_state=random.randint(1,5000)
  x_train, x_test, y_train, y_test = train_test_split(data, res, random_state=rand_state, train_size=0.8,test_size=0.2)
  rf=RandomForestClassifier(n_estimators=100,max_depth=10)
  rf.fit(x_train,y_train)
  y_pred=rf.predict(x_test)
  accuracy+=accuracy_score(y_test,y_pred,True)
  count=count+1
print ("Random Forest accuracy:",end='')
print(accuracy/count)'''