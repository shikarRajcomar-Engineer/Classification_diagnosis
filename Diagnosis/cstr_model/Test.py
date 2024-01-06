# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
plot_confusion_matrix,
plot_roc_curve,
precision_recall_curve,
average_precision_score)
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from itertools import cycle
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df=pd.read_excel('ti_Fault_15_1.xls')
df=df[['SystemResponse_ 4','SystemResponse_ 5','SystemResponse_ 6','SystemResponse_ 7','SystemResponse_ 8','SystemResponse_ 9','SystemResponse_10','new_column']]
df=df.rename(columns={'SystemResponse_ 4':'Ci','SystemResponse_ 5':'Ti','SystemResponse_ 6':'Tci','SystemResponse_ 7':'Tsp','SystemResponse_ 8':'Qc','SystemResponse_ 9':'Tc','SystemResponse_10':'T','new_column':'Flag'})
df=df[['Ci','Ti','Tci','Tsp','Qc','Tc','T','Flag']]
dfx=df[['Ci','Ti','Tci','Tsp','Qc','Tc','T']]

master=pd.read_excel('testmaster.xlsx')
X=master.drop(columns=['Flag'],axis=1)
y=master[['Flag']]
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(x_train,y_train)
y_pred = random_forest.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)
ypred=random_forest.predict(dfx)
ypred=pd.DataFrame(ypred)
ypred.to_excel('test1.xlsx')