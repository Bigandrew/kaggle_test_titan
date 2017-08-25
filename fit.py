# -*- coding: utf-8 -*-
# author: Gavin
# Time: 24/8/2017

import pandas as pd

df_train = pd.read_csv(r'C:\Users\yinghe\pyworkspace\KaggleTita\train_cleaned_data.csv')
df_test = pd.read_csv(r'C:\Users\yinghe\pyworkspace\KaggleTita\test_cleaned_data.csv')
# x = pd.read_csv(r'C:\Users\yinghe\pyworkspace\KaggleTita\train_feature.csv', header=None).values
# print(x)
train_feature = df_train.iloc[:,2:]
train_result = df_train.loc[:,['Survived']]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

x_train, x_test, y_train, y_test = train_test_split(train_feature, train_result, test_size=0.15, random_state=None)

# randomforest
rfc = RandomForestClassifier(n_estimators=70, random_state=2)
rfc.fit(x_train, y_train)
#print(rfc.score(x_test, y_test))
y_train_pred = rfc.predict(x_train)
y_test_pred = rfc.predict(x_test)
rf_train = accuracy_score(y_train, y_train_pred)
rf_test = accuracy_score(y_test, y_test_pred)
print('rfc/test accuracies %.4f/%.4f' % (rf_train, rf_test))

#svm
svmMod = svm.SVC(probability=True, random_state=2)
svmMod.fit(x_train, y_train)
y_train_pred = svmMod.predict(x_train)
y_test_pred = svmMod.predict(x_test)
svm_train = accuracy_score(y_train, y_train_pred)
svm_test = accuracy_score(y_test, y_test_pred)
print('SVM train/test accuracies %.4f/%.4f' % (svm_train, svm_test))

# gbdt
gbdt = GradientBoostingClassifier(random_state=2)
gbdt = gbdt.fit(x_train, y_train)
y_train_pred = gbdt.predict(x_train)
y_test_pred = gbdt.predict(x_test)
gb_train = accuracy_score(y_train, y_train_pred)
gb_test = accuracy_score(y_test, y_test_pred)
print('gbdt/test accuracies %.4f/%.4f' % (gb_train, gb_test))
