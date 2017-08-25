# -*- coding: utf-8 -*-
# author: Gavin
# Time: 24/8/2017
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


x_train = pd.read_csv(r'C:\Users\yinghe\pyworkspace\KaggleTita\train_feature.csv', header=None).values
y_train = pd.read_csv(r'C:\Users\yinghe\pyworkspace\KaggleTita\target.csv', header=None).values
x_test = pd.read_csv(r'C:\Users\yinghe\pyworkspace\KaggleTita\test_featrue.csv', header=None).values



clf1 = svm.SVC(probability=True, random_state=1)
clf2 = RandomForestClassifier(n_estimators=70, random_state=1)
clf3 = GradientBoostingClassifier(random_state=1)
voting_class = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='soft', weights=[1, 1, 1])

# # try:
# #     vote = voting_class.fit(x_train, y_train)
# # except ValueError as e:
# #      print(e)
vote = voting_class.fit(x_train[1:, :], y_train[1:,:].ravel())
y_test_pred = vote.predict(x_test[1:, :])
pre = pd.DataFrame(y_test_pred, index=None, columns=['Survived'])
pre.to_csv(r"C:\Users\yinghe\pyworkspace\KaggleTita\pre1.csv", index=None)
df = pd.read_csv(r'C:\Users\yinghe\pyworkspace\KaggleTita\test.csv')
dfID = df.PassengerId
ex1 = pd.concat([dfID, pre], axis=1)
ex1.to_csv(r"C:\Users\yinghe\pyworkspace\KaggleTita\ex1.csv", index=None)


# clf1 = svm.SVC(probability=True, random_state=5)
# clf2 = RandomForestClassifier(n_estimators=70, random_state=5)
# clf3 = GradientBoostingClassifier(random_state=5)
# voting_class = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='soft', weights=[1, 1, 1])
# vote = voting_class.fit(x_train[1:, :], y_train[1:,:].ravel())
# y_test_pred = vote.predict(x_test[1:, :])
# pre = pd.DataFrame(y_test_pred, index=None, columns=['Survived'])
# pre.to_csv(r"C:\Users\yinghe\pyworkspace\KaggleTita\pre4.csv", index=None)
# df = pd.read_csv(r'C:\Users\yinghe\pyworkspace\KaggleTita\test.csv')
# dfID = df.PassengerId
# ex5 = pd.concat([dfID, pre], axis=1)
# ex5.to_csv(r"C:\Users\yinghe\pyworkspace\KaggleTita\ex5.csv", index=None)