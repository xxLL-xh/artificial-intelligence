#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# 加载数据集
wine_data = load_wine()
data = wine_data.data
target = wine_data.target

wine_data = load_wine()
data = wine_data.data
target = wine_data.target

# 法1：交叉验证进行测试
knn_clf = KNeighborsClassifier()
scores = cross_val_score(estimator=knn_clf, X=data, y=target, scoring="accuracy", cv=4)
print("CV Scores: ", scores)
print("CV Scores Mean:   ", scores.mean())
wine_data = load_wine()
data = wine_data.data
target = wine_data.target

# 法2：划分测试集训练集进行测试
knn_clf = KNeighborsClassifier()
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.25,
    random_state=1, 
)
knn_clf.fit(X_train, Y_train)
print("\nNormal Score: ", knn_clf.score(X_test, Y_test))


# In[ ]:




