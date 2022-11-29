#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# 导入数据集
diabetes_data = load_diabetes()
data = diabetes_data.data
target = diabetes_data.target
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=1
)
# 标准化
STD = StandardScaler()
STD.fit(X_train)
X_train = STD.transform(X_train)
X_test = STD.transform(X_test)

# 网格搜索
param_grid = {"kernel":["linear", "rbf", "sigmoid"], "degree":[1, 3, 5, 7],
              "gamma":["scale", "auto"], "C":[0.1, 1, 10, 100]}
model = SVR()
grid_search = GridSearchCV(model, param_grid, cv=4, scoring="r2")
grid_search.fit(data, target)
print(grid_search.best_params_)

# 用网格搜索出的参数训练模型
model = SVR(C=10, degree=1, gamma="scale", kernel="sigmoid")
model.fit(X_train, Y_train)
print("parameters = {'C': 10, 'degree': 1, 'gamma': 'scale', 'kernel': 'sigmoid'}:")
print("Score :", model.score(X_test, Y_test))
# 默认参数的对照组
model = SVR()
model.fit(X_train, Y_train)
print("Default Parameters SVR: ", model.score(X_test, Y_test))


# In[ ]:




