#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据集
diabetes_data = load_diabetes()
data = diabetes_data.data
target = diabetes_data.target
# 划分数据集
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

model = SVR(kernel="sigmoid", gamma="scale", C=9)

model.fit(X_train, Y_train)
print("Train Score:", model.score(X_train, Y_train))
print("Test Score:", model.score(X_test, Y_test))


# In[ ]:




