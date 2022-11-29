#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()
data = iris_data.data
target = iris_data.target

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.1,
    random_state=1
)

model4 = LogisticRegression(max_iter=1000)

model4.fit(X_train, Y_train)
print("测试数据分类结果", model4.predict(X_test))
print("测试集实际值", Y_test )
print("\n训练集评分", model4.score(X_train, Y_train))
print("测试集评分", model4.score(X_test, Y_test))

print("\n分类函数的决策系数为", model4.coef_)
print("分类函数的偏差为", model4.intercept_)
print("\n实际迭代次数为", model4.n_iter_)


# In[ ]:




