#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# 加载数据
data, target = make_regression(n_samples=100, n_targets=1, n_features=10, n_informative=5, 
                               bias=10, noise=3, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(
  data, 
  target, 
  test_size=0.2,
  random_state=1
)
# 创建并训练模型
model2 = Ridge(alpha=1)
model2.fit(X_train, Y_train)
# 验证模型
print("训练集评分", model2.score(X_train, Y_train))
print("测试集评分", model2.score(X_test, Y_test))
print("\n估计权值为", model2.coef_)
print("估计出的截距为", model2.intercept_)


# In[ ]:




