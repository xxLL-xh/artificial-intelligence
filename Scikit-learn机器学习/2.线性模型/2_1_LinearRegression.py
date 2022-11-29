#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入matplotlib用于可视化
import matplotlib.pyplot as plt
# 导入线性模型中的线性回归算法
from sklearn.linear_model import LinearRegression
# 导入数据集
from sklearn.datasets import make_regression
# 用于划分数据集
from sklearn.model_selection import train_test_split

# 载入数据
data, target = make_regression(n_samples=100, n_targets=1, n_features=10, effective_rank=5, bias=10, noise=3)
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
  data, 
  target, 
  test_size=0.2,
  random_state=1
)

model1 = LinearRegression()

model1.fit(X_train, Y_train)
# 预测
result1 = model1.predict(X_test)
# 评分
print("\n训练集评分", model1.score(X_train, Y_train))
print("测试集评分", model1.score(X_test, Y_test))
# 可视化
plt.plot(result1,"ro-",label="predict value")
plt.plot(Y_test,"bo--",label="real value")
plt.title("LinearRegression")
plt.xlabel("index")
plt.ylabel("value")
plt.legend()
plt.show()

