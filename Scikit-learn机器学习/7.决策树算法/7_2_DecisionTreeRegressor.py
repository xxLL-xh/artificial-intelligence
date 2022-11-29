#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 载入数据
data = np.array(range(100)).reshape(-1, 1)
target = 2 * data + 10 + np.random.randn(100, 1) * 20

X_train, X_test, Y_train, Y_test = train_test_split(
  data, 
  target, 
  test_size=0.2,
  random_state=1,
)
Y_test = np.sort(Y_test, axis=0)
X_test = np.sort(X_test, axis=0)

model1 = DecisionTreeRegressor(max_depth=1)
model2 = DecisionTreeRegressor(max_depth=3)
model3 = DecisionTreeRegressor(max_depth=5)
model4 = LinearRegression()
model1.fit(X_train, Y_train)
model2.fit(X_train, Y_train)
model3.fit(X_train, Y_train)
model4.fit(X_train, Y_train)

# 拟合图像
plt.figure(figsize=(10, 8))
plt.scatter(X_test, Y_test, s=40, c="black", label="data")
plt.plot(X_test, model1.predict(X_test), "co--", label="max_depth=1", alpha=0.6)
plt.plot(X_test, model2.predict(X_test), "go--", label="max_depth=3", alpha=0.6)
plt.plot(X_test, model3.predict(X_test), "ro--", label="max_depth=5", alpha=0.6)
plt.plot(X_test, model4.predict(X_test), "bo--", label="liner regression", alpha=0.6)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
# 训练集上的评分
print("max_depth=1 Train Score", model1.score(X_train, Y_train))
print("max_depth=3 Train Score", model2.score(X_train, Y_train))
print("max_depth=5 Train Score", model3.score(X_train, Y_train))
print("liner regression Train Score", model4.score(X_train, Y_train))
# 测试集上的评分
print("\nmax_depth=1 Test Score", model1.score(X_test, Y_test))
print("max_depth=3 Test Score", model2.score(X_test, Y_test))
print("max_depth=5 Test Score", model3.score(X_test, Y_test))
print("liner regression Test Score", model4.score(X_test, Y_test))


# In[ ]:




