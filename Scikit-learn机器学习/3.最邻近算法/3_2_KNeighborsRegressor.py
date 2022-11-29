#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 加载数据集
iris_data = load_iris()
# 用鸢尾花的花萼长度、花萼宽度、花瓣长度，来预测花瓣宽度
data = iris_data.data[:, :-1]    
target = iris_data.data[:, np.newaxis, -1]
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

# 创建实例
model2 = KNeighborsRegressor(n_neighbors=6)
# 训练
model2.fit(X_train, Y_train)
# 预测测试集
result = model2.predict(X_test)
# 评分
print("\n训练集评分", model2.score(X_train, Y_train))
print("测试集评分", model2.score(X_test, Y_test))

# 可视化
plt.plot(result,"ro-",label="predict value")
plt.plot(Y_test,"bo--",label="real value")
plt.title("KNNRegression")
plt.xlabel("index")
plt.ylabel("value")
plt.legend()
plt.show()


# In[ ]:




