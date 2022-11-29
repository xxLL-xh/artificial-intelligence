#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入numpy用于数据处理
import numpy as np
# 导入matplotlib库用于数据可视化
import matplotlib.pyplot as plt
# 导入线性模型中的线性回归算法
from sklearn.linear_model import LinearRegression
# 导入划分数据集方法
from sklearn.model_selection import train_test_split

# 生成100个[0， 10]的随机数，改为列向量
X = np.random.random(200) * 20 - 10
Y = 1.5 * X - 1.2 + np.random.randn(200)

# 改为scikit-learn库中算法可以直接接受的二维上的列向量
data = X.reshape(-1, 1)
target = Y.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=1
)

model = LinearRegression()
model.fit(X_train, Y_train)

# 打印训练出的参数信息
print("训练出的截距为：%.4f" % model.intercept_)
print("训练出的权值为：%.4f\n" % model.coef_)

# 预测
Y_predict = model.predict(X_test)
# 模型评分
print("训练出的模型在训练集上得分为：%.6f" % model.score(X_train, Y_train))
print("训练出的模型在测试集上得分为：%.6f\n" % model.score(X_test, Y_test))

# 预测结果图像
plt.title("Predict Result")
plt.scatter(X_test, Y_test, label="Test Data", s=30, c="r")
plt.plot(X_test, Y_predict, label="Fitted", c="b")
plt.legend(loc=0)
plt.show()


# In[ ]:




