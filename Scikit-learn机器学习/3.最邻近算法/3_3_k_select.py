#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# 导入双月型数据集
from sklearn.datasets import make_moons
moons = make_moons(n_samples=(60, 50), noise=0.4, random_state=5)
data = moons[0]
target = moons[1]
# 绘制图像
plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap="cool")
plt.xlabel('X')
plt.ylabel('Y')
plt.title("moons noise=0.4")
plt.show()
# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.3,
    random_state=1
)

k_list = [1, 10, 20, 30, 40, 50, 70]
KNN_models = [KNeighborsClassifier(n_neighbors=k) for k in k_list]
for model in KNN_models:
    model.fit(X_train, Y_train)
    print("k=%d" % model.n_neighbors)
    print("Train Score: ", model.score(X_train, Y_train))
    print("Test Score:  ", model.score(X_test, Y_test))


# In[ ]:




