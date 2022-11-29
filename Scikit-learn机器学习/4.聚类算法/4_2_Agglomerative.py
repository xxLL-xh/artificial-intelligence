#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
# 加载数据集
iris_data = load_iris()
# 从花萼的长度和宽度角度聚类
data = iris_data.data[:, :2]
target = iris_data.target

# 创建聚合聚类算法实例
agcs = [AgglomerativeClustering(n_clusters=3, affinity="euclidean",
        linkage=criterion) for criterion in ["single", "complete", "average"]]
index = 1   # 画布序号
# 打开画布，定义画布大小
plt.figure(num=1, figsize=(12, 3))
# 调节子图边距
plt.subplots_adjust(wspace=0.3,hspace=0.3)
# 对三种模型分别训练并可视化
for agc in agcs:
    # 训练并得到结果
    agc.fit(data)
    labels = agc.labels_
    # 可视化
    plt.subplot(1, 3, index)
    index += 1
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=30, cmap='cool')
    plt.xlabel('Calyx Lenth')
    plt.ylabel('Calyx Width')
    plt.title(agc.linkage)
plt.show()


# In[ ]:




