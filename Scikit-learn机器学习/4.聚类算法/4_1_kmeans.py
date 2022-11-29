#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 加载数据
iris_data = load_iris()
# 从花萼的长度和宽度角度聚类
data = iris_data.data[:, :2]
target = iris_data.target

# 创建k-均值聚类算法实例
km = KMeans(n_clusters=3)

# 训练数据
km.fit(data)
print("前两条数据(X)\n", data[:2])
print("前两条数据在簇距离空间中的坐标(X_new)\n", km.transform(data)[:2])
print("质心坐标\n", km.cluster_centers_)
print("前两条数据所属类别",km.labels_[:2])

# 得到聚类结果
labels = km.labels_    # 每个点的聚类结果标签
centers = km.cluster_centers_    # 聚类完成后各个簇的质心
# 查看参数
print("\ncluster_centers_\n", centers)
print("labels_\n", labels)
print("inertia_\n", km.inertia_)
print("n_iter_\n", km.n_iter_)

# 打开画布，定义画布大小
plt.figure(num=1, figsize=(12, 3))
# 调节子图边距
plt.subplots_adjust(wspace=0.5,hspace=0.5)
# 用花萼特征聚类前的分布情况
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], s=30)
plt.xlabel('Calyx Lenth')
plt.ylabel('Calyx Width')
plt.title("Raw Data of Calyx")
# 用花萼特征聚类后
plt.subplot(1, 3, 2)
plt.scatter(data[:, 0], data[:, 1], c=labels, s=30, cmap='cool')
# 绘制质心点
plt.scatter(centers[:,0], centers[:,1], c='red', marker='o', s=60)
plt.xlabel('Calyx Lenth')
plt.ylabel('Calyx Width')
plt.title("K-Means by Calyx Features")
# 实际的类别
plt.subplot(1, 3, 3)
plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap='summer')
plt.xlabel('Calyx Lenth')
plt.ylabel('Calyx Width')
plt.title("Real Label")
plt.show()


# In[ ]:




