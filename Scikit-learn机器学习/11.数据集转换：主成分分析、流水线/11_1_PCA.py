#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# 加载数据
digits_data = load_digits()
data = digits_data.data
target = digits_data.target
# 归一化
MMS = MinMaxScaler()
MMS.fit(data)
data = MMS.transform(data)
# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=1
)

# 计算最佳维度（方差解释率达到95%视为一个好的降维）
pca = PCA(n_components=0.95)
pca.fit(X_train)
# 查看出成分分析结果
print("Original Train Features Number：", pca.n_features_)
print("95% Best Dimensions:", pca.n_components_)
print("Components:\n", pca.components_)
print("Explained Variance Ratio:\n", pca.explained_variance_ratio_)

model = SVC()
model.fit(X_train, Y_train)
print("Train Score Before decomposition: ", model.score(X_train, Y_train))
print("Test Score Before Decomposition: ", model.score(X_test, Y_test))

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# 降维后测试
model = SVC()
model.fit(X_train_pca, Y_train)
print("Train Score After decomposition: ", model.score(X_train_pca, Y_train))
print("Test Score After decomposition: ", model.score(X_test_pca, Y_test))


# In[ ]:




