#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris_data = load_iris()
data = iris_data.data
target = iris_data.target
# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=1
)
X = X_test[-3:]    # 取最后3个测试数据单独检验
# 标准化
STD = StandardScaler()
STD.fit(X_train)
X_train = STD.transform(X_train)
X_test = STD.transform(X_test)

model1 = KNeighborsClassifier(n_neighbors=6, p=3)

# 训练
model1.fit(X_train, Y_train)
# 评分
print("\n训练集评分", model1.score(X_train, Y_train))
print("测试集评分", model1.score(X_test, Y_test))
print("最后三条测试数据", X)
X = STD.transform(X)
print("最后三条测试数据最近的6个点的距离及其序号索引")
print(model1.kneighbors(X, n_neighbors=6, return_distance=True))
print("最后三条测试数据属于每一类的概率")
print(model1.predict_proba(X))
print("最后三条测试数据的分类结果为", model1.predict(X))
print("最后三条测试数据的实际类别位", Y_test[-3:])

# 查看参数
print("classes_", model1.classes_)
print("effective_metric_", model1.effective_metric_)
print("effective_metric_params_", model1.effective_metric_params_)
print("n_features_in_", model1.n_features_in_)
# print("feature_names_in_", model1.feature_names_in_)
print("n_samples_fit_", model1.n_samples_fit_)


# In[ ]:




