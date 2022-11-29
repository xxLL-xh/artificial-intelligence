#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# 加载数据
wine_data = load_wine()
data = wine_data.data
target = wine_data.target

# 普通KNN实例作为对照
knn_clf = KNeighborsClassifier(n_neighbors=7)
# 流水线
knn_pipeline = Pipeline([
    ("std_scaler", StandardScaler()),
    ("knn_clf", KNeighborsClassifier(n_neighbors=7))
])
# 对未标准化数据对普通KNN进行交叉验证
scores = cross_val_score(knn_clf, data, target, scoring="accuracy", cv=4)
print("KNN Scores: ", scores)
print("KNN Mean:   ", scores.mean())
# 对带标准化功能的KNN流水线进行交叉验证
scores = cross_val_score(knn_pipeline, data, target, scoring="accuracy", cv=4)
print("KNN_pipeline Scores: ", scores)
print("KNN_pipeline Mean:   ", scores.mean())

# 更改流水线中估算器参数
print("以字典形式查看流水线中的估算器：\n", knn_pipeline.named_steps)
# 查看更改前的参数
print("\n更改前", knn_pipeline.named_steps["knn_clf"])
knn_pipeline.set_params(knn_clf__n_neighbors=10)
# 查看更改后的参数
print("更改后", knn_pipeline.named_steps["knn_clf"])

