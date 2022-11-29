#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
# 构造一个用于演示的交错半圆数据集
moons = make_moons(n_samples=(60, 40), noise=0.3, random_state=5)
data = moons[0]
target = moons[1]
# 双月数据集可视化
plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap="cool")
plt.xlabel('X')
plt.ylabel('Y')
plt.title("moons noise=0.3")
plt.show()
# 划分训练集测试集
X_train, X_test, Y_train, Y_test = train_test_split(
  data, 
  target, 
  test_size=0.2,
  random_state=1
)

log_clf = LogisticRegression()
svc_clf = SVC(probability=True)
tree_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier(n_neighbors=10)

vote_clf = VotingClassifier(
    estimators=[("log_clf", log_clf), ("svc_clf", svc_clf), ("tree_clf", tree_clf), ("knn_clf", knn_clf)],
    voting="soft",
)

for clf in (log_clf, svc_clf, tree_clf, knn_clf, vote_clf):
    clf.fit(X_train, Y_train)
    print(clf.__class__.__name__, clf.score(X_test, Y_test))

print(np.round(vote_clf.transform(X_test), 4))


# In[ ]:




