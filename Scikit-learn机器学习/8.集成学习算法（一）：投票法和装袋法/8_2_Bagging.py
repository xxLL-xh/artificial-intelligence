#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
# 创建双月型数据集
moons = make_moons(n_samples=(120, 80), noise=0.3, random_state=5)
data = moons[0]
target = moons[1]
X_train, X_test, Y_train, Y_test = train_test_split(
  data, 
  target, 
  test_size=0.2,
  random_state=1
)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_depth=5), n_estimators=500, 
    max_samples=20, bootstrap=True, n_jobs=-1, oob_score=True
)
past_clf = BaggingClassifier(
    DecisionTreeClassifier(max_depth=5), n_estimators=20, 
    max_samples=20, bootstrap=False, n_jobs=-1
)
tree_clf = DecisionTreeClassifier(max_depth=5)

for clf in (tree_clf, bag_clf, past_clf):
    clf.fit(X_train, Y_train)
    print(clf)
    print("train score:", clf.score(X_train, Y_train))
    print("test score:", clf.score(X_test, Y_test))
print("\n有放回抽样集成的包外评估：", bag_clf.oob_score_)    # 包外评估


# In[ ]:




