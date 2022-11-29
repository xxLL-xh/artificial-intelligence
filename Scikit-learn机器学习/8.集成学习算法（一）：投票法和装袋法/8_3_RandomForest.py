#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
data = iris.data
target = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(
  data, 
  target, 
  test_size=0.2,
  random_state=1
)
rnd_clf = RandomForestClassifier(max_depth=5, n_estimators=20,max_samples=20, 
                                 bootstrap=True, n_jobs=-1, oob_score=True)
rnd_clf.fit(X_train, Y_train)
print("RandomForestClassifier Score: ", rnd_clf.score(X_test, Y_test))


print("Feature Importance: ", rnd_clf.feature_importances_)


# In[ ]:




