#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler

# 导入鸢尾花数据
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
# 先对数据归一化，然后把样本特征改成k元离散的数据
MMS = MinMaxScaler()
MMS.fit(X_train)
X_train = MMS.transform(X_train)
X_test = MMS.transform(X_test)
kbs = KBinsDiscretizer(n_bins=6)
kbs.fit(X_train)
X_train = kbs.transform(X_train)
X_test = kbs.transform(X_test)

# 创建实例
model = MultinomialNB()
# 训练
model.fit(X_train, Y_train)
# 评分
print("MultinomialNB Train Score", model.score(X_train, Y_train))
print("MultinomialNB Test Score", model.score(X_test, Y_test))


# In[ ]:




