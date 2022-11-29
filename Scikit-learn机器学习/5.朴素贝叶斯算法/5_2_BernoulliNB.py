#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
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
# 对数据归一化
threshold = 0.5    # 二元离散化阈值
MMS = MinMaxScaler()
MMS.fit(X_train)    # 只在训练集上训练归一化处理器
X_train = MMS.transform(X_train)
X_test = MMS.transform(X_test)
# 本实验在之后用算法构造函数的binarize参数进行二元离散化
# 故，以下步骤仅供参考，不需要加上。
# X_train = binarize(X_train, threshold=threshold)
# X_test = binarize(X_test, threshold=threshold)

model_c = BernoulliNB()
model_bi = BernoulliNB(binarize=threshold)    

# 训练
model_c.fit(X_train, Y_train)
model_bi.fit(X_train, Y_train)
# 评分
print("BernoulliNB Train Score(Continuous)\n", model_c.score(X_train, Y_train))
print("BernoulliNB Test Score(Continuous)\n", model_c.score(X_test, Y_test))
print("BernoulliNB Train Score(Binary Discrete)\n", model_bi.score(X_train, Y_train))
print("BernoulliNB Test Score(Binary Discrete)\n", model_bi.score(X_test, Y_test))

# 查看参数
print("class_count_", model_bi.class_count_)
print("class_log_prior_", model_bi.class_log_prior_)
print("classes_", model_bi.classes_)
print("feature_count_\n", model_bi.feature_count_)
print("feature_log_prob_\n", model_bi.feature_log_prob_)


# In[ ]:




