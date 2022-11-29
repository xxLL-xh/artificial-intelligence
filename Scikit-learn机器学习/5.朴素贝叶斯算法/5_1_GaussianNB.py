#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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

model = GaussianNB()

# 训练
model.fit(X_train, Y_train)
# 测试
print("测试数据预测值：", model.predict(X_test))
print("测试数据实际值：", Y_test)
# 得分
print("训练集上得分为：", model.score(X_train, Y_train))
print("测试集上得分为：", model.score(X_test, Y_test))
# X_test中前五个样本属于每一个类别的概率
print("前五条测试数据属于每个类的概率（保留三位）：\n", np.round(model.predict_proba(X_test[:5]), 3))

# 查看参数
print("class_count_", model.class_count_)
print("class_prior_", model.class_prior_)
print("classes_", model.classes_)
print("var_\n", model.var_)
print("theta_\n", model.theta_)


# In[ ]:




