#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
# 用于可视化决策树
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import export_text


# 加载数据集
wine_data = load_wine()
data = wine_data.data
target = wine_data.target

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=12
)

model = DecisionTreeClassifier(criterion="gini")

# 训练
model.fit(X_train, Y_train)
# 评分
print("\n纯度的衡量方法为" + model.criterion + "时") 
print("训练集得分为：%.6f" % model.score(X_train, Y_train))
print("测试集得分为：%.6f" % model.score(X_test, Y_test))
# 测试
print("由测试样本得到的预测值")
print(model.predict(X_test))
print("实际值")
print(Y_test)
print("决策树深度：", model.get_depth())
print("决策树叶子节点个数：", model.get_n_leaves())

# 查看参数
print("\nclasses_\n", model.classes_)
print("feature_importances_\n", model.feature_importances_)
print("n_classes_\n", model.n_classes_)
print("tree_\n", model.tree_)

tree_text = export_text(
    model, 
    feature_names=wine_data["feature_names"],
    show_weights=True
) 
print(tree_text)

plt.figure(figsize=(9, 9))
tree.plot_tree(model)
plt.show()


# In[ ]:




