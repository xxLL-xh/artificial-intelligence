#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split
# 导入数字图像数据集
digits = load_digits()
data = digits.data
target = digits.target
images = digits.images
# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=1
)
# 在对数据预处理前，先保存下测试集的数字图像和实际值
test_images = X_test.reshape(360, 8, 8)
# 归一化
MMS = MinMaxScaler()
MMS.fit(X_train)
X_train = MMS.transform(X_train)
X_test = MMS.transform(X_test)

# 创建模型
model = SVC(kernel="rbf")

# 训练模型
model.fit(X_train, Y_train)
# 评分
print("Train Score：\n", model.score(X_train, Y_train))
print("Test Score:\n", model.score(X_test, Y_test))
# 测试
for i in range(4):
    print("\n第%d个测试数据" % (i + 1))
    print("图像为：")
    plt.matshow(test_images[i])
    plt.show()
    print("识别结果为：", model.predict(X_test[i].reshape(1, -1))[0])
    print("实际值为：", Y_test[i])

print("class_weight_", model.class_weight_)
print("classes_", model.classes_)
#print("coef_", model.coef_)    # coef_是线性核函数的参数
print("intercept_\n", model.intercept_)
print("fit_status_", model.fit_status_)
#print("support_", model.support_)
#print("support_vectors_", model.support_vectors_)
print("n_support_", model.n_support_)
print("shape_fit_", model.shape_fit_)


# In[ ]:




