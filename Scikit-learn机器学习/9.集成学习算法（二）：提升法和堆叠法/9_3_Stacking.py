#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes()
data = diabetes.data[:440]
target = diabetes.target[:440]
# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    train_size=380,
    random_state=1
)


def Data_preprocess(X_train, X_test):
    MMS = MinMaxScaler()
    MMS.fit(X_train)
    X1 = MMS.transform(X_train)
    X2 = MMS.transform(X_test)
    STD = StandardScaler()
    STD.fit(X1)
    X1 = STD.transform(X1)
    X2 = STD.transform(X2)
    return X1, X2
# 将数据标准归一化
X_train, X_test = Data_preprocess(X_train, X_test)


# 创建基础学习器
base_estimators = []
knn = KNeighborsRegressor(n_neighbors=12)
base_estimators.append(("knn", knn))
tree = DecisionTreeRegressor(max_depth=3 , random_state=2)
base_estimators.append(("tree", tree))
svr = SVR(kernel="linear")
base_estimators.append(("svr", svr))
# 创建堆叠法集成学习器
model = StackingRegressor(estimators=base_estimators, 
                         final_estimator=LinearRegression(),
                         cv=5)

model.fit(X_train, Y_train)
print("Stacking Ensemble Train Score: ", model.score(X_train, Y_train))
print("Stacking Ensemble Test Score: ", model.score(X_test, Y_test))


# In[2]:


from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


diabetes = load_diabetes()
data = diabetes.data[:440]
target = diabetes.target[:440]
# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    data,
    target,
    train_size=380,
    random_state=1
)

def Data_preprocess(X_train, X_test):
    MMS = MinMaxScaler()
    MMS.fit(X_train)
    X1 = MMS.transform(X_train)
    X2 = MMS.transform(X_test)
    STD = StandardScaler()
    STD.fit(X1)
    X1 = STD.transform(X1)
    X2 = STD.transform(X2)
    return X1, X2

# 基础学习器
base_estimators = []
knn = KNeighborsRegressor(n_neighbors=12)
base_estimators.append(knn)
tree = DecisionTreeRegressor(max_depth=3 , random_state=2)
base_estimators.append(tree)
svr = SVR(kernel="linear")
base_estimators.append(svr)

# 元学习器
meta_estimator = LinearRegression()

# 初始化数据
X_meta_train = np.ones((len(base_estimators), len(X_train)))
Y_meta_train = Y_train

KF = KFold(n_splits=5)
# 用每一折形成的数据切片为每个训练基础学习器学习模型
for train_indices, pred_indices in KF.split(X_train):
    train, to_pred = Data_preprocess(X_train[train_indices], X_train[pred_indices])
    for i in range(len(base_estimators)):
        model = base_estimators[i]
        model.fit(train, Y_train[train_indices])
        # 第i个元学习器的预测结果是元数据的第i列
        X_meta_train[i][pred_indices] = model.predict(to_pred)

X_meta_train = X_meta_train.T    # 转置

X_meta_test = np.ones((len(base_estimators), len(X_test)))
# 初始化数据
Y_meta_test = Y_test
X_train, X_test = Data_preprocess(X_train, X_test)

scores = []
for i in range(len(base_estimators)):
    model = base_estimators[i]
    model.fit(X_train, Y_train)
    X_meta_test[i] = model.predict(X_test)
    scores.append(model.score(X_test, Y_test))

X_meta_test = X_meta_test.T

# 训练元学习器
meta_estimator.fit(X_meta_train, Y_meta_train)
meta_score = meta_estimator.score(X_meta_test, Y_meta_test)
# 打印评分
for i in range(len(base_estimators)):
    print(base_estimators[i].__class__.__name__ + " Test Score:")
    print(scores[i])
print("Stacking Ensemble Test Score:\n", meta_score)


# In[ ]:




