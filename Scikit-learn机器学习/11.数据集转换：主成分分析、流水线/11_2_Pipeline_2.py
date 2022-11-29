#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
# 加载数字图像数据集
digits = load_digits()

from sklearn.base import BaseEstimator, TransformerMixin
# 自定义转换器。
class Data_Value_Selector(BaseEstimator, TransformerMixin):
    def __init__(self, Data_or_Value):
        self.Data_or_Value = Data_or_Value
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.Data_or_Value]

from sklearn.base import TransformerMixin     # 可以让类得到fit、transform、fit_transform
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

    
# 对数据进行归一化
data_pipeline = Pipeline([
    ("selector", Data_Value_Selector("data")),
    ("mms_scaler", MinMaxScaler()),
    ("reduce_dim", PCA(n_components=0.95)),
])

# 对标签进行编码
target_pipeline = Pipeline([
    ("selector", Data_Value_Selector("target")),
    ("LB", MyLabelBinarizer()),
])


# 将对数据和标签的操作合在一起
full_pipeline = FeatureUnion(transformer_list=[
    ("data_pipeline", data_pipeline),
    ("target_pipeline", target_pipeline),
    ]) 
# 处理书据
digits_preprocessed = full_pipeline.fit_transform(digits)
# 打印出处理后的数据
print("data")
data = digits_preprocessed[:, :30]
print(data)
print("\ntarget")
target = digits_preprocessed[:, 30:]
print(target)


# In[ ]:




