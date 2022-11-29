import numpy as np


# 激励函数sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 激励函数导数sigmoid_d
def sigmoid_d(x):
    return x * (1 - x)


# 激励函数tanh
def tanh(x):
    return np.tanh(x)


# 激励函数导数tanh_d
def tanh_d(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


# 找出训练数据的最大值
def find_max(case, label):
    a = case.max()
    b = label.max()
    return max(a, b)


class BP_ANN:

    # layers保存结构信息，w是权重初始值范围上限，可利用activation变量选择激活函数，默认用sigmoid函数
    def __init__(self, layers, w, activation=tanh, activation_d=tanh_d):
        self.w0 = w
        self.activation = activation
        self.activation_d = activation_d

        """
        构建网络结构，根据网络结构，设置权重矩阵，并给权重矩阵以一个期望为0,范围在[-w,w]的随机分布赋初始值
        举例：输入层有i个输入、1个偏置，第一个隐层有j个节点、1个偏置
        则，输入层是1*（i+1）矩阵，第一个隐层是1*（j+1）矩阵
        故，应设置（i+1）*（j+1）的权重矩阵，以实现输入层到第一个隐层的变换
        """
        self.weights = []  # 权重矩阵
        self.nodes_in = []  # 节点的输入值u_(k+1) = y_k * w_k + b_k
        self.nodes_out = []  # 节点的输出值y_(k+1) = Sigmoid(u_(k+1))
        self.deltas = []  # 误差

        # 输入层  nodes_out = Sigmoid(nodes_in)
        self.nodes_in.append(np.zeros(layers[0] + 1))  # “+1”是因为多设了一个偏置节点
        self.nodes_out.append(np.zeros(layers[0] + 1))
        # 隐层
        for i in range(1, len(layers) - 1):
            # 初始时，以一个期望为0,范围在[-w,w]的随机分布设置各层的权重
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * self.w0)
            self.nodes_in.append(np.zeros(layers[i] + 1))  # 第i层隐层，每个节点初始为0
            self.nodes_out.append(np.zeros(layers[i] + 1))
            self.deltas.append(np.zeros(layers[i] + 1))
        # 输出层
        self.weights.append((2 * np.random.random((layers[-2] + 1, layers[-1])) - 1) * self.w0)  # 输出节点不需要多设偏置节点
        self.nodes_in.append(np.zeros(layers[-1]))
        self.nodes_out.append(np.zeros(layers[-1]))
        self.deltas.append(np.zeros(layers[-1]))

    # 输入值前向传输, inputs是'numpy.ndarray'类型
    def predict(self, inputs):
        inputs = np.append(inputs, 1.0)
        # 计算输入层
        self.nodes_in[0] = inputs
        self.nodes_out[0] = inputs
        # 计算隐层（偏置节点恒为1）
        for i in range(len(self.weights)):
            # a(i+1) = zi * wi + bi
            self.nodes_in[i + 1] = np.dot(self.nodes_out[i], self.weights[i])
            self.nodes_in[i + 1][-1] = 1.0
            # z(i+1) = S(a(i+1))
            self.nodes_out[i + 1] = self.activation(self.nodes_in[i + 1])
            self.nodes_out[i + 1][-1] = 1.0
        # 计算输出层（没有偏置节点）
        self.nodes_in[-1] = np.dot(self.nodes_out[-2], self.weights[-1])
        self.nodes_out[-1] = self.activation(self.nodes_in[-1])
        return self.nodes_out[-1]

    # 误差值反向传播
    def back_propagate(self, label, learn_step):
        """
        y_k = f_k[u_k]  u_(k+1) = y_k * w_k + b_k
        d_m = [y_m - y_real] * f_m'[u_m]
        d_k = f_k'(u_k) * d_(k+1)*w_k

        """
        # 计算输出层误差: d_m = [y_m - y_real] * f_m'[u_m]
        loss = label - self.nodes_out[-1]
        self.deltas[-1] = loss * self.activation_d(self.nodes_in[-1])

        # 计算其他层误差: d_k = f_k'(u_k) * d_(k+1)*w_k
        for i in range(len(self.nodes_out) - 2, 0, -1):  # 这里-2是因为输入层没有deltas
            # 隐含层节点误差
            self.deltas[i - 1] = np.dot(self.deltas[i], self.weights[i].T) * self.activation_d(self.nodes_in[i])

        # 更新各层权重、偏置值
        for i in range(len(self.weights)):
            """
            权重更新
            变成二维矩阵，输入层没有weight和delta 
            所以这里的nodes_out是第k-1层输出，deltas[i]是第k层偏导，weight[i]是第k到k+1层的权重矩阵
            """
            self.weights[i] += learn_step * np.atleast_2d(self.nodes_out[i]).T.dot(
                np.atleast_2d(self.deltas[i]))  # 权重更新

    def train(self, X, Y, epochs=1000, learn_step=0.1):
        for k in range(epochs):
            for m in range(len(X)):
                case = X[m]
                label = Y[m]
                self.predict(case)
                self.back_propagate(label, learn_step)


if __name__ == '__main__':
    train_data = [[0, 0, 0, 0, 0, 0, 0,  # 0
                   0, 1, 1, 1, 1, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 1
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 1, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 1, 1, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 2
                   0, 0, 1, 1, 1, 0, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 3
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 1, 1, 0, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 4
                   0, 1, 0, 0, 0, 0, 0,
                   0, 1, 0, 1, 0, 0, 0,
                   0, 1, 0, 1, 0, 0, 0,
                   0, 1, 0, 1, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 5
                   0, 1, 1, 1, 1, 1, 0,
                   0, 1, 0, 0, 0, 0, 0,
                   0, 1, 0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 6
                   0, 1, 1, 1, 1, 1, 0,
                   0, 1, 0, 0, 0, 0, 0,
                   0, 1, 0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 7
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 1, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 8
                   0, 1, 1, 1, 1, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0,
                   0, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0,  # 9
                   0, 0, 1, 1, 1, 1, 0,
                   0, 0, 1, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 1, 0,
                   0, 0, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0],
                  ]
    """label = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]"""
    label = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

    test_data = [[0, 0, 0, 0, 0, 0, 0,  # 0
                  0, 1, 1, 1, 1, 1, 0,
                  0, 1, 0, 0, 0, 0.9, 0,
                  0, 1, 0, 0, 0, 1, 0,
                  0, 1, 0, 0, 0, 1, 0,
                  0, 0.9, 0, 0, 0, 1, 0,
                  0, 1, 0, 0, 0, 1, 0,
                  0, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 1
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 1, 0.9, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 0.9, 0, 0, 0,
                  0, 0, 1, 1, 1, 0, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 2
                  0, 0.1, 1, 1, 1, 0, 0,
                  0, 0.9, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 1, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 0,
                  0, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 3
                  0, 0.9, 1, 1, 1, 1, 0,
                  0, 0.1, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 1, 1, 0, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 4
                  0, 1, 0, 0, 0, 0, 0,
                  0.1, 0.9, 0, 1, 0, 0, 0,
                  0, 1, 0, 1, 0, 0, 0,
                  0, 1, 0, 1, 0, 0, 0,
                  0, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 5
                  0, 1, 1, 1, 1, 1, 0,
                  0, 1, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, 0, 0,
                  0, 0.9, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0.1, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 6
                  0, 1, 1, 1, 1, 1, 0,
                  0, 1, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, 0, 0,
                  0, 1, 1, 1, 1, 1, 0,
                  0, 1, 0, 0, 0, 0.9, 0.1,
                  0, 1, 0, 0, 0, 1, 0,
                  0, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 7
                  0, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 1, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0.9, 0.1, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 8
                  0, 1, 1, 1, 1, 1, 0,
                  0, 1, 0, 0, 0, 1, 0,
                  0, 0.9, 0, 0, 0, 1, 0,
                  0, 1, 1, 1, 1, 1, 0,
                  0, 1, 0, 0, 0, 1, 0,
                  0, 1, 0.1, 0, 0, 1, 0,
                  0, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0],

                 [0, 0, 0, 0, 0, 0, 0,  # 9
                  0, 0, 1, 1, 1, 1, 0,
                  0, 0.1, 1, 0, 0, 1, 0,
                  0, 0, 1, 0, 0, 1, 0,
                  0, 0, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0.9, 0,
                  0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0, 0],
                 ]

    X = np.array(train_data)
    Y = np.array(label)
    TEST = np.array(test_data)

    # 这里选用了tanh作为激励函数
    nn = BP_ANN([63, 12, 10], 0.25, activation=tanh, activation_d=tanh_d)
    nn.train(X, Y, 1000, 0.1)

    # 训练
    print("训练样本")
    for index in range(len(X)):
        print("标准%d：" % index)
        print(nn.predict(X[index]))

    print("测试样本")
    for index in range(len(TEST)):
        print("%d的预测结果：" % index)
        result = nn.predict(TEST[index])
        print(result)
        for n in range(result.shape[0]):
            if 0.5 < np.absolute(result[n]) < 0.9:
                print("可能为%d。" % n)
            elif np.absolute(result[n]) > 0.9:
                print("应该是%d。" % n)
