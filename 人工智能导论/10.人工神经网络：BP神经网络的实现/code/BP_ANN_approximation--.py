import numpy as np
from time import time


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


def function(x, y, z):
    return x + np.power(y, 2) + np.power(z, 3)


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


def function(x, y, z):
    return x + np.power(y, 2) + np.power(z, 3)


if __name__ == '__main__':

    train_num = 300
    test_num = 100
    train_data = np.random.rand(train_num, 3) * 10
    label = []
    for x_y_z in train_data:
        label.append(function(x_y_z[0], x_y_z[1], x_y_z[2]))
    test_data = np.random.rand(test_num, 3) * 10

    X = np.array(train_data)
    Y = np.array(label)
    TEST = np.array(test_data)
    X = X / 10
    Y = Y / function(10, 10, 10)
    TEST = TEST / 10

    # 设置不同的迭代次数
    epochs = [100, 1000]
    for epoch in epochs:
        # 设置不同的隐层节点数
        HNode_nums = [2, 18, 34, 40, 60, 100, 150, 250]
        for HNode_num in HNode_nums:
            # 这里选用了sigmoid作为激励函数
            nn = BP_ANN([3, HNode_num, 1], 0.25, activation=tanh, activation_d=tanh_d)
            # 训练
            start = time()
            nn.train(X, Y, epoch, 0.1)
            end = time()

            print("测试样本")
            error = 0  # 计算和方差
            for index in range(len(TEST)):
                """print(TEST[index] * max_TEST)"""
                pred = nn.predict(TEST[index])[0] * function(10, 10, 10)
                real = function(TEST[index][0] * 10, TEST[index][1] * 10, TEST[index][2] * 10)
                """print("预测值为：%.2f" % pred)
                    print("实际值为：%.2f" % real)"""
                """loss = np.absolute(round(pred - real, 2))
                    print("偏倚为：%.2f" % loss)"""
                error += (real - pred) ** 2
            RMSE = np.sqrt(error / test_num)
            print("*****迭代次数设置为%d," % epoch + "隐层节点数设置为%d时：*****" % HNode_num)
            print("均方根误差为：%.2f" % RMSE)
            print("训练用时间：%s seconds" % (end - start))
