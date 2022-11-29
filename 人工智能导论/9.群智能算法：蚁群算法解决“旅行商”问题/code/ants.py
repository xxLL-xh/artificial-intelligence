import random
import copy
import numpy as np


class Ant:

    def __init__(self, index):
        self.index = index  # 蚂蚁的序号
        self.init_Ant()

    # 初始化蚂蚁的各个数据结构，随机选择一个出发点
    def init_Ant(self):
        self.path = []  # 当前路径
        self.total_distance = 0
        self.unvisited = np.ones(city_size, int)  # 还没被访问的城市(1代表未访问，0代表已访问)

        first_city = random.randint(0, city_size - 1)  # 随机一个起始城市序号
        # 更新信息
        self.current_city = first_city
        self.path.append(first_city)
        self.unvisited[first_city] = 0

    # 选择下一个城市
    # return 0 ：全部访问完
    # return 1：还要继续游历
    def move_to_next(self):

        # 第一步：计算当前城市到其他城市的概率
        p_numerator = np.zeros(city_size)  # 选择概率的分子
        p_denominator = 0.0  # 选择概率的分母（分子之和）
        trans_p = np.zeros(city_size)  # 概率矩阵

        for i in range(city_size):
            # 如果第i个城市已经访问过，则再访问i的概率为0
            if self.unvisited[i] == 0:
                p_numerator[i] = 0
            else:
                # 随机比例规则公式
                p_numerator[i] = np.dot(pheromone_tb[self.current_city][i], a) * np.dot(eta[self.current_city][i], b)
            p_denominator += p_numerator[i]  # 将分子累加

        # 分母等于0说明每个城市都已经访问过了
        if p_denominator != 0:
            trans_p = p_numerator / p_denominator
        else:
            return 0

        # 第二步：根据概率轮盘赌选下一个城市
        next_city = np.random.choice(np.array(range(city_size)), size=1, replace=True, p=trans_p)[0]

        # 第三步：移动到下一个城市
        self.path.append(next_city)
        self.unvisited[next_city] = 0
        self.current_city = next_city
        return 1

    # 计算路径总距离
    def get_distance(self):
        for i in range(-1, len(self.path) - 1):
            self.total_distance += map[self.path[i]][self.path[i + 1]]

    def travel(self):
        # 初始化蚂蚁，选择出发点
        self.init_Ant()

        flag = 1
        # 蚂蚁开始周游
        while flag == 1:
            flag = self.move_to_next()

        # 计算路径，输出结果
        self.get_distance()


# 更新信息素
def update_pheromone():
    global ant_pop
    global pheromone_tb

    # 信息素增加矩阵
    pheromone_increase = np.zeros((city_size, city_size))

    for ant in ant_pop:
        path_temp = copy.deepcopy(ant.path)
        for i in range(-1, len(path_temp) - 1):
            start = path_temp[i]
            end = path_temp[i + 1]
            pheromone_increase[start][end] += Q / ant.total_distance
    # 信息量更新公式
    pheromone_tb = (1 - r) * pheromone_tb + pheromone_increase


def searchTSP():
    global best_distance, best_path
    # 迭代搜索
    i = time
    while best_distance > 21 and i > 0:
        i -= 1
        """
        print("第%d次迭代：" % (time - i + 1))
        print("信息素矩阵：")
        print(pheromone_tb)
        """

        for ant in ant_pop:
            ant.travel()
            if ant.total_distance < best_distance:
                best_distance = ant.total_distance
                best_path = copy.deepcopy(ant.path)

        # 本代的所有蚂蚁都周游完，再更新信息素
        update_pheromone()

        """print("本次迭代的结果：")
        print(best_path)
        print(best_distance)

    print("搜索完成，最佳路径及其长度为：")
    print(best_distance)
    print(best_path)
    """
    return best_distance, time - i + 1


if __name__ == '__main__':
    # 几个常量
    a = 2
    b = 2
    r = 0.5
    Q = 21

    city_size = 10
    ant_size = 9

    # 地图矩阵(对角线上实际为0，为了计算eta矩阵方便，这里对角线设置为1)
    map = np.array([
        [1, 4, 11, 7, 15, 9, 1, 8, 10, 5],
        [4, 1, 12, 3, 6, 5, 10, 7, 4, 7],
        [11, 12, 1, 14, 7, 2, 3, 1, 2, 11],
        [7, 3, 14, 1, 6, 13, 8, 15, 7, 2],
        [15, 6, 7, 6, 1, 6, 5, 2, 16, 1],
        [9, 5, 2, 13, 6, 1, 2, 18, 10, 9],
        [1, 10, 3, 8, 5, 2, 1, 14, 5, 3],
        [8, 7, 1, 15, 2, 18, 14, 1, 2, 8],
        [10, 4, 2, 7, 16, 10, 5, 2, 1, 14],
        [5, 7, 11, 2, 1, 9, 3, 8, 14, 1]
    ], dtype='int').reshape([city_size, city_size])

    # 能见度矩阵
    eta = 1 / map

    time = 100  # 最大迭代次数

    parameters = [[2, 2, 0.5, 21, 7], [2, 50, 0.5, 21, 7], [50, 2, 0.5, 21, 7], [2, 2, 0.95, 21, 7],
                  [2, 2, 0.05, 21, 7], [2, 2, 0.5, 500, 7], [2, 2, 0.5, 21, 3]]
    for parameter in parameters:
        a = parameter[0]
        b = parameter[1]
        r = parameter[2]
        Q = parameter[3]
        ant_size = parameter[4]

        best_distances = []
        record = []
        best_count = 0

        # 对每种参数组合执行50次实验
        for j in range(50):
            # 最佳路径及其距离
            best_path = []
            best_distance = 80
            # 初始化信息素矩阵，全是为1组成的矩阵
            pheromone_tb = np.ones((city_size, city_size))
            # 初始化蚂蚁群
            ant_pop = [Ant(index) for index in range(ant_size)]

            # 蚁群搜索
            dis, num = searchTSP()

            best_distances.append(dis)
            record.append(num)
            if dis == 21:
                best_count += 1

        print("参数为（顺序是a,b,r,Q,ant_size)：")
        print(parameter)
        print("最佳距离与迭代次数为：")
        print(best_distances)
        print(record)
        print("得到最优解的次数：%d / 50" % best_count)
        print("平均值")
        print(np.mean(best_distances))
        print(np.mean(record))
        print("中位数")
        print(np.median(best_distances))
        print(np.median(record))

