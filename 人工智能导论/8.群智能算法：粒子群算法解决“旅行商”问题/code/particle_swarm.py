import numpy as np
import copy


# 随机初始化粒子的位置（代表一条路径）
# 将每一条路径的最后一位固定为0，表示从0这个城市出发返回0这个城市，这样减少了很多总长度一样的等价路径
def init_xs(pop_size):
    pop = []
    for i in range(pop_size):
        path = np.arange(1, map.shape[0])
        np.random.shuffle(path)  # 随机打乱数组
        xi = list(path)
        xi.append(0)
        pop.append(xi)  # 加入种群
    # print(pop)
    return pop


# 初始化粒子的速度
# 用交换子（一个二元组）代表速度
def init_vs(pop_size):
    v = []
    for i in range(pop_size):
        index1 = np.random.randint(0, city_size - 3)
        index2 = np.random.randint(index1 + 2, city_size - 1)
        vi = [index1, index2, w]  # 某个粒子对应的速度
        v.append([vi])
    # print(v)
    return v


def get_distance(path):
    distance = 0
    for i in range(- 1, len(path) - 1):
        distance += map[path[i]][path[i + 1]]
    return distance


# 分别计算群体中每个例子的适应度，返回
def get_fitness(pop):
    f = []
    for i in range(pop_size):
        path = pop[i]
        fi = 1 / get_distance(path)  # 因为要求最短路径，所以适应度函数设为路径总距离的倒数
        f.append(fi)
    return f


# 更新pbest、gbest
def set_best():
    global pbest_fitness, pbest_list, gbest_fitness, gbest, fitness, Xs
    for k in range(pop_size):
        if pbest_fitness[k] < fitness[k]:
            pbest_fitness[k] = fitness[k]
            pbest_list[k] = copy.deepcopy(Xs[k])

    a = max(fitness)
    if gbest_fitness < a:
        index = fitness.index(a)
        gbest_fitness = a
        gbest = copy.deepcopy(Xs[index])
    return get_distance(gbest)


# 交换子，将path中i位与j位交换
def swap(path, slist):
    for list in slist:
        k = list[0]
        j = list[1]
        r = list[2]
        r0 = np.random.random()
        if r0 <= r:
            path[k], path[j] = path[j], path[k]
    return path


# 求粒子当前位置到目的位置的交换序列
def get_ss(goal, xi, r):
    """
    param goal: pbest 或者 gbest
    param xi: 粒子当前解
    r: 进行这种交换的概率
    return:交换序列
    """
    ss = []
    xic = copy.deepcopy(xi)  # 先将xi、goal复制，因为之后要进行交换，才能求出下一步的交换子。
    goalc = copy.deepcopy(goal)

    for k in range(len(xi)):
        if xic[k] != goalc[k]:  # 如果当前索引的粒子元素不等于目标粒子的对应元素

            j = np.where(xic == goalc[k])[0][0]  # 找到当前粒子和目标粒子不同的元素所在索引
            so = [k, j, r]  # 得到交换子,表示以r的概率对i,j进行操作
            ss.append(so)
            xic[k], xic[j] = xic[j], xic[k]  # 执行交换操作
    return ss  # 返回交换子序列


# 更新第i个粒子的位置和速度
def update(i):
    global Xs, Vs
    # 计算交换序列(更新速度)，即v(i+1) = wv（i）+ r1(pbest-xi) + r2(gbest-xi)
    swap_p = get_ss(pbest_list[i], Xs[i], r1)
    swap_g = get_ss(gbest, Xs[i], r2)
    swap_list = Vs[i] + swap_p + swap_g
    Xs[i] = swap(Xs[i], swap_list)
    for swapi in swap_list:
        swapi[2] = w
    Vs[i] = (swap_list[-V_max:])



# 种群优化
def optimize():
    global num, fitness, best_distance
    # 迭代num次
    i = num
    while best_distance > 22 and i > 0:
        i -= 1
        # 对每个粒子迭代，更新每个粒子的位置和速度
        for j in range(pop_size):
            update(j)  # 更新粒子

        # 计算新的粒子群的适应度
        fitness = get_fitness(Xs)

        # 获取pbest、gbest
        best_distance = set_best()

    """print("找到的最佳路径和最佳路径的距离为：")
    print(gbest)
    print(best_distance)"""
    a = num - i
    return a, best_distance


if __name__ == '__main__':

    city_size = 10  # 城市大小
    pop_size = 60  # 种群大小
    num = 300  # 迭代次数

    fitness = []  # 适应值数组，存放每个个体的适应值
    w = 0.3  # 加速度
    r1 = 0.8  # 认知因子
    r2 = 0.8  # 社会因子

    pbest_list = []  # 每一项是一个粒子的历史最佳位置
    pbest_fitness = []  # 每个粒子历史最佳位置对应的适应值
    gbest = []  # 种群最佳位置
    gbest_fitness = 0  # 种群最佳位置对应的适应值
    best_distance = 0  # 最佳距离

    Xs = []  # 保存粒子的位置信息
    Vs = []  # 保存粒子当前交子序列
    V_max = 10

    # 地图数组
    map = np.array([
        [0, 4, 11, 7, 15, 9, 1, 8, 10, 5],
        [4, 0, 12, 3, 6, 5, 10, 7, 4, 7],
        [11, 12, 0, 14, 7, 2, 3, 1, 2, 11],
        [7, 3, 14, 0, 6, 13, 8, 15, 7, 2],
        [15, 6, 7, 6, 0, 6, 5, 2, 16, 1],
        [9, 5, 2, 13, 6, 0, 2, 18, 10, 9],
        [1, 10, 3, 8, 5, 2, 0, 14, 5, 3],
        [8, 7, 1, 15, 2, 18, 14, 0, 2, 8],
        [10, 4, 2, 7, 16, 10, 5, 2, 0, 14],
        [5, 7, 11, 2, 1, 9, 3, 8, 14, 0]
    ], dtype='int').reshape([city_size, city_size])

    parameters = [[0.4, 0.8, 0.8, 60], [0.8, 0.8, 0.8, 60], [0.4, 0.4, 0.4, 60], [0.4, 0.8, 0.4, 60],
                  [0.4, 0.4, 0.8, 60], [0.4, 0.8, 0.8, 20], [0.4, 0.8, 0.8, 150]]
    for parameter in parameters:
        w = parameter[0]
        r1 = parameter[1]
        r2 = parameter[2]
        pop_size = parameter[3]

        record = []
        best_distances = []
        best_count = 0

        # 对每种参数组合执行50次实验
        for count in range(50):
            Xs = init_xs(pop_size)
            Vs = init_vs(pop_size)
            fitness = get_fitness(Xs)
            pbest_list = copy.deepcopy(Xs)  # 复杂列表要用deepcopy！！！
            pbest_fitness = copy.deepcopy(fitness)
            gbest = []  # 种群最佳位置
            gbest_fitness = 0  # 种群最佳位置对应的适应值
            best_distance = set_best()

            # 优化
            a, d = optimize()

            record.append(a)
            best_distances.append(d)
            if d == 21:
                best_count += 1

        print("参数为（顺序是w、r1、r2、pop_size)：")
        print(parameter)
        print("最佳距离与迭代次数为：")
        print(record)
        print(best_distances)
        print("得到最优解的次数：%d / 50" % best_count)
        print("平均值")
        print(np.mean(record))
        print(np.mean(best_distances))
        print("中位数")
        print(np.median(best_distances))
        print(np.median(record))
