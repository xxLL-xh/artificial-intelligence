import numpy as np
import copy


# 随机创建大小为pop_size的种群
# 将每一条基因的最后一位固定为0，表示从0这个城市出发返回0这个城市，这样减少了很多总长度一样的等价路径
def create_pop(pop_size):
    pop = []
    for i in range(pop_size):
        path = np.arange(1, map.shape[0])
        np.random.shuffle(path)  # 随机打乱数组
        pop.append(np.append(path, 0))  # 加入种群
    # print(np.array(pop))
    return np.array(pop)


# 计算沿着染色体上基因路径走一圈的距离
def get_distance(path):
    distance = 0
    for i in range(- 1, len(path) - 1):
        distance += map[path[i]][path[i + 1]]
    return distance


# 计算pop中每一个染色体的适应值
def get_fitness(pop):
    f = np.array([])
    for i in range(pop.shape[0]):  # 此时子代种群的的个体数不为size，所以要用pop.shape[0]获取当前种群大小
        path = pop[i]
        fi = 1 / get_distance(path)  # 因为要求最短路径，所以适应度函数设为路径总距离的倒数
        f = np.append(f, fi)
    return f


# 利用了np.random.choice()函数，用轮盘赌选法得到一个新群体
def select_pop(pop, f):
    possibility = f / f.sum()
    chosen = np.random.choice(np.arange(pop.shape[0]), size=pop_size, replace=False, p=possibility)
    # print(chosen)
    pop_new = pop[chosen, :]
    # print(pop_new)
    return pop_new


def cross_pop(p1, p2):
    if np.random.random() > pc:
        # print("因为概率,不交差")
        return p1, p2

    else:
        # S1:交换
        # 随机生成两个交叉点
        index1 = np.random.randint(0, city_size - 3)  # randint(a, b)函数返回的是[a, b)中的随机整数
        index2 = np.random.randint(index1 + 2, city_size)  # 保证两个交叉点中至少有一个元素
        # print("%d与"%index1 + "%d内的部分交叉"%index2)

        # 将交叉点内的部分交换
        p1[index1: index2], p2[index1: index2] = p2[index1: index2], p1[index1: index2]

        # S2：修正（去重复）
        # 找出重复部分
        i, j = find_repeat(p1, index1, index2), find_repeat(p2, index1, index2)
        # 不断交换，直到不再有重复
        while i is not None and j is not None:
            p1[i], p2[j] = p2[j], p1[i]
            i, j = find_repeat(p1, index1, index2), find_repeat(p2, index1, index2)
        # print("交叉后为")
        # print(p1, p2)
        return p1, p2


def find_repeat(path, index1, index2):
    for k in path:
        if path.count(k) > 1:
            return path.index(k)
    return None


# 这里采用的是逆转异变，将基因中随机一段序列以逆向排序插回原序列
def mut_pop(path):
    if np.random.random() > pm:
        # print("因为概率不异变")
        return path
    else:
        # 异变
        index1 = np.random.randint(0, city_size - 3)
        index2 = np.random.randint(index1 + 2, city_size - 1)
        # print("%d和"%index1 + "%d两个点位交换来变异"%index2)
        while (index2 - index1) >= 0:
            path[index1], path[index2] = path[index2], path[index1]
            index1 += 1
            index2 -= 1
        # print("变异后为")
        # print(path)
        return path


def evolution(pop):
    global num
    # 迭代num次
    i = num
    best_distance = 100    # 刚开始时把best_distance初始化为一个较大的数
    while best_distance > 22 and i > 0:
        i -= 1

        fitness = get_fitness(pop)

        # 计算当前种群中最佳适应的个体、对应路径、路径总长度
        local_best_index = np.argmax(fitness)
        local_best_path = copy.deepcopy(pop[local_best_index])
        local_best_distance = get_distance(local_best_path)

        # 每次循环前，更新最佳路径及其总长度
        if local_best_distance < best_distance:
            best_path = copy.deepcopy(local_best_path)
            best_distance = local_best_distance

        # 正式开始
        # 选择
        # print("第%d次进化选择的序号和路径："%i)
        pop = select_pop(pop, fitness)
        # 交叉
        # 每次从选择出的个体中选两个进行交配
        pool = list(range(pop_size))
        for n in range(int(pop_size / 2)):
            parent1 = np.random.choice(pool)
            pool.remove(parent1)
            parent2 = np.random.choice(pool)
            pool.remove(parent2)
            child1, child2 = cross_pop(list(pop[parent1]), list(pop[parent2]))
            pop = np.append(pop, [child1], axis=0)
            pop = np.append(pop, [child2], axis=0)

        # 变异
        for j in range(pop_size):
            pop[j] = np.array(mut_pop(pop[j]))

    """print("结束！最后得到最佳路径为")
    print(best_path)
    print("最佳路径长度为")
    print(best_distance)"""
    return num - i, best_distance


if __name__ == '__main__':
    city_size = 10    # 城市个数
    pop_size = 60    # 种群大小(最好是偶数)
    pop = np.array([])    # 种群数组
    fitness = np.zeros(pop_size)    # 适应值数组，存放每个个体的适应值
    pc = 0.85    # 交叉概率
    pm = 0.1    # 变异概率
    num = 300    # 最大迭代次数

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

    parameters = [[0.85, 0.1, 60], [0.45, 0.1, 60], [0.85, 0.45, 60], [0.85, 0.1, 20], [0.85, 0.1, 150]]
    for parameter in parameters:
        pc = parameter[0]
        pm = parameter[1]
        pop_size = parameter[2]

        record = []
        best_distances = []
        best_count = 0

        # 对每种参数组合执行50次实验
        for j in range(50):
            pop = create_pop(pop_size)

            # 进化
            a, d = evolution(pop)

            record.append(a)
            best_distances.append(d)
            if d == 21:
                best_count += 1

        print("参数为（顺序是pc、pm、pop_size)：")
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
