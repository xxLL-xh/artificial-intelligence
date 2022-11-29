# 扩展出不同状态
def expend(state):  # state为扩展前的状态

    # 检验当前深度是否已经超出上限，若超出上限，则不进行扩展
    if gn[state] > int(depth_limit):
        return None

    expended = []
    k = state.index("0")  # k为0所在的位置
    for a in range(0, len(movs[k])):
        i = k  # i为0的位置
        j = movs[i][a]  # j为待交换元素的位置
        if i > j:
            i, j = j, i
        new = state[: i] + state[j] + state[i + 1: j] + state[i] + state[j + 1:]  # 扩展出的一个新状态
        expended.append(new)
    return expended


def reverse_number(state):
    Sum = 0
    for i in range(1, 9):
        num = 0
        for j in range(0, i):
            if state[j] > state[i] != '0':
                num = num + 1
        Sum += num
    return Sum


def is_solvable(S0):
    i = reverse_number(S0)
    j = reverse_number(goal)
    if i % 2 == j % 2:
        return True
    else:
        return False


def print_result(state):
    # 根据parent中的索引，找出路径
    results = [state]  # 用来存放路径
    while parent[state] != -1:
        state = parent[state]
        results.append(state)
    results.reverse()  # 逆序
    print("可求解，求解过程如下：")
    i = -1
    for result in results:
        i = i + 1
        print("")
        print("step----" + str(i))
        print(result[:3])
        print(result[3:6])
        print(result[6:])
        print("")


def search_depth(S0):
    global parent
    global gn
    global limit

    sum = limit
    opened = []
    closed = []

    # S0加入opened表
    opened.append(S0)

    # 开始搜索
    while opened:

        # 检验搜索次数是否超出限制
        limit = limit - 1
        search_times = sum - limit
        print("正在进行第%d次搜索" % search_times)
        if limit < 1:
            return current

        # opened表中删除第一个状态n，将n放入closed表，
        current = opened.pop()  # 宽度优先：opened表使用先进先出的堆栈结构，使搜索优先偏向先生成的状态
        closed.append(current)
        print("正在搜索第%d层" % gn[current])
        print("curret:" + current)
        print("goal:" + goal)

        # 搜索成功，结束循环
        if current == goal:
            break

        # 扩展当前状态，删除子状态中在opened表或closed表中出现过的状态，避免重复循环搜索
        # 其余子状态加入opened表
        newStates = expend(current)
        if newStates is None:
            continue

        for s in newStates:
            if s not in opened and s not in closed:
                gn[s] = gn[current] + 1
                parent[s] = current
                opened.append(s)
    # 检验是由于opened表为空而停止，还是因为搜索到了解路径而停止。
    if not opened:
        print("有解但是由于深度限制无法求出解！")
    return current


if __name__ == '__main__':

    # 操作算子集合
    movs = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [3, 1, 5, 7], 5: [4, 2, 8], 6: [3, 7], 7: [6, 4, 8],
            8: [7, 5]}

    gn = {}  # gn用于存放各个状态所在深度
    parent = {}  # 用于存放各个状态的父状态，用于输出解路径
    # 输入初始状态和目标状态
    state0 = input("请输入初始状态（从左到右从上到下）：")
    goal = input("请输入目标状态（从左到右从上到下）：")

    # 输入搜索次数上限
    limit = int(input("请输入搜索次数的上限（例如：50000）："))

    # 输入深度上限
    depth_limit = int(input("请输入搜索深度的上限（例如：5000）："))

    parent[state0] = -1  # 初始状态的父状态设置为-1
    gn[state0] = 0  # 初始状态的深度设置为0

    # 判断是否有解
    if state0 == goal:
        print("初始状态与目标状态一致，搜索结束。")
    elif not is_solvable(state0) or len(state0) != 9:
        print("不可达，无解！")
    else:
        current = search_depth(state0)  # 开始搜索
        print_result(current)  # 按格式输出结果
    if limit == 0:
        print("有解但搜索超时，建议更换搜索算法或目标序列！！！")
