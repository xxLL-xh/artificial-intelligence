"""
"~"：非
"|"：析取
"-"：空集
"""

Set = ["~T(A)|~T(B)", "~T(A)|~T(C)", "~T(B)|~T(C)", "T(A)|T(B)|T(C)", "~T(C)|~T(A)|~T(B)", "T(C)|T(A)", "T(B)|T(C)",
       "~T(x)|A(x)"]
X = ['x', 'y', 'z', 'x0', 'x1', 'x2', 'x3']
a = ['a', 'b', 'c', 'd', 'A', 'B', 'C', 'D']

S = []
Var = []


# 读入子句集
def read_set():
    global S, Var
    for clause in Set:
        clause = clause.split("|")

        Var_temp = []
        P_temp = []
        for word in clause:
            # 找出谓词
            if word[0] == "~":
                P_temp.append(word[0:2])
            else:
                P_temp.append(word[0])

            # 找出谓词的项
            for i in range(len(word)):
                for j in range(len(word) - 1, i, -1):
                    if word[i] == "(" and word[j] == ")":
                        word_slice = word[i + 1:j]
                        for character in word_slice:
                            if character in X or a:
                                Var_temp.append(character)
        Var.append(Var_temp)
        S.append(P_temp)


# 返回某个文字的逆文字
def opposite(word):
    if word[0] == "~":
        return word.replace("~", "")
    else:
        return "~" + word


# 去除子句中重复的文字。将永真子句从子句集中删去。
def decline_self(I):
    global S, Var
    for count1 in range(len(S[I])):
        if count1 >= len(S[I]):
            continue
        for count2 in range(count1 + 1, len(S[I])):
            if count1 >= len(S[I]):
                continue
            if count2 >= len(S[I]):
                continue
            # 删去重复的文字
            if S[I][count1] == S[I][count2] and Var[I][count2] == Var[I][count2]:
                del S[I][count2]
                del Var[I][count2]
            # 如果存在子式~T(A)|T(A)，说明这个子句永真，直接从子句集删掉
            if opposite(S[I][count1]) == S[I][count2] and Var[I][count1] == Var[I][count2]:
                del S[I]
                del Var[I]
                return 1
    return 0


# return 1:continue
# return 0:flag = 0,break
def decline(I, J, K, L):
    global S, Var
    L1 = S[I][J]
    L2 = S[K][L]
    # 变元用最一般合一代换
    L1_var = Var[I][J]
    L2_var = Var[K][L]
    if L1_var != L2_var:
        # 都在集合a则不能消解
        if L1_var in a and L2_var in a:
            return 1

        elif L1_var in a:
            for n in range(len(Var[K])):
                if Var[K][n] == L2_var:
                    Var[K][n] = L1_var

        elif L2_var in a:
            for m in range(len(Var[I])):
                if Var[I][m] == L1_var:
                    Var[I][m] = L2_var
    print("用%d" % I + "与%d" % K + "进行归结")
    print("用" + "v".join(S[I]) + "与" + "v".join(S[K]) + "进行归结")

    # 消解
    del S[I][J]
    del Var[I][J]
    del S[K][L]
    del Var[K][L]

    # 为了保证最后一个子句的最后一个文字是Answer
    if I == -1:
        S[K] = S[K] + S[I]
        Var[K] = Var[K] + Var[I]
    else:
        S[I] = S[I] + S[K]
        Var[I] = Var[I] + Var[K]

    # 求值时，出现Answer（A）时停止
    if len(S[-1]) == 1 and Var[-1][-1] in a:
        print(S)
        print(Var)
        print("答案是" + Var[-1][-1])

        return 0

    del_self_flag = decline_self(I)
    if K < I:
        del_self_flag = 0
    del S[K - del_self_flag]
    del Var[K - del_self_flag]
    print(S)
    print(Var)

    """
    用反演推理时，出现空集就停
    for index in range(len(S)):
        # 出现空集
        if len(set(S[index])) == 1 and S[index][0] == "-":
            print("NIL")
            return 0
            """


if __name__ == '__main__':
    read_set()
    print("S")
    print(S)
    print("Var")
    print(Var)

    flag = 1
    while True:
        if flag == 0:
            break
        for i in range(len(S)):
            # print("i%d" % i)
            if flag == 0:
                break
            if i >= len(S):
                continue
            for j in range(len(S[i])):
                if flag == 0:
                    break
                if i >= len(S):
                    continue
                if j >= len(S[i]):
                    continue
                # print("j%d" % j)
                for k in range(i + 1, len(S)):
                    if flag == 0:
                        break
                    if k >= len(S):
                        continue
                    # print("k%d" % k)
                    for l in range(len(S[k])):
                        if k >= len(S):
                            continue
                        if l >= len(S[k]):
                            continue
                        # print("l%d" % l)

                        if opposite(S[i][j]) == S[k][l]:
                            flag = decline(i, j, k, l)
                            if flag == 1:
                                continue
                            elif flag == 0:
                                break

                        if opposite(S[-1][0]) == S[k][l]:
                            flag = decline(-1, 0, k, l)
                            if flag == 1:
                                continue
                            elif flag == 0:
                                break
