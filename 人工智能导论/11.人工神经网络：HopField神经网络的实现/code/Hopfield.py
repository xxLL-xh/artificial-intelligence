import numpy as np
import copy

train1 = np.array([0, 0, 1, 0, 0,  # +
                   0, 0, 1, 0, 0,
                   1, 1, 1, 1, 1,
                   0, 0, 1, 0, 0,
                   0, 0, 1, 0, 0,
                   ])

train2 = np.array([1, 0, 0, 0, 1,  # x
                   0, 1, 0, 1, 0,
                   0, 0, 1, 0, 0,
                   0, 1, 0, 1, 0,
                   1, 0, 0, 0, 1,
                   ])

test1 = np.array([1, 0, 1, 0, 1,  # +
                  0, 0, 1, 0, 0,
                  1, 1, 1, 1, 1,
                  0, 0, 1, 0, 0,
                  1, 0, 1, 0, 1,

                  0, 0, 1, 0, 0,  # +
                  0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1,
                  0, 0, 1, 0, 0,
                  0, 0, 1, 0, 0,

                  0, 0, 1, 0, 0,  # +
                  0, 0, 1, 0, 0,
                  1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0,

                  0, 0, 1, 0, 0,  # +
                  0, 1, 1, 0, 0,
                  1, 1, 1, 1, 1,
                  0, 0, 1, 0, 0,
                  0, 0, 1, 0, 0,
                  ])

test2 = np.array([1, 0, 1, 0, 1,  # x
                  0, 1, 0, 1, 0,
                  1, 0, 1, 0, 1,
                  0, 1, 0, 1, 0,
                  1, 0, 1, 0, 1,

                  0, 0, 0, 0, 0,  # x
                  0, 1, 0, 1, 0,
                  0, 0, 1, 0, 0,
                  0, 1, 0, 1, 0,
                  1, 0, 0, 0, 1,

                  1, 0, 0, 0, 1,  # x
                  0, 1, 0, 1, 0,
                  0, 0, 1, 0, 0,
                  0, 1, 0, 1, 0,
                  0, 0, 0, 0, 0,

                  1, 0, 0, 0, 1,  # x
                  0, 1, 1, 1, 0,
                  0, 0, 1, 0, 0,
                  0, 1, 0, 1, 0,
                  1, 0, 0, 0, 1,
                  ])

train1 = train1.reshape((1, 25))
train2 = train2.reshape((1, 25))
test1 = test1.reshape((4, 25))
test2 = test2.reshape((4, 25))


def print_num(num):
    i = 0
    while i < len(num):
        print(num[i: i + 5])
        i += 5


def sign(a):
    if a >= 0:
        return 1
    if a < 0:
        return 0


# ?????????????????????????????????
def get_weight(case):
    weight = np.zeros((25, 25))
    case_copy = copy.deepcopy(case)
    for c in case_copy:
        c = 2 * c - 1
        weight += np.dot(np.atleast_2d(c).T, np.atleast_2d(c))
    return weight


# ???????????????????????????????????????
def recognize(case, weight):
    case_copy = copy.deepcopy(case)
    # ????????????????????????????????????????????????????????????
    while 1:
        case_before = copy.deepcopy(case_copy)

        # ????????????
        for index in range(case_copy.shape[0]):
            case_copy[index] = sign(np.dot(np.atleast_2d(case_copy), np.atleast_2d(weight[index]).T)[0][0])
        if case_before.any() == case_copy.any():
            return case_copy


weight1 = get_weight(train1)
weight2 = get_weight(train2)

weight = weight1 + weight2
for i in range(len(weight)):
    for j in range(len(weight[i])):
        if i == j:
            weight[i][j] = 0

print("??????+")
for i in range(len(train1)):
    print("+???????????????%d" % (i + 1))
    print_num(recognize(train1[i], weight))
print("??????+")
for i in range(len(test1)):
    print("??????%d??????????????????????????????" % (i + 1))
    print_num(recognize(test1[i], weight))

print("??????x")
for i in range(len(train2)):
    print("x???????????????%d" % (i + 1))
    print_num(recognize(train2[i], weight))
print("??????x")
for i in range(len(test2)):
    print("??????%d??????????????????????????????" % (i + 1))
    print_num(recognize(test2[i], weight))
