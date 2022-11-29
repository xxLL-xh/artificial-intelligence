"""
规则一：任何人的兄弟不可能是女性。
规则二：任何人的姐妹不可能是男性。
已知事实：Marry是Bill的姐姐或妹妹
问：（1）Marry是不是Tom的兄弟？
（2）Marry是不是Bill的姐妹？
（3）Marry是不是Tom的姐妹？
"""

# 创建几个集合
sister_of_Bill = {""}
sister_of_Tom = {""}
brother_of_Bill = {""}
brother_of_Tom = {""}
male = {""}
female = {""}


# 实现 谓词sister(x1,x2):x1是x2的姐妹
def sister(x1, x2, sister_of_x2):
    print(x1 + "是" + x2 + "的姐妹")
    sister_of_x2.add(x1)
    print("已将" + x1 + "添加进" + x2 + "姐妹的集合")


# 实现 谓词brother(x3,x4):x3是x4的兄弟
def brother(x3, x4, brother_of_x4):
    print(x3 + "是" + x4 + "的兄弟")
    brother_of_x4.add(x3)
    print("已将" + x3 + "添加进" + x4 + "兄弟的集合")


# 实现 谓词is_female(y1): y1是女性
def is_female(y1):
    global female
    print(y1 + "是女性")
    female.add(y1)


# 实现 谓词is_male(y2): y2是男性
def is_male(y2):
    global male
    print(y2 + "是男性")
    male.add(y2)


# 用两条规则判断性别
def judge_gender(A):
    # 规则一 ：任何人的兄弟不可能是女性（所以是男性）。
    if A in brother_of_Tom or A in brother_of_Bill:
        is_male(A)

    # 规则二 ：任何人的姐妹不可能是男性（所以是女性）。
    if A in sister_of_Tom or A in sister_of_Bill:
        is_female(A)


# 判断x是否是y的兄弟
def judge_brother(x, y, brother_of_y):
    if x in brother_of_y:
        print(x + "是" + y + "的兄弟")

    # 这里用到了规则一的逆否命题： 女性不可能是任何人的兄弟
    elif x in female:
        print(x + "不是" + y + "的兄弟")
    else:
        print("信息不足！" + x + "有可能是" + y + "的兄弟")


# 判断x是否是y的姐妹
def judge_sister(x, y, sister_of_y):
    if x in sister_of_y:
        print(x + "是" + y + "的姐妹")

    # 这里用到了规则二的逆否命题： 男性不可能是任何人的姐妹
    elif x in male:
        print(x + "不是" + y + "的姐妹")
    else:
        print("信息不足！" + x + "有可能是" + y + "的姐妹")


if __name__ == '__main__':

    # 事实：Marry是Bill的姐姐
    sister("Marry", "Bill", sister_of_Bill)

    # 利用已知规则和事实进行推理
    judge_gender("Marry")
    print("")
    print("可以得出结论：")
    judge_brother("Marry", "Tom", brother_of_Tom)
    judge_sister("Marry", "Bill", sister_of_Bill)
    judge_sister("Marry", "Tom", sister_of_Tom)