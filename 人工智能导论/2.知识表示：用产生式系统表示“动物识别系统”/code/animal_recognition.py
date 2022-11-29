"""特征库：1至11是用于粗分类的简单特征，21至24是动物较粗的分类，
12至20是7种动物可能具有的个性特征
"""
# 特征
Features = ["    ", "有毛发", "有奶", "有羽毛", "会飞", "会下蛋", "吃肉", "有犬齿", "有爪", "眼盯前方", "有蹄",
            "反刍", "黄褐色", "暗斑点", "黑色条纹", "长脖子", "长腿", "不会飞", "有黑白二色", "会游泳", "善飞",
            "哺乳动物", "鸟", "食肉动物", "蹄类动物"]
# 动物种类
Animals = ["    ", "老虎", "金钱豹", "斑马", "长颈鹿", "企鹅", "鸵鸟", "信天翁"]

# 建立规则库
"""
规则：
R1：if 动物有毛发 then 动物是哺乳动物
R2：if 动物有奶 then 动物是哺乳动物
R3：if 动物有羽毛 then 动物是鸟
R4：if 动物会飞 and 会生蛋 then 动物是鸟
R5：if 动物吃肉 then 动物是食肉动物
R6：if 动物有犀利牙齿 and 有爪 and 眼向前方 then 动物是食肉动物
R7：if 动物是哺乳动物and有蹄then动物是有蹄类动物
R8：if 动物是哺乳动物and反刍then动物是有蹄类动物
R9：if 动物是哺乳动物and是食肉动物and有黄褐色 and 有暗斑点 then 动物是豹
R10：if 动物是哺乳动物 and是食肉动物and有黄褐色 and有黑色条纹 then 动物是虎
R11：if动物是有蹄类动物 and 有长脖子and有长腿and有暗斑点 then 动物是长颈鹿
R12：if 动物是有蹄类动物 and有黑色条纹 then 动物是斑马
R13：if 动物是鸟and不会飞 and有长脖子and有长腿 and有黑白二色 then 动物是鸵鸟
R14：if 动物是鸟 and不会飞 and会游泳 and有黑白二色 then 动物是企鹅
R15：if 动物是鸟 and善飞 then 动物是信天翁
"""
# 规则
r1 = {"有毛发"}  # 哺乳动物
r2 = {"有奶"}  # 哺乳动物
r3 = {"有羽毛"}  # 鸟
r4 = {"会飞", "会下蛋"}  # 鸟
r5 = {"吃肉"}  # 食肉动物
r6 = {"有犬齿", "有爪", "眼盯前方"}  # 食肉动物
r7 = {"哺乳动物", "有蹄"}  # 蹄类动物
r8 = {"哺乳动物", "反刍"}  # 蹄类动物
r9 = {"哺乳动物", "食肉动物", "黄褐色", "暗斑点"}  # 金钱豹
r10 = {"哺乳动物", "食肉动物", "黄褐色", "黑色条纹"}  # 老虎
r11 = {"蹄类动物", "长脖子", "长腿", "暗斑点"}  # 长颈鹿
r12 = {"蹄类动物", "黑色条纹"}  # 斑马
r13 = {"鸟", "长脖子", "长腿", "不会飞", "有黑白二色"}  # 鸵鸟
r14 = {"鸟", "会游泳", "不会飞", "有黑白二色"}  # 企鹅
r15 = {"鸟", "善飞"}  # 信天翁
# 规则库
rules = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15]  # 规则库


# 找到规则
def rule_result(number):
    if number == 1 or number == 2:
        print("由规则%d推出该动物是哺乳动物" % number)
        return "哺乳动物"
    if number == 3 or number == 4:
        print("由规则%d推出该动物是鸟" % number)
        return "鸟"
    if number == 5 or number == 6:
        print("由规则%d推出该动物是肉食动物" % number)
        return "肉食动物"
    if number == 7 or number == 8:
        print("由规则%d推出该动物是蹄类动物" % number)
        return "蹄类动物"
    if number == 9:
        print("由规则%d推出该动物是金钱豹" % number)
        return "金钱豹"
    if number == 10:
        print("由规则%d推出该动物是老虎" % number)
        return "老虎"
    if number == 11:
        print("由规则%d推出该动物是长颈鹿" % number)
        return "长颈鹿"
    if number == 12:
        print("由规则%d推出该动物是斑马" % number)
        return "斑马"
    if number == 13:
        print("由规则%d推出该动物是鸵鸟" % number)
        return "鸵鸟"
    if number == 14:
        print("由规则%d推出该动物是企鹅" % number)
        return "企鹅"
    if number == 15:
        print("由规则%d推出该动物是信天翁" % number)
        return "信天翁"


if __name__ == '__main__':
    # 提示信息
    print("欢迎使用动物识别系统！")
    print("")
    print("本识别系统可识别的动物种类：")
    for info in Animals:
        print(info, end=' ')
    print("")
    print("")
    print("可选择提供的已知信息：")
    for info in Features:
        print(info, end=' ')
    print("")
    print("")

    # 用户输入已知特征
    temp = input("请输入已知事实：（用空格将各个特征隔开）")
    known = set(temp.split(" "))  # 综合数据库

    # 循环搜索是否有匹配的规则并进行推理
    flag = 1
    j = 0
    new_fact = ""
    while flag:
        flag = 0
        for k in range(0, 14):
            if set(rules[k]) & known == set(rules[k]):
                flag = 1  # 本次找到了一条符合的规则，执行完本规则后需再次搜索是否有其他规则
                rules[k] = "Matched"  # 从规则库中删除已匹配过的规则
                j = k + 1  # 列表由0开始计数，规则由1开始计数
                new_fact = rule_result(j)
                known.add(new_fact)
                # 终止条件:如果使用了规则R9~R15说明已经得到了具体的动物种类
                if j >= 9:
                    break
    if new_fact == "":
        print("信息严重不足或错误，无法进行任何推理！")
    elif j < 9:
        print("信息不足，无法准确识别，只能得出动物是" + new_fact)
    else:
        print("最终识别出来的动物是" + new_fact)
