"""
=========================================================
decision tree
主要分为：
1. 特征选择
2. 决策树的生成
3. 决策树的减枝
=========================================================
"""


# results是一个字典，r相当于"键"，results[r]相当于"值"，对y可能的取值出现的个数进行计数
def uniquecounts(rows):
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results


def entropy(rows):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)
    # 开始计算熵的值
    ent = 0.0
    for r in results.keys():
        p = float(results[r] / len(rows))
        ent = ent - p * log2(p)
    return ent


# 定义节点的属性
class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col  # col 待检验的判断条件对应的列索引值
        self.value = value  # value 为了使结果为True，当前列必须匹配的值
        self.results = results  # 保存当前结果的分支，必须是一个字典，（键-值对，key-value）
        self.tb = tb  # 对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fb = fb  # 对应于结果为false时，树上相对于当前节点的子树上的节点


# 基尼指数,近似替代分类误差的概率
# CART算法中，对于分类树，采用基尼指数最小化准则，基尼指数越小，结果越好
# CART决策树是一个二叉树，递归地二分每个特征
def ginnimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        # 样本点属于第k1类别的概率为p1
        p1 = float(counts[k1]) / total
        imp += p1 * (1 - p1)
    return imp


# 通过切分变量（spliting value) 对数据集拆分
# R1(j,s)={x|x<s}, R2(j,s)={x|x>=s}
# 或者通过 =，！=来进行切分
def divideset(rows, column, value):
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    # 使用列表生成器创建list并返回满足拆分条件的set1，和不满足拆分条件的set2
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


# 以递归的方式构造树
def buildtree(rows, scoref=entropy):
    if len(rows) == 0: return decisionnode()
    current_score = scoref(rows)
    # gain代表信息增益，criteria 定义最佳拆分条件,切分点
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1
    # 选择信息增益最大的特征作为结点特征，
    for col in range(0, column_count):
        # 在当前列中生成一个由不同值构成的字典
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1  # 初始化
        # 根据这一列中的每个值，对数据集进行拆分
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)
            # 信息增益=经验熵-条件熵
            p = float(len(set1) / len(rows))
            gain = current_score - (p * scoref(set1) + (1 - p) * scoref(set2))
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # 由该特征创建子分支
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        # 在当前子分支下递归调用
        return decisionnode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


# 决策树的显示
def printtree(tree, indent=''):
    if tree.results != None:
        print(str(tree.results))
    else:
        # 列表索引值和当前列匹配值
        print(str(tree.col) + ":" + str(tree.value) + "?")
        print(indent + "T->")
        print(tree.tb, indent + " ")
        print(indent + "F->")
        print(tree.fb, indent + " ")


# 对观测数据进行分类
def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


# 决策树的减枝
def prune(tree, mingain):
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)
    if tree.tb.results != None and tree.fb.results != None:
        # 构造合并后的数据集
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c
        delta = entropy(tb + fb) - ((entropy(tb) + entropy(fb)) / 2)
        if delta < mingain:
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)


# 测试集测试
my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]

divideset(my_data, 2, 'yes')
ginnimpurity(my_data)
tree = buildtree(my_data)
printtree(tree=tree)
print(printtree)
print(classify(['(direct)', 'USA', 'yes', 5], tree))
