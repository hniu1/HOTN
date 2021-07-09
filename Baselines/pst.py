# encoding:utf-8
import re
from graphviz import Digraph
import pickle
import string

'''构建概率后缀树
'''
TREE_PIC = Digraph(comment='The Probabilitic Suffix Tree')
TOTAL_SEQUENCE = ''


class TreeNode(object):
    def __init__(self):
        self.count = 1  # 统计此结点代表的字符串出现的次数
        self.name = 'root'  # 标记node的名称
        self.children = {}
        self.totalchild = 0  # 统计node子节点个数
        self.probability_vector = {}  # 转移概率向量
        self.pre_pv = {}  # 上一节点的概率向量s
        self.pre_node = None  # 上一节点


class Tree(object):
    def __init__(self):
        self.root = TreeNode()
        self.P_min = 0.00  # 入树概率阈值
        self.gamma = 0.01  # 节点转移概率阈值
        self.alpha = 0.01  # 另一个阈值 在判断候选节点时起作用
        self.L = 5  # 树最大深度L，即L阶PST
        self.current_deepth = 1  # 当前树深度
        self.pre_node_len = 0  # 判断叶节点是否向上回溯，该值为上一次处理序列的长度

    def build(self, sequence):
        print("------------------ sequence:", sequence, " ------------------")
        seq_list = list(sequence)
        dis_seq_list = list(set(seq_list))
        dis_seq_list.sort(key=list(sequence).index)
        dis_seq_list.remove('$')
        print('获取到有序去重字符集:', dis_seq_list)

        '''
        add_pst为递归函数，该函数的主要作用是递归构造PST
        需要传入当前node节点，总序列sequence,以及候选节点list：candidate_list
        '''

        def add_pst(node, sequence, candidate_list):
            if len(candidate_list) == 0:
                # 如果是根节点传入，将候选节点设置为去重后的字符集
                candidate_list = dis_seq_list
                # 计算根节点概率向量，保留三位小数
                for candidate_p in dis_seq_list:
                    node.probability_vector[candidate_p] = round(compute_pro(candidate_p, sequence), 3)
            else:
                # total_count为查找到的、以候选字符串作为后缀的字符串的数量
                total_count = 0
                for single_tag in dis_seq_list:
                    find_str = node.name + single_tag
                    total_count += find_str_count(find_str, TOTAL_SEQUENCE)
                for single_tag in dis_seq_list:
                    # 计算概率向量，保留三位小数
                    if total_count == 0:
                        node.probability_vector[single_tag] = 0
                    else:
                        node.probability_vector[single_tag] = round(float(
                            find_str_count(node.name + single_tag, TOTAL_SEQUENCE)) / float(
                            total_count), 3)

            for candidate_r in candidate_list:
                # 遍历候选字符列表
                if candidate_r in dis_seq_list:
                    print("候选字符在去重列表中，表示又一次从根节点开始,树深需要置为1,同时node恢复为root")
                    self.current_deepth = 1
                    node = self.root

                if compute_pro(candidate_r, sequence) > self.gamma:
                    # 此时candidate_r为出现概率大于P_min的候选节点,符合入树条件 TODO:对于严格意义上的PST,需要和gamma、alpha进行比较
                    # 切换支点
                    print("上次处理：", self.pre_node_len, "本次长度:", len(candidate_r))
                    if node is not self.root and (self.pre_node_len >= len(candidate_r)):
                        # 如果上一次保存候选序列的长度大于该序列长度，说明此时节点需要向上回溯
                        node = node.pre_node
                    self.pre_node_len = len(candidate_r)
                    print("********符合条件:", candidate_r, "树深度:", len(candidate_r), "节点名：", node.name)
                    child = TreeNode()
                    child.pre_node = node
                    child.name = candidate_r
                    node.children[candidate_r] = child
                    node = child
                    q = [] # extended nodes name list
                    for x in dis_seq_list:
                        # 给每一个候选节点增加前缀，增加的前缀为去重字符集dis_seq_list的遍历
                        str_x_r = ''
                        str_x_r = x + candidate_r
                        q.append(str_x_r)
                    # 此时获得的q，是通过合法候选字符（P>P_min）构造好的前缀数组
                    print("当前字符：", candidate_r, " 候选序列：", q)
                    if len(candidate_r) < self.L:  # if self.current_deepth < self.L:
                        # 树在本分支上的深度自增，步长为1；但此时用的是字符集长度计算 todo:如果表示一次行为的符号不是一个字母或符号，需要review
                        # self.current_deepth += 1
                        # print("当前node：", node.name)
                        # print("probability_vector:", node.pre_node.probability_vector)
                        add_pst(node, sequence, q)
                    else:
                        # 不再向下深入，需要填充该节点概率向量
                        # total_count为查找到的、以候选字符串作为后缀的字符串的数量
                        total_count = 0
                        for single_tag in dis_seq_list:
                            find_str = node.name + single_tag
                            total_count += find_str_count(find_str, TOTAL_SEQUENCE)
                        for single_tag in dis_seq_list:
                            # 计算概率向量，保留三位小数
                            if total_count == 0:
                                node.probability_vector[single_tag] = 0
                            else:
                                node.probability_vector[single_tag] = round(float(
                                    find_str_count(node.name + single_tag, TOTAL_SEQUENCE)) / float(
                                    total_count), 3)

        root_node = self.root
        add_pst(root_node, sequence, [])


def compute_pro(s, total_sequence):
    '''计算s在total_sequence中出现的概率'''
    return float(total_sequence.count(s)) / float(len(total_sequence.replace('$', '')))


def compute_list_pro(s, sequence_list):
    '''计算s在sequence_list出现的概率'''
    return float(sequence_list.count(s)) / float(len(sequence_list))


def gen_tree(input_file):
    '''生成PST'''
    tree = Tree()
    global TOTAL_SEQUENCE
    global RawTrajectories
    RawTrajectories = ReadSequentialData(input_file)
    for T in RawTrajectories:
        t = ''.join(T[1]) + '$'
        TOTAL_SEQUENCE += t
    tree.build(TOTAL_SEQUENCE)
    # with open(input_file) as f:
    #     for line in f:
    #         # 增加'$'用来区别是否是完整后缀  todo:PST是否需要？
    #         line = line.strip() + '$'
    #         TOTAL_SEQUENCE += line
    #     tree.build(TOTAL_SEQUENCE)
    return tree

def ReadSequentialData(InputFileName):
    MinimumLengthForTraining = 1
    LastStepsHoldOutForTesting = 0
    RawTrajectories = []
    with open(InputFileName, 'rb') as f:
        list_raw = pickle.load(f)
    LoopCounter = 0
    alphabet_string = string.ascii_lowercase
    alphabet_list = list(alphabet_string)
    for data in list_raw:
        ## [Ship1] [(Port1, time)] [(Port2, time)] [(Port3, time)]...
        ship = data[0]
        movements = [alphabet_list[int(node[0])] for node in data[1:]]
        timestamp = [node[1] for node in data[1:]]

        LoopCounter += 1
        if LoopCounter % 10000 == 0:
            print(LoopCounter)
        ## Test for movement length
        MinMovementLength = MinimumLengthForTraining + LastStepsHoldOutForTesting
        if len(movements) < MinMovementLength:
            continue

        RawTrajectories.append([ship, movements, timestamp])

    return RawTrajectories

def find_str_count(x, total_str):
    start = 0
    count = 0
    while True:
        index = total_str.find(x, start)
        # if search string not found, find() returns -1
        # search is complete, break out of the while loop
        if index == -1:
            break
        # move to next possible start position
        start = index + 1
        count += 1
    return count


def draw_pst(tree):
    '''绘制PST'''

    def draw_node(node):
        TREE_PIC.node(node.name, node.name + '\n' + str(node.probability_vector))
        if node.children:
            for key in node.children:
                TREE_PIC.edge(node.name, key, constraint='true')
                draw_node(node.children[key])

    draw_node(tree.root)
    TREE_PIC.render('pst-output/PST.gv', view=True)

def prune_tree(tree):
    def prune(node):
        for act in node.probability_vector:
            if node.probability_vector[act] < tree.gamma:
                node.probability_vector[act] = 0

    def prune_node(node):
        if node.children:
            for key in node.children:
                prune(node.children[key])
                prune_node(node.children[key])

    prune_node(tree.root)
    return tree

def calculate_similarity(tree):
    def locate_node(pre_s, node):
        try:
            for ix in range(len(pre_s)):
                node = node.children[''.join(pre_s[-ix - 1:])]
        except:
            pre_s = pre_s[1:]
            node = 'NotExist'
        return node, pre_s

    def SIM(t,tree):
        Pt = 1.0
        P = 1.0
        for i in range(len(t)):
            pre_s = ''.join(t[:i])
            s = t[i]
            if not pre_s:
                pt = tree.root.probability_vector[s]
            else:
                node = tree.root
                node, pre_s = locate_node(pre_s, node)
                while node == 'NotExist':
                    node = tree.root
                    node, pre_s = locate_node(pre_s, node)
                pt = node.probability_vector[s]
            Pt = Pt * pt
            p = tree.root.probability_vector[s]
            P = P * p
        sim = Pt/P
        return sim

    list_sim = []
    for T in RawTrajectories:
        t = T[1]
        sim = SIM(t, tree)
        list_sim.append(sim)

    print('check')


if __name__ == '__main__':
    txt = 'pst_data.txt'
    pkl = 'pst_result.pkl'
    InputFolder = '../../data_preprocessed/tensor/Synthetic_1000/'
    InputFileName = InputFolder + 'tensor_seq.pkl'
    tree = gen_tree(InputFileName)
    print("------------------------------------------------------------------")
    #print(tree.root.children['t'].children['ct'].children['act'].probability_vector)
    print("------------------------------------------------------------------")
    # draw_pst(tree)
    # tree = prune_tree(tree)
    calculate_similarity(tree)
