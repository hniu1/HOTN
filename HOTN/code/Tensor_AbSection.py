#coding=utf-8
'''
find the abnormal section for anomalies
update:
1. calculating the edge difference instead of only consider the unique edge
2. recommend the target for the absection
'''
import os
import numpy as np
import copy
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
import pickle
from datetime import datetime, timedelta
import re
import sys
sys.path.insert(0,'../../utils/')
from file_load_save import load_json_dict, save_json_dict, load_csv_dict

# find out outlier
def ReadOutlier(dir):
    list_outlier_id = []
    list_outlier_data = []
    with open(path_ADHD + dir + 'data/outlier.csv') as f:
        for line in f:
            order_id = line.split(',')[0]
            lables = line.split('\n')[0].split(',')[1:]
            list_outlier_id.append([order_id, int(lables[0]), int(lables[1])])
            # list_outlier_data.append(data)
    return list_outlier_id

# comparing the results from EHR only and EHR inject 10. find out the union, overlap and difference
def CompareResults(dir0, dir1):
    list_outlier_0 = [] # outlier in EHR only
    list_outlier_1 = [] # outlier in EHR with 10 inject
    for dir in [dir0,dir1]:
        # with open(path_ADHD + dir + 'data/outlier.csv') as f:
        with open(dir + 'data/outlier.csv') as f:
            for line in f:
                order_id = line.split()[0]
                if int(order_id) <= 1000:
                    if dir == dir0:
                        list_outlier_0.append(order_id)
                    else:
                        list_outlier_1.append(order_id)
    union_outlier = set(list_outlier_0).union(list_outlier_1)
    overlap_outlier = set(list_outlier_0).intersection(list_outlier_1)
    outlier_In0not1 = set(list_outlier_0).difference(list_outlier_1)
    outlier_In1not0 = set(list_outlier_1).difference(list_outlier_0)
    print('the number of the union of outliers are ', len(union_outlier))
    print('the number of the intersection of outliers are ', len(overlap_outlier))
    print('the number of the outlier in {} not in {} is {}'.format(dir0, dir1, len(outlier_In0not1)))
    print('the number of the outlier in {} not in {} is {}'.format(dir1, dir0, len(outlier_In1not0)))
    print("compare results finished!")
    return list_outlier_0, overlap_outlier, outlier_In0not1

def ReadSequentialData(InputFileName):
    RawTrajectories = []
    with open(InputFileName, 'rb') as f:
        list_raw = pickle.load(f)
    LoopCounter = 0
    for data in list_raw:
        ## [Ship1] [(Port1, time)] [(Port2, time)] [(Port3, time)]...
        ship = data[0]
        movements = [node[0] for node in data[1:]]
        timestamp = [node[1] for node in data[1:]]

        LoopCounter += 1
        if LoopCounter % 10000 == 0:
            print(LoopCounter)
        ## Test for movement length
        RawTrajectories.append([ship, movements, timestamp])

    return RawTrajectories

def ParseTimedelta(s):
    if 'day' in s:
        m = re.match(r'(?P<days>[-\d]+) day[s]* (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    else:
        m = re.match(r'(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    t = {key: float(val) for key, val in m.groupdict().items()}
    return timedelta(**t)

# compare the paths for the whole graph and subgraph of anomalies
def FindAbnormalPath(dir):
    InputFileName = path_preprocess + 'tensor_seq.pkl'
    RawTrajectories = ReadSequentialData(InputFileName)
    list_outlier_0 = ReadOutlier(dir)
    list_outlier_data = []
    RawTrajectories = ReadSequentialData(path_results + 'data/tensor_seq.pkl')
    for t in RawTrajectories:
        id = t[0]
        if id in [item[0] for item in list_outlier_0]:
        # if id in list_outlier_0:
            list_outlier_data.append(t[1:])
    path_data = path_ADHD + dir + 'data/'
    path_network = path_ADHD + dir + 'network/'
    # read the whole graph
    GEdges = {}
    GEdgesSet = set()
    with open(path_data + 'network.csv') as f:
        for line in f:
            fields = line.split('\n')[0].split(',')
            FromNode = fields[0]
            ToNode = fields[1]
            weight = float(fields[2].strip())
            duration = ParseTimedelta(fields[3])
            GEdges[(FromNode, ToNode)] = np.array([weight, duration])
            GEdgesSet.add((FromNode, ToNode))
    dict_abnormal_path = {} # key: order_id, value: list of unique path
    dict_RecommendPaths = {} # key: order_id, value: list of recommend paths and possibility
    dict_abnormal_duration = {} # key: order_id, value: [path, abnoraml_duration, recomend_duration]
    for idx, item in enumerate(list_outlier_0):
        order_id, label_0, label_1 = item
        HEdges = {}
        HEdgesSet = set()
        with open(path_network + 'network' + str(int(order_id)-1) + '.csv') as f:
            for line in f:
                fields = line.split('\n')[0].split(',')
                FromNode = fields[0]
                ToNode = fields[1]
                weight = float(fields[2].strip())
                duration = ParseTimedelta(fields[3])
                HEdges[(FromNode, ToNode)] = np.array([weight, duration])
                HEdgesSet.add((FromNode, ToNode))
        AllEdgesSet = GEdgesSet.union(HEdgesSet)
        list_AbPaths, list_AbDuration = EdgeDiff(GEdges, HEdges, AllEdgesSet)
        if label_0 and label_1:
            list_AbDuration = [item for item in list_AbDuration if item not in list_AbPaths] # remove those paths that exist in abpath from abdurtion
            # if len(AbDuration_unique) == 0:
            #     label_1 = 0
        if label_0:
            list_AbExistPaths = CheckAbPathExist(list_AbPaths, list_outlier_data[idx])
            # list_AbExistDuration = CheckAbPathExist(list_AbDuration, list_outlier_data[idx])
            if len(list_AbExistPaths) == 0:
                list_AbExistPaths = PickExistSectionPath(list_AbPaths, list_outlier_data[idx])
            list_LeastPath = CheckLeastPath(list_AbExistPaths)
            list_LeastPath, list_RecEdges = FullTargetRecommend(list_LeastPath, HEdges)
            dict_abnormal_path[order_id] = [item for item in list_LeastPath]
            dict_RecommendPaths[order_id] = list_RecEdges
        if label_1 and (len(list_AbDuration)>0):
            for data in list_AbDuration:
                path = data[0]
                source = path[0].split('|')[0]
                target = path[1].split('|')[0]
                raw_order = RawTrajectories[int(order_id)-1]
                path_index = [i for i, act in enumerate(raw_order[1]) if act == source]
                for index in path_index:
                    if raw_order[1][index+1] == target:
                        source_time = datetime.strptime(raw_order[2][index], '%Y-%m-%d %H:%M:%S')
                        target_time = datetime.strptime(raw_order[2][index+1], '%Y-%m-%d %H:%M:%S')
                        duration_tran = target_time - source_time
                ab_duration = data[1]
                duration = HEdges[path]
                path_raw = (source,target)
                # dict_abnormal_duration[order_id] = [path, ab_duration, duration]
                dict_abnormal_duration[order_id] = [path_raw, duration_tran, duration[1]]

    path_AbTrans = path_results + 'AbTrans/'
    os.makedirs(path_AbTrans, exist_ok=True)
    with open(path_AbTrans + 'AbTrans.csv', 'w') as f:
        f.write('id, Abnormal Tran, Recommend Trans\n')
        for id in dict_abnormal_path:
            for ix, path in enumerate(dict_abnormal_path[id]):
                path_rec = dict_RecommendPaths[id][ix]
                f.write(id + ',' + ' '.join([path[0][0], path[0][1], str(path[1][0]), str(path[1][1]).replace(',', '')]))
                for pr in path_rec:
                    f.write(',' + ' '.join([pr[0][0], pr[0][1], str(pr[1][0]), str(pr[1][1]).replace(',', '')]))
                f.write('\n')
    with open(path_AbTrans + 'AbDuration.csv', 'w') as f:
        f.write('id, Transition, Abnormal Duration, Recommend Duration\n')
        for id in dict_abnormal_duration:
            item = dict_abnormal_duration[id]
            f.write(id + ',' + ' '.join([item[0][0], item[0][1]]) + ','
                    + str(item[1]).replace(',', '')+ ','
                    + str(item[2]).replace(',', '') + '\n')

    # with open(path_AbTrans + 'AbDuration.csv', 'w') as f:
    #     f.write('id, Transition, Abnormal Duration, Recommend Duration\n')
    #     for id in dict_abnormal_duration:
    #         item = dict_abnormal_duration[id]
    #         f.write(id + ',' + ' '.join([item[0][0], item[0][1]]) + ','
    #                 + ' '.join([str(item[1][0]), str(item[1][1]).replace(',', '')])+ ','
    #                 + ' '.join([str(item[2][0]), str(item[2][1]).replace(',', '')]) + '\n')

    print('find abnormal path finished!!!')
    return dict_abnormal_path, dict_RecommendPaths

# def DurationRecommend(path, HEdges):
#     for item in HEdges:
#         if path == item[0]:
#             duration = item[1]
#             return duration

def FullTargetRecommend(paths,HEdges):
    Recommend=[]
    list_AbPath = []
    for path in paths:
        SourceNodes = path[0][0]
        AbSource = NetworkNode2Path(path[0][0])
        AbTarget = NetworkNode2Path(path[0][1])
        AbPath = Connect2Path(AbSource, AbTarget)
        RecEdges = RecommendPath(path,SourceNodes,HEdges)
        sorted_RecEdges = sorted(RecEdges, key=lambda kv: kv[1][0], reverse=True)
        list_RecEdges = [] # RecEdge is not same with AbPath
        for RecEdge in sorted_RecEdges:
            edge = RecEdge[0]
            RecSource = NetworkNode2Path(edge[0])
            RecTarget = NetworkNode2Path(edge[1])
            RecPath = Connect2Path(RecSource,RecTarget)
            if RecPath != AbPath:
                list_RecEdges.append(RecEdge)
        if len(list_RecEdges) != 0:
            list_AbPath.append(path)
            Recommend.append(list_RecEdges)
    return list_AbPath, Recommend

def Connect2Path(SourcePath, TargetPath):
    if TargetPath[:-1] == SourcePath[-1 - len(TargetPath[:-2]):]:
        Path = SourcePath + [TargetPath[-1]]
    else:
        Path = SourcePath + TargetPath
    return Path
# def RecommendPath(path,SourceNodes,HEdges):
#     RecEdges = {}
#     for nodes in HEdges:
#         if SourceNodes == nodes[0]:
#             if HEdges[nodes] > 0.2:
#                 RecEdges[nodes] = HEdges[nodes]
#     RecEdges = CheckRecPath(path, RecEdges, HEdges)
#     return RecEdges
def RecommendPath(path,SourceNodes,HEdges):
    list_RecEdges = []
    for nodes in HEdges:
        if SourceNodes == nodes[0]:
            if HEdges[nodes][0] > 0.2:
                weight = HEdges[nodes]
                list_Path = [nodes,weight]
                list_RecEdges.append(list_Path)
    RecEdges = CheckRecPath(path, list_RecEdges, HEdges)
    return RecEdges

def PickExistSectionPath(list_AbPaths, outlier): # if there is no Exist path, pick a exist section from them.
    list_AbExistPath = []
    list_AbExistEdges = []
    list_AbPaths = sorted(list_AbPaths, key=lambda x: (len(NetworkNode2Path(x[0][0])), len(NetworkNode2Path(x[0][1]))))
    for path in list_AbPaths:
        SourceNodes = path[0][0]
        TargetNodes = path[0][1]
        SourcePath = NetworkNode2Path(SourceNodes)
        TargetPath = NetworkNode2Path(TargetNodes)
        # need to remove the overlap between source and target. e.g., (a|b,c|a) path is b,a,a,c . the a need to be removed
        if TargetPath[:-1] == SourcePath[-1 - len(TargetPath[:-2]):]:
            AbPath = SourcePath + [TargetPath[-1]]
        else:
            AbPath = SourcePath + TargetPath
        # check if the source path in the data
        if len(AbPath) == 2:
            continue
        list_subpath = []
        for x in range(2,len(AbPath)):
            for id in range(0, len(AbPath)-x+1):
                list_subpath.append(AbPath[id:id+x])
        for subpath in list_subpath:
            x = len(subpath)
            for id in range(0, len(outlier)-x+1):
                if subpath == outlier[id:id+x]:
                    if subpath not in list_AbExistPath:
                        list_AbExistPath.append(subpath)
    for path in list_AbExistPath:
        for x in range(0,len(path)-1):
            source = path[0:x+1]
            target = path[x+1:]
            SourceNodes = NetworkPath2Node(source)
            TargetNodes = NetworkPath2Node(target)
            list_AbExistEdges.append((SourceNodes,TargetNodes))
    return list_AbExistEdges

def CheckRecPath(path,RecEdges,HEdges):
    # if recommend path has same path with abnormal path, or no recommend path at all
    # like a|b,c, then, recommend b|c as source.
    if RecEdges:
        for edge in RecEdges:
            AbSource = path[0][0]
            RecSource = edge[0][0]
            if AbSource == RecSource: # if source of ab and rec are same.
                AbTarget = path[0][1].split('|')[0]
                RecTarget = edge[0][1].split('|')[0]
                if AbTarget == RecTarget: # if the target in rec path and ab path is same, then re recommend
                    Source = edge[0][0]
                    PreSource = Source.split('|')[-1]
                    PreTarget = Source.split('|')[0] + '|'
                    if PreSource:
                        if len(PreSource.split('.')) == 1:
                            PreSource = PreSource + '|'
                        else:
                            PreSource = PreSource.replace('.', '|', 1)
                        NewPath = (PreSource, PreTarget)
                        PreSourceRecPath = RecommendPath(NewPath, PreSource, HEdges)
                        RecEdges = RecEdges + PreSourceRecPath
    if not RecEdges:
        Source = path[0][0]
        PreSource = Source.split('|')[-1]
        PreTarget = Source.split('|')[0] + '|'
        if PreSource:
            if len(PreSource.split('.')) == 1:
                PreSource = PreSource + '|'
            else:
                PreSource = PreSource.replace('.', '|', 1)
            NewPath = (PreSource,PreTarget)
            PreSourceRecPath = RecommendPath(NewPath,PreSource,HEdges)
            RecEdges = RecEdges + PreSourceRecPath
    return RecEdges

def CheckAbPathExist(paths,outlier): # if path not in the data, remove
    list_ExistAb = []
    for path in paths:
        SourceNodes = path[0][0]
        TargetNodes = path[0][1]
        SourcePath = NetworkNode2Path(SourceNodes)
        TargetPath = NetworkNode2Path(TargetNodes)
        # need to remove the overlap between source and target. e.g., (a|b,c|a) path is b,a,a,c . the a need to be removed
        if TargetPath[:-1] == SourcePath[-1 - len(TargetPath[:-2]):]:
            AbPath = SourcePath + [TargetPath[-1]]
        else:
            # if len(TargetPath) > 1:  # if (a|b,d|e) exists, ignore
            #     print(SourcePath, TargetPath)
            #     continue
            # else:
            AbPath = SourcePath + TargetPath
        # check if the source path in the data
        ExistFlag = False
        for id in range(0, len(outlier[0])-1):
            if AbPath == outlier[0][id:id+len(AbPath)]:
                ExistFlag = True
                break
        if ExistFlag:
            list_ExistAb.append(path)
    return list_ExistAb

def CheckSourceExist(paths,outlier): # if path not in the data, remove
    list_ExistAb = []
    for path in paths:
        SourceNodes = path[0]
        TargetNodes = path[1]
        SourcePath = NetworkNode2Path(SourceNodes)
        TargetPath = NetworkNode2Path(TargetNodes)
        # need to remove the overlap between source and target. e.g., (a|b,c|a) path is b,a,a,c . the a need to be removed
        if TargetPath[:-1] == SourcePath[-1 - len(TargetPath[:-2]):]:
            AbPath = SourcePath + [TargetPath[-1]]
        else:
            # if len(TargetPath) > 1:  # if (a|b,d|e) exists, ignore
            #     print(SourcePath, TargetPath)
            #     continue
            # else:
            AbPath = SourcePath + TargetPath
        # check if the source path in the data
        ExistFlag = False
        for id in range(0, len(outlier)-1):
            if AbPath == outlier[id:id+len(AbPath)]:
                ExistFlag = True
                break
        if ExistFlag:
            list_ExistAb.append(path)
    return list_ExistAb

def TargetRecommend(paths,HEdges):
    Recommend=[]
    for path in paths:
        SourceNodes = path[1][0]
        for id, node in enumerate(reversed(SourceNodes)):
            if id == 0:
                source = SourceNodes[-1] + '|'
            elif id == 1:
                source += SourceNodes[-1-id]
            else:
                source += '.' + SourceNodes[-1-id]
        RecEdges = {}
        for nodes in HEdges:
            if source == nodes[0]:
                if HEdges[nodes] > 0.2:
                    RecEdges[nodes]=HEdges[nodes]
        sorted_RecEdges = sorted(RecEdges.items(), key=lambda kv: kv[1], reverse=True)
        Recommend.append(sorted_RecEdges)
    return Recommend

def EdgeDiff(GEdges, HEdges, AllEdges):
    VerboseDistances = []
    Distances = []
    itime = datetime.now()
    initial_timedelta = itime - itime
    for edge in AllEdges:
        if edge in GEdges:
            GWeight = GEdges[edge]
        else:
            GWeight = np.array([0,initial_timedelta])
        if edge in HEdges:
            HWeight = HEdges[edge]
        else:
            HWeight = np.array([0,initial_timedelta])
        # print(GWeight, HWeight, abs(GWeight - HWeight) / max(GWeight, HWeight))
        # value = abs(GWeight - HWeight) / max(GWeight, HWeight)
        value = np.array([0, 0], dtype='float')
        value[0] = abs(GWeight[0] - HWeight[0]) / max(GWeight[0], HWeight[0])
        value[1] = abs(GWeight[1] - HWeight[1]) / max(GWeight[1], HWeight[1])
        Distances.append(value)
        VerboseDistances.append((value, edge, GWeight, HWeight))
    arr_dis = np.asarray(Distances, dtype=np.float32)
    mean = np.mean(arr_dis, axis=0)
    sd = np.std(arr_dis, axis=0)
    Threshold = mean + 2 * sd
    SortDistances_tp = sorted(VerboseDistances, key=lambda x:x[0][0], reverse=True)
    AbPaths = [[item[1],item[2]] for item in SortDistances_tp if (item[0][0]>Threshold[0]) and (item[2][0]!=0)]
    SortDistances_dur = sorted(VerboseDistances, key=lambda x:x[0][1], reverse=True)
    AbDuration = [[item[1],item[2]] for item in SortDistances_dur if item[0][1]>Threshold[1]]
    # AbDuration_unique = [item for item in AbDuration if item not in AbPaths]

    # print('edge diff finish')
    return AbPaths, AbDuration

def CheckLeastPath(list_edge): # find least order dependency paths for each anomaly
    list_AbEdge = [] # all abnormal edges for one order
    list_edge = sorted(list_edge, key=lambda x: (len(NetworkNode2Path(x[0][0])), len(NetworkNode2Path(x[0][1]))))
    for edge in list_edge:
        (source,target) = edge[0]
        # convert the source from a|b.c to [c,b,a]
        source_path = NetworkNode2Path(source)
        # convert the target from a|b.c to [c,b,a]
        target_path = NetworkNode2Path(target)
        # need to remove the overlap between source and target. e.g., (a|b,c|a) path is b,a,a,c . the a need to be removed
        if target_path[:-1] == source_path[-1-len(target_path[:-2]):]:
            path = source_path + [target_path[-1]]
        else:
            path = source_path + target_path
        list_AbEdge.append((path,edge))
    # remove the higher order path if lower order path exists
    # e.g. (b|a,c|) and (b|a,c|b) then (b|a,c|b) remove.
    # (b|a,c|) and (c|b,a|) will keep both, because source are b and a, they are not same.
    list_SortEdge = copy.deepcopy(list_AbEdge)
    list_SortEdge.sort(key=lambda x: len(x[0][0]))
    for id, EdgeShort in enumerate(list_SortEdge):
        ShortSource = EdgeShort[1][0][0].split('|')[0]
        ShortTarget = EdgeShort[1][0][1].split('|')[0]
        for id_Edge, Edge in enumerate(list_SortEdge[id + 1:]):
            EdgeSource = Edge[1][0][0].split('|')[0]
            EdgeTarget = Edge[1][0][1].split('|')[0]
            if ShortSource==EdgeSource and ShortTarget==EdgeTarget:
                len_Edge = len(Edge[0])
                len_EdgeShort = len(EdgeShort[0])
                for x in range(0, len_Edge - len_EdgeShort + 1):
                    subEdge = Edge[0][x:x + len_EdgeShort]
                    if subEdge == EdgeShort[0]:  # sub edge contains the edge short
                        try:
                            list_SortEdge.remove(Edge)
                        except:
                            print('check')
                        print('delete edge ', Edge, ' since it contains ', EdgeShort)
    list_LeastPath = [x[1] for x in list_SortEdge]
    return list_LeastPath

def LeastPath(list_edge): # find least order dependency paths for each anomaly
    list_AbEdge = [] # all abnormal edges for one order
    for edge in list_edge:
        (source,target) = edge
        # convert the source from a|b.c to [c,b,a]
        source_path = NetworkNode2Path(source)
        # convert the target from a|b.c to [c,b,a]
        target_path = NetworkNode2Path(target)
        # need to remove the overlap between source and target. e.g., (a|b,c|a) path is b,a,a,c . the a need to be removed
        if target_path[:-1] == source_path[-1-len(target_path[:-2]):]:
            path = source_path + [target_path[-1]]
            path_node = (source_path,[target_path[-1]])
        else:
            if len(target_path) > 1: # if (a|b,d|e) exists, ignore
                print(source_path,target_path)
                continue
            else:
                path = source_path + target_path
                path_node = (source_path, target_path)
        list_AbEdge.append((path,path_node))
    # remove the higher order path if lower order path exists
    # e.g. a,b,c and b,c. we will eliminate the path a,b,c since the b,c should be the anomaly
    list_SortEdge = copy.deepcopy(list_AbEdge)
    list_SortEdge.sort(key=lambda x:len(x[0]))
    for id, EdgeShort in enumerate(list_SortEdge):
        for id_Edge, Edge in enumerate(list_SortEdge[id+1:]):
            len_Edge = len(Edge[0])
            len_EdgeShort = len(EdgeShort[0])
            for x in range(0,len_Edge-len_EdgeShort+1):
                subEdge = Edge[0][x:x+len_EdgeShort]
                if subEdge == EdgeShort[0]: # sub edge contains the edge short
                    try:
                        list_SortEdge.remove(Edge)
                    except:
                        print('check')
                    print('delete edge ', Edge, ' since it contains ', EdgeShort)
    list_LeastPath = list_SortEdge
    return list_LeastPath

def NetworkNode2Path(network_node):
    path = []  # convert the source from a|b.c to [c,b,a]
    list_source = network_node.split('|')
    if not list_source[-1]:  # if a|
        path.append(list_source[0])
    else:  # if a|b or a|b.c
        for node in reversed(list_source[-1].split('.')):
            path.append(node)
        path.append(list_source[0])
    return path

def NetworkPath2Node(path):
    for id, act in enumerate(reversed(path)):
        if id == 0:
            node = act + '|'
        elif id == 1:
            node += act
        else:
            node += '.' + act
    return node


if __name__ == '__main__':
    dir = 'Synthetic_1000/'
    path_preprocess = '../../data_preprocessed/tensor/Synthetic_1000/'
    path_ADHD = '../../results/hon_tensor/Synthetic/'
    path_results = path_ADHD + dir

    dict_AbnormalPaths, dict_RecommendPaths = FindAbnormalPath(dir)


    print('test finished!!!')
