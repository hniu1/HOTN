'''
1. split normal and abnormal data
2. build networks for them separately
3. draw sankey diagram or network for them
'''

import json
import os
import re
import pandas as pd
from graphviz import Digraph
from datetime import datetime, timedelta

def SplitData():
    # split normal and abnormal data
    list_outid = []
    with open(path_data + 'outliers.csv', 'r') as of:
        for line in of:
            id = line.split(' ')[0]
            list_outid.append(id)
    with open(path_data + 'normal.csv', 'w') as nf:
        with open(path_data + 'Rad_seq.csv', 'r') as rf:
            for line in rf:
                id = line.split(' ')[0]
                if id not in list_outid:
                    nf.write(line)
    print('check')

def VisualNetwork(dict_tran, SaveName):
    g = Digraph('G', filename= SaveName, graph_attr={'rankdir':'LR'})
    # g.graph_attr['rankdir'] = 'LR'
    # g.node('Start', shape='Mdiamond', style='filled', color='green')
    g.node('END(2)', shape='Msquare', style='filled', color='red')
    for tran in dict_tran:
        tp = dict_tran[tran]
        action_0 = tran.split('-->')[0]
        action_1 = tran.split('-->')[1]
        if tp > 0.3:
            g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(tp*5),
                   color='blue')
        else:
            g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(tp*5),
                   style="dashed", color='red')
    g.view()
    print('DFA')

def ReadNetwork(ntkfile):
    dict_ntk = {} # key: id, value: [action, action]
    with open(ntkfile, 'r') as nf:
        for line in nf:
            source = line.split(',')[0].split('|')[0]
            target = line.split(',')[1].split('|')[0]
            tp = float(line.split('\n')[0].split(',')[-1])
            source_action = source
            target_action = target
            tran = source_action + '-->' +target_action
            dict_ntk[tran] = tp
    return dict_ntk

def ViewNetwork():
    # dict_normal = {} # key: tran, value: tp
    # dict_outliers = {} # key: id, value: [action, action]
    dict_actid = {} # key: action id, value: action name

    df_actid = pd.read_csv(path_data + 'RadActionID.csv')
    for index, row in df_actid.iterrows():
        id = row['action_id']
        act = row['action']
        dict_actid[id] = act

    dict_normal = ReadNetwork(path_data + 'network-normal.csv', dict_actid)

    VisualNetwork(dict_normal, path_results + 'NormalNetwork')

    print('view network finished!!!')

def ReadTensorNetwork(ntkfile):
    dict_ntk = {} # key: id, value: [action, action]
    with open(ntkfile, 'r') as nf:
        for line in nf:
            # source = line.split(',')[0].split('|')[0]
            # target = line.split(',')[1].split('|')[0]
            source = line.split(',')[0]
            target = line.split(',')[1]
            source_action = ReadHONode(source)
            target_action = ReadHONode(target)
            tp = float(line.split('\n')[0].split(',')[-2])
            duration = line.split('\n')[0].split(',')[-1]
            # source_action = source
            # target_action = target
            tran = source_action + '-->' +target_action
            dict_ntk[tran] = (tp,duration)
    return dict_ntk

def ParseTimedelta(s):
    if 'day' in s:
        m = re.match(r'(?P<days>[-\d]+) day[s]* (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    else:
        m = re.match(r'(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    t = {key: float(val) for key, val in m.groupdict().items()}
    return timedelta(**t)

def ReadTensorNetworkFreq(ntkfile):
    dict_ntk = {} # key: id, value: [action, action]
    with open(ntkfile, 'r') as nf:
        for line in nf:
            source = line.split(',')[0]
            target = line.split(',')[1]
            source_action = ReadHONode(source)
            target_action = ReadHONode(target)
            tp = int(line.split('\n')[0].split(',')[-2])
            duration = line.split('\n')[0].split(',')[-1]
            try:
                duration_ave = ParseTimedelta(duration)/tp
                duration_ave = str(duration_ave).replace(',', '')
            except:
                continue
            tran = source_action + '-->' +target_action
            dict_ntk[tran] = (tp,duration_ave)
    return dict_ntk

def VisualTensorNetwork(dict_tran, SaveName):
    g = Digraph('G', filename= SaveName, graph_attr={'rankdir':'LR'})
    # g.graph_attr['rankdir'] = 'LR'
    g.node('Created', shape='Mdiamond', style='filled', color='green')
    g.node('END', shape='Msquare', style='filled', color='red')
    for tran in dict_tran:
        (tp, duration) = dict_tran[tran]
        action_0 = tran.split('-->')[0]
        action_1 = tran.split('-->')[1]
        if tp > 0.3:
            g.edge(action_0, action_1, str(round(tp,3))+', '+duration, penwidth= str(tp/100),
                   color='blue')
        else:
            g.edge(action_0, action_1, str(round(tp,3))+', '+duration, penwidth= str(tp/100),
                   color='red')
    g.view()
    print('DFA')

def ViewTensorNetwork(network_file, graph_file):

    dict_normal = ReadTensorNetwork(path_data + network_file)

    VisualTensorNetwork(dict_normal, path_visual + graph_file)

    print('view network finished!!!')

def ViewTensorNetworkFreq(network_file, graph_file):

    dict_normal = ReadTensorNetworkFreq(path_data + network_file)

    VisualTensorNetwork(dict_normal, path_visual + graph_file)

    print('view network finished!!!')

def ReadHONode(hon):
    root = hon.split('|')[0]
    # node = dict_actid[int(root)]
    node = root
    list_prenodes = hon.split('|')[1].split('.')
    if not list_prenodes==['']:
        node += '|'
        for ix, prenode in enumerate(list_prenodes):
            # node += dict_actid[int(prenode)]
            node += prenode

            if ix < len(list_prenodes)-1:
                node += '.'
    # node += '(' + hon + ')'
    return node

def ReadHONetwork(ntkfile):
    dict_ntk = {} # key: id, value: [action, action]
    with open(ntkfile, 'r') as nf:
        for line in nf:
            source = line.split(',')[0]
            target = line.split(',')[1]
            tp = float(line.split('\n')[0].split(',')[-1])
            source_action = ReadHONode(source)
            target_action = ReadHONode(target)
            tran = source_action + '-->' +target_action
            dict_ntk[tran] = tp
    return dict_ntk

def VisualHONetwork(dict_tran, list_abtrans, SaveName):
    g = Digraph('G', filename= SaveName, graph_attr={'rankdir':'LR'})
    # g.graph_attr['rankdir'] = 'LR'
    # g.node('Start', shape='Mdiamond', style='filled', color='green')
    # g.node('END(2)', shape='Msquare', style='filled', color='red')
    for tran in dict_tran:
        tp = dict_tran[tran]
        action_0 = tran.split('-->')[0]
        action_1 = tran.split('-->')[1]
        if action_0 == 'START*DRUG':
            g.node('START*DRUG', shape='Mdiamond', style='filled', color='green')
        if action_1 == 'END':
            g.node('END', shape='Msquare', style='filled', color='red')
        if tran not in list_abtrans:
            g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(tp*5),
                   color='blue')
        else:
            g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(tp*5),
                   style="dashed", color='red')
    g.view()
    print('DFA')

def ReadAbTrans():
    # read abnormal transitions
    list_abtrans = []
    with open (path_abtrans + 'AbTrans.json') as json_file:
        dict_abtrans = json.load(json_file)
    for id in dict_abtrans:
        list_trans = dict_abtrans[id]
        for list_nodes in list_trans:
            [source, target] = list_nodes
            source_action = ReadHONode(source)
            target_action = ReadHONode(target)
            tran = source_action + '-->' +target_action
            list_abtrans.append(tran)
    return list_abtrans

def ViewHON():

    list_abtrans = ReadAbTrans()

    dict_ntk = ReadHONetwork(path_data + 'network-ehr.csv')

    VisualHONetwork(dict_ntk, list_abtrans, path_visual + 'HON')

    print('view network finished!!!')

def VisualHONetworkColors(dict_tran, SaveName):
    g = Digraph('G', filename= SaveName, graph_attr={'rankdir':'LR'})
    # g.graph_attr['rankdir'] = 'LR'
    # g.node('START*DRUG', shape='Mdiamond', style='filled', color='green')
    # g.node('END', shape='Msquare', style='filled', color='red')
    for tran in dict_tran:
        tp = dict_tran[tran]
        action_0 = tran.split('-->')[0]
        action_1 = tran.split('-->')[1]
        nodeorder = []
        for node in [action_0, action_1]:
            list_Nodes = node.split('|')
            if len(list_Nodes) == 1:
                order = 1
            else:
                list_PreNode = list_Nodes[1].split('.')
                order = len(list_PreNode) + 1
            nodeorder.append(order)
        path_order = max(nodeorder)
        if path_order == 1:
            g.edge(action_0, action_1, str(round(tp, 3)), penwidth=str(1),
                   color='black')
        elif path_order == 2:
            g.edge(action_0, action_1, str(round(tp, 3)), penwidth=str(1),
                   color='blue')
        elif path_order == 3:
            g.edge(action_0, action_1, str(round(tp, 3)), penwidth=str(1),
                   color='red')
        else:
            g.edge(action_0, action_1, str(round(tp, 3)), penwidth=str(1),
                   color='forestgreen')
        # if tran not in list_abtrans:
        #     g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(tp*5),
        #            color='blue')
        # else:
        #     g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(tp*5),
        #            style="dashed", color='red')
    g.view()
    print('DFA')

def ViewHONColors():

    # list_abtrans = ReadAbTrans()

    dict_ntk = ReadHONetwork(path_data + 'network-ehr-freq.csv')

    VisualHONetworkColors(dict_ntk, path_visual + 'HON_colors')

    print('view network finished!!!')

def ViewEachDependencies():
    # view each dependencies separately from 1 to highest
    list_abtrans = ReadAbTrans()
    dict_ntk = ReadHONetwork(path_data + 'network-ehr.csv')
    # VisualEachHON(dict_ntk)

    dict_path_order = {} # key: the order dependency num, value: [{path:tp}]

    for path in dict_ntk:
        source,target = path.split('-->')
        nodeorder = []
        for node in [source, target]:
            list_Nodes = node.split('|')
            if len(list_Nodes) == 1:
                order = 1
            else:
                list_PreNode = list_Nodes[1].split('.')
                order = len(list_PreNode) + 1
            nodeorder.append(order)
        path_order = max(nodeorder)
        if path_order not in dict_path_order.keys():
            dict_path_order[path_order] = {}
        dict_path_order[path_order][path] = dict_ntk[path]
    list_order = list(dict_path_order.keys())
    list_order.sort()
    for order in list_order:
        dict_path = dict_path_order[order]
        VisualHONetwork(dict_path, list_abtrans, path_visual + 'order_' + str(order))
    print('')

def ReadAbTransASFON(abfile):
    # read abnormal transitions as FON
    list_abtrans = []
    with open (abfile) as json_file:
        dict_abtrans = json.load(json_file)
    for id in dict_abtrans:
        list_trans = dict_abtrans[id]
        for list_nodes in list_trans:
            [source, target] = list_nodes
            source_action = source.split('|')[0]
            target_action = target.split('|')[0]
            tran = source_action + '-->' +target_action
            list_abtrans.append(tran)
    return list_abtrans

def ReadHONetworkAsFON(ntkfile, dict_actid):
    dict_ntk = {} # key: id, value: [action, action]
    with open(ntkfile, 'r') as nf:
        for line in nf:
            source = line.split(',')[0]
            target = line.split(',')[1]
            tp = float(line.split('\n')[0].split(',')[-1])
            source_action = dict_actid[int(source.split('|')[0])] + '(' + source.split('|')[0] + ')'
            target_action = dict_actid[int(target.split('|')[0])] + '(' + target.split('|')[0] + ')'
            tran = source_action + '-->' +target_action
            dict_ntk[tran] = tp
    return dict_ntk

def VisualFONetwork(dict_tran, list_abtrans, SaveName):
    g = Digraph('G', filename= SaveName, graph_attr={'rankdir':'LR'})
    # g = Digraph('G', filename= SaveName)
    # g.graph_attr['rankdir'] = 'LR'
    g.node('START*DRUG', shape='Mdiamond', style='filled', color='green')
    g.node('END', shape='Msquare', style='filled', color='red')
    for tran in dict_tran:
        tp = dict_tran[tran]
        action_0 = tran.split('-->')[0]
        action_1 = tran.split('-->')[1]
        if tran not in list_abtrans:
            g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(1.5),
                   color='black')
        else:
            g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(1.5),
                color='red')
    g.view()
    print('DFA')

def ViewFON():

    list_abtrans = ReadAbTransASFON(path_results_HON + 'AbTrans/AbTrans.json')

    dict_ntk = ReadNetwork(path_results_FON + 'data/network-ehr.csv')

    VisualFONetwork(dict_ntk, list_abtrans, path_visual + 'HON_Outliers')

    print('view network finished!!!')

def Visual2Networks(dict_tran, list_abtrans_FON, list_abtrans_HON, SaveName):
    union_outlier = set(list_abtrans_FON).union(list_abtrans_HON)
    common_outlier = set(list_abtrans_FON).intersection(list_abtrans_HON)
    uniqueFON_outlier = set(list_abtrans_FON).difference(list_abtrans_HON)
    uniqueHON_outlier = set(list_abtrans_HON).difference(list_abtrans_FON)

    list_nomal = ['Ready(6)-->Reserved(7)',
                   'Reserved(7)-->InProgress(5)',
                   'InProgress(5)-->Completed(0)',
                   'Completed(0)-->END(2)']

    g = Digraph('G', filename= SaveName, graph_attr={'rankdir':'LR'})
    # g = Digraph('G', filename= SaveName)
    # g.graph_attr['rankdir'] = 'LR'
    # g.node('Start', shape='Mdiamond', style='filled', color='green')
    g.node('END(2)', shape='Msquare', style='filled', color='red')
    # g.node('Created(1)', style='filled', color='green')
    # g.node('Exited(3)', style='filled', color='orange')
    # g.node('Failed(4)', style='filled', color='orange')
    # g.node('InProgress(5)', style='filled', color='green')
    # g.node('Ready(6)', style='filled', color='green')
    # g.node('Reserved(7)', style='filled', color='green')
    # g.node('Completed(0)', style='filled', color='green')

    for tran in dict_tran:
        tp = dict_tran[tran]
        action_0 = tran.split('-->')[0]
        action_1 = tran.split('-->')[1]
        if tran not in union_outlier: # normal
            g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(tp*5),
                   color='black')
                   # color='grey')
        elif tran in uniqueFON_outlier: # only fon detected
            if tran == 'Created(1)-->Completed(0)':
                g.edge(action_0, action_1, str(round(tp, 3)), penwidth=str(1.5),
                       color='blue')
                       # color='grey')
            else:
                g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(2),
                           color='darkgreen')
                       # color='grey')
        elif tran in uniqueHON_outlier: # only hon detected
            if tran not in list_nomal:
                if tp < 0.2:
                    g.edge(action_0, action_1, str(round(tp, 3)), penwidth=str(2),
                           color='red')
                else:
                    g.edge(action_0, action_1, str(round(tp,3)), penwidth= str(tp*10),
                           color='red')
                        # color='grey')
            else:
                g.edge(action_0, action_1, str(round(tp, 3)), penwidth=str(tp * 5),
                       color='black')
                       # color='grey')
        else: # detected by both
            g.edge(action_0, action_1, str(round(tp, 3)), penwidth=str(1.5),
                   color='blue')
                   # color='grey')
    g.view()
    print('DFA')

def ViewTwoNtk():
    dict_actid = {}  # key: action id, value: action name

    df_actid = pd.read_csv(path_data + 'RadActionID.csv')
    for index, row in df_actid.iterrows():
        id = row['action_id']
        act = row['action']
        dict_actid[id] = act

    list_abtrans_FON = ReadAbTransASFON(path_results_FON + 'AbTrans/AbTrans.json')
    list_abtrans_HON = ReadAbTransASFON(path_results_HON + 'AbTrans/AbTrans.json')

    dict_ntk = ReadNetwork(path_results_FON + 'data/network-ehr.csv', dict_actid)

    Visual2Networks(dict_ntk, list_abtrans_FON, list_abtrans_HON, path_visual + 'con_abnormal')

    print('view network finished!!!')


if __name__ == '__main__':
    path_results_FON = '../../results/hon_tensor/OASIS/ConShortEndFON_1k/'
    path_results_HON = '../../results/hon_tensor/OASIS/ConShortEnd_1k/'

    path_results = path_results_FON
    path_data = path_results + 'data/'
    path_visual = path_results + 'visual/'
    path_abtrans = path_results + 'AbTrans/'

    os.makedirs(path_visual, exist_ok=True)

    # SplitData()
    # ViewNetwork()
    # ViewHON()
    # ViewFON()
    # ViewTwoNtk()
    # ViewEachDependencies()
    # ViewHONColors()
    ViewTensorNetworkFreq('network-freq.csv', 'TensorNetworkFreq')
    # ViewTensorNetwork('network.csv', 'TensorNetwork')


    print('test finished!!!')

