

import BuildRulesFastParameterFree
import BuildRulesFastParameterFreeFreq
import BuildNetwork
import itertools
import random
import matplotlib.pyplot as plt
import glob
import copy
import numpy as np
import os
import shutil
import pickle
from datetime import datetime, timedelta
import re

###########################################
# Functions
###########################################

def ReadSequentialData(InputFileName):
    if Verbose:
        print('Reading raw sequential data')
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
            VPrint(LoopCounter)
        ## Test for movement length
        MinMovementLength = MinimumLengthForTraining + LastStepsHoldOutForTesting
        if len(movements) < MinMovementLength:
            continue

        RawTrajectories.append([ship, movements, timestamp])

    return RawTrajectories


def BuildTrainingAndTesting(RawTrajectories):
    VPrint('Building training and testing')
    Training = []
    Testing = []
    for trajectory in RawTrajectories:
        ship, movement = trajectory
        movement = [key for key,grp in itertools.groupby(movement)] # remove adjacent duplications
        if LastStepsHoldOutForTesting > 0:
            Training.append([ship, movement[:-LastStepsHoldOutForTesting]])
            Testing.append([ship, movement[-LastStepsHoldOutForTesting]])
        else:
            Training.append([ship, movement])
    return Training, Testing

def DumpRules(Rules, OutputRulesFile):
    VPrint('Dumping rules to file')
    with open(OutputRulesFile, 'w') as f:
        for Source in Rules:
            for Target in Rules[Source]:
                f.write(' '.join([' '.join([str(x) for x in Source]), '=>', Target, str(Rules[Source][Target][0]), str(Rules[Source][Target][1])]) + '\n')

def DumpNetwork(Network, OutputNetworkFile):
    VPrint('Dumping network to file')
    LineCount = 0
    with open(OutputNetworkFile, 'w') as f:
        for source in Network:
            for target in Network[source]:
                f.write(','.join([SequenceToNode(source), SequenceToNode(target), str(Network[source][target][0]), str(Network[source][target][1]).replace(',', '')]) + '\n')
                LineCount += 1
    VPrint(str(LineCount) + ' lines written.')

def SequenceToNode(seq):
    curr = seq[-1]
    node = curr + '|'
    seq = seq[:-1]
    while len(seq) > 0:
        curr = seq[-1]
        node = node + curr + '.'
        seq = seq[:-1]
    if node[-1] == '.':
        return node[:-1]
    else:
        return node

def VPrint(string):
    if Verbose:
        print(string)


def BuildHON(InputFileName, OutputNetworkFile):
    RawTrajectories = ReadSequentialData(InputFileName)
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    VPrint(len(TrainingTrajectory))
    Rules = BuildRulesFastParameterFree.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    # DumpRules(Rules, OutputRulesFile)
    Network = BuildNetwork.BuildNetwork(Rules)
    DumpNetwork(Network, OutputNetworkFile)
    VPrint('Done: '+InputFileName)

def BuildHONfreq(InputFileName, OutputNetworkFile):
    print('FREQ mode!!!!!!')
    RawTrajectories = ReadSequentialData(InputFileName)
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    VPrint(len(TrainingTrajectory))
    Rules = BuildRulesFastParameterFreeFreq.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    # DumpRules(Rules, OutputRulesFile)
    Network = BuildNetwork.BuildNetwork(Rules)
    DumpNetwork(Network, OutputNetworkFile)
    VPrint('Done: '+InputFileName)

def ParseTimedelta(s):
    if 'day' in s:
        m = re.match(r'(?P<days>[-\d]+) day[s]* (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    else:
        m = re.match(r'(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    t = {key: float(val) for key, val in m.groupdict().items()}
    return timedelta(**t)

def GraphDiff():
    print('calculating graph diff')
    NetworkPath = path_network + '*'
    nw = path_data + 'network.csv'
    fn = glob.glob(NetworkPath)
    # fn = sorted(fn)
    print(fn)
    pairs = [(fn[i], nw) for i in range(len(fn))]
    distances = []
    itime = datetime.now()
    initial_timedelta = itime - itime
    for pair in pairs:
        fileG = pair[0]
        fileH = pair[1]
        # print(fileG)
        distance = np.array([0,0], dtype='float')
        GEdges = {}
        HEdges = {}
        GEdgesSet = set()
        HEdgesSet = set()
        VerboseDistances = []
        with open(fileG) as FG:
            with open(fileH) as FH:
                for line in FG:
                    fields = line.split('\n')[0].split(',')
                    FromNode = fields[0]
                    ToNode = fields[1]
                    weight = float(fields[2].strip())
                    duration = ParseTimedelta(fields[3])
                    GEdges[(FromNode, ToNode)] = np.array([weight,duration])
                    GEdgesSet.add((FromNode, ToNode))
                for line in FH:
                    fields = line.split('\n')[0].split(',')
                    FromNode = fields[0]
                    ToNode = fields[1]
                    weight = float(fields[2].strip())
                    duration = ParseTimedelta(fields[3])
                    HEdges[(FromNode, ToNode)] = np.array([weight,duration])
                    HEdgesSet.add((FromNode, ToNode))
        # edges as the union of the two graphs
        AllEdges = GEdgesSet | HEdgesSet
        for edge in AllEdges:
            if edge in GEdges:
                GWeight = GEdges[edge]
            else:
                GWeight = np.array([0,initial_timedelta])
                # GWeight = np.array([0,HEdges[edge][1]])
            if edge in HEdges:
                HWeight = HEdges[edge]
            else:
                HWeight = np.array([0,initial_timedelta])
                # HWeight = np.array([0,GEdges[edge][1]])

            # print(GWeight, HWeight, abs(GWeight - HWeight) / max(GWeight, HWeight))
            value = np.array([0,0], dtype='float')
            value[0] = abs(GWeight[0] - HWeight[0]) / max(GWeight[0], HWeight[0])
            value[1] = abs(GWeight[1] - HWeight[1]) / max(GWeight[1], HWeight[1])
            # value = np.array([value_path, value_path])
            distance += value
            VerboseDistances.append((value, edge, GWeight, HWeight))
        distance /= len(AllEdges)
        distances.append(distance)
        # pprint.pprint(sorted(VerboseDistances, reverse=True)[:10])
    # print(distances)
    arr_dis = np.asarray(distances, dtype=np.float32)
    mean = np.mean(arr_dis, axis=0)
    sd = np.std(arr_dis, axis=0)
    Threshold = mean + 2 * sd
    print('mean: ', mean)
    print('sd: ', sd)
    print('Threshold: ', Threshold)
    with open(path_data + 'distances.csv', 'w') as f:
        for distance in distances:
            if distance[0] > Threshold[0]:  # label = 1
                label_path = 1
            else:  # label = 0
                label_path = 0
            if distance[1] > Threshold[1]:  # label = 1
                label_duration = 1
            else:  # label = 0
                label_duration = 0
            f.write(str(distance[0]) + ',' + str(distance[1]) + ',' + str(label_path) + ',' + str(label_duration)
                    + ',' + str(Threshold[0]) + ',' + str(Threshold[1]) + '\n')


def FindAbnorm():
    print('find out abnormal data based on the graph differences')
    dic_outlier = {}
    with open(path_data + 'distances.csv') as Dis:
        with open(path_data + 'outlier.csv', 'w') as f:
            LoopCounter = 0
            for line in Dis:
                # print(line)
                label_path = line.split(',')[2]
                label_duration = line.split(',')[3]
                # dict_label[LoopCounter] = (label_path, label_duration)
                if (label_path == '1') or (label_duration == '1'):
                    # dic_outlier[LoopCounter] =
                    f.write(str(RawTrajectories[LoopCounter][0]) + ',' + label_path + ',' + label_duration + '\n')
                LoopCounter += 1

    print('EHR anomalies are write into ' + path_data + 'outlier.csv file')
    print('finding abnormal data finished!!!')

# def Anom_inject(RawTrajectories, num_anom):
#     print('inject abnormal data')
#     list_ab_data = []
#     # length = random.sample(range(10, 50), num_anom)
#     length = [random.randint(10, 50) for i in range(num_anom)]
#     for id in range(0,num_anom):
#         # list_anom = random.sample(range(0, 34), length[id])
#         list_anom = [random.randint(0, 34) for i in range(length[id])]
#         list_ab_data.append([str(len(RawTrajectories)+id+1), [str(seq) for seq in list_anom]])
#     LineCount = 0
#     with open(path_data + 'RandomData' + str(num_anom) + '.csv', 'w') as f:
#         for data in list_ab_data:
#             id = data[0]
#             seq = ' '.join(data[1])
#             f.write(str(id) + ',' + str(seq) + '\n')
#             LineCount += 1
#     print(str(LineCount) + ' lines written.')
#     # for item in list_ab_data:
#     #     RawTrajectories.append(item)
#     RawTrajectories = RawTrajectories + list_ab_data
#     return RawTrajectories

def RdDataGen(num_anom, read_data=True):
    list_ab_data = []
    if not read_data:
        # generate random data as test
        print('generating random data')
        # length = random.sample(range(10, 50), num_anom)
        length = [random.randint(10, 50) for i in range(num_anom)]
        for id in range(0,num_anom):
            # list_anom = random.sample(range(0, 34), length[id])
            list_anom = [random.randint(0, 33) for i in range(length[id])]
            list_ab_data.append([str(len(RawTrajectories) + id + 1), [str(seq) for seq in list_anom]])
        print('Dumping generate data to file ' + path_data + 'RandomData' + str(num_anom) + '.csv')
        LineCount = 0
        with open(path_data + 'RandomData' + str(num_anom) + '.csv', 'w') as f:
            for data in list_ab_data:
                id = data[0]
                seq = ' '.join(data[1])
                f.write(str(id) + ' ' + str(seq) + '\n')
                LineCount += 1
        print(str(LineCount) + ' lines written.')
    else: # read data from file
        with open(PathRandomData) as f:
        # with open(PathRandomData) as f:
            for line in f:
                id = line.split(' ')[0]
                str_seq = line.split('\n')[0]
                list_anom = str_seq.split(' ')[1:]
                list_ab_data.append([id, [str(seq) for seq in list_anom]])
    return list_ab_data

def GenerateWholeGraph():
    ###
    # generate network with freq or possibility as edge weight
    ###
    OutputNetworkFileFreq = path_data + 'network-freq.csv'
    OutputNetworkFile = path_data + 'network.csv'
    OutputRulesFile = path_data + 'rules.csv'
    # print(OutputRulesFile, OutputNetworkFile)
    # TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    # VPrint(len(TrainingTrajectory))
    TrainingTrajectory = RawTrajectories
    Rules_Freq = BuildRulesFastParameterFreeFreq.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    # DumpRules(Rules_Freq, OutputRulesFile)
    # print(len(Rules_Freq))
    Network_Freq, Network = BuildNetwork.BuildNetwork(Rules_Freq)
    print(len(Network))
    DumpNetwork(Network_Freq, OutputNetworkFileFreq)
    DumpNetwork(Network, OutputNetworkFile)

def GeneratePartGraph():
    for x in range(len(RawTrajectories)):
        order_id = int(RawTrajectories[x][0])
        print('remove ' + str(order_id) + ' from dataset')
        OutputNetworkFile = path_network + 'network' + str(order_id-1) + '.csv'
        input = copy.deepcopy(RawTrajectories)
        input.pop(x)
        # TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(input)
        # VPrint(len(TrainingTrajectory))
        TrainingTrajectory = input
        Rules = BuildRulesFastParameterFreeFreq.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
        # DumpRules(Rules, OutputRulesFile)
        print(len(Rules))
        Network_Freq, Network = BuildNetwork.BuildNetwork(Rules)
        print(len(Network))
        DumpNetwork(Network, OutputNetworkFile)

def PlotDistance():
    path_file = path_data + 'distances.csv'
    path_visual = path_results + 'visual/'
    os.makedirs(path_visual, exist_ok=True)
    list_distance_path = []
    list_label_path = []
    list_threshold_path = []
    list_distance_duration = []
    list_label_duration = []
    list_threshold_duration = []
    with open(path_file) as f:
        for line in f:
            distance_path = line.split(',')[0]
            distance_duration = line.split(',')[1]
            label_path = line.split(',')[2]
            label_duration = line.split(',')[3]
            threshold_path = line.split(',')[4]
            threshold_duration = line.split(',')[5].split('\n')[0]
            list_distance_path.append(float(distance_path))
            list_label_path.append(label_path)
            list_threshold_path.append(float(threshold_path))
            list_distance_duration.append(float(distance_duration))
            list_label_duration.append(label_duration)
            list_threshold_duration.append(float(threshold_duration))
    PlotFigure(list_distance_path, list_threshold_path, list_label_path, path_visual, 'distance_path.png')
    PlotFigure(list_distance_duration, list_threshold_duration, list_label_duration, path_visual, 'distance_duration.png')

def PlotFigure(list_distance, list_threshold, list_label, path_visual, SaveFile):
    x = list(range(len(list_distance)))
    # Threshold = [0.018]*len(list_distance)
    fig = plt.figure(figsize=(8, 5))
    plt.xlabel('Data Index')
    plt.ylabel('Weight Distance')
    # plt.title('ADFON')
    # if MaxOrder == 1:
    #     plt.title('Weight Distance with FON')
    # else:
    #     plt.title('ADROS Weight Distance with HON')
    l1 = plt.plot(x, list_distance,'b', label='Weight Distance', alpha=0.7)
    # l3 = plt.bar(x, list_distance, align='center', width=0.1, fc='b', alpha=0.7)
    l2 = plt.plot(x, list_threshold, 'r--', label='Threshold', alpha=0.7)
    anomalyLabel = True
    for idx, label in enumerate(list_label):
        if (label == '1'):
            if anomalyLabel:
                plt.scatter(idx, list_distance[idx], marker='o', c='r', label='Anomaly', alpha=0.9)
                anomalyLabel = False
            else:
                plt.scatter(idx, list_distance[idx], marker='o', c='r', alpha=0.9)
    plt.legend()
    plt.show()
    fig.savefig(path_visual + SaveFile, dpi=300)
    plt.close()


def PlotComparison():
    ADHD = [9,16,32,46,100]
    AADHD = [9,19,48,97,487]
    x = [10,20,50,100,500]
    acc_ADHD = []
    acc_AADHD = []
    for id, num in enumerate(ADHD):
        acc1 = float(num)/x[id]
        acc2 = float(AADHD[id]) / x[id]
        acc_ADHD.append(acc1)
        acc_AADHD.append(acc2)
    x_x = list(range(len(x)))
    fig = plt.figure(figsize=(8, 5))
    plt.xlabel('Number of Injected Anomalies')
    plt.ylabel('Accuracy')
    plt.title('Accuracy comparison between ADHD and AADHD')
    l1 = plt.plot(x_x, acc_ADHD, '--bo', label='ADHD')
    l2 = plt.plot(x_x, acc_AADHD, '--r*', label='AADHD')
    plt.xticks(x_x, x)
    plt.legend()
    plt.show()
    fig.savefig(path_ADHD + 'Comparison.png')
    plt.close()

def CalAnoInRdm():
    # calculate how many anomalies in the generated random data
    numAnmRd = 0
    with open(path_data + 'outlier.csv') as f:
        for line in f:
            data_id = float(line.split(' ')[0])
            if data_id > num_Raw:
                numAnmRd += 1
    print('detect ', numAnmRd, '/' , NumInjectData, ' anomalies from injected data')

    # with open(path_ADHD + 'numAnomInRandm.csv', 'a+') as f:
    #     f.write(str(NumInjectData) + ' ' + str(numAnmRd) + '\n')
    # print('update the results to ', path_ADHD, 'numAnomInRandm.csv')


def CalOrderNum():
    Order = {} # key: order, value: numbers
    Order[1] = 0
    with open(path_data + 'network-ehr.csv') as f:
        for line in f:
            node = line.split(',')[0]
            dependency = node.split('|')[1]
            if dependency == '':
                Order[1] += 1
            else:
                OrderNum = len(dependency.split('.')) + 1
                if OrderNum not in Order.keys():
                    Order[OrderNum] = 0
                Order[OrderNum] += 1
    with open(path_data + 'order.csv', 'w') as f:
        f.write('Order,Number\n')
        for order in Order:
            num = Order[order]
            f.write(str(order) + ',' + str(num) + '\n')
    print('calculate order num finished!!!')

def GenerateNetworkWithoutAnomalies():
    list_IDAnomalies = []
    with open(path_data + 'outliers.csv') as f:
        for line in f:
            id = int(line.split(' ')[0])
            list_IDAnomalies.append(id)
    Trajectories = copy.deepcopy(RawTrajectories)
    list_IDAnomalies.sort(reverse=True)
    for id in list_IDAnomalies:
        Trajectories.pop(id-1)
    OutputNetworkFile = path_data + 'network-ehr-RemoveAnomalies.csv'
    # print(OutputRulesFile, OutputNetworkFile)
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(Trajectories)
    VPrint(len(TrainingTrajectory))
    Rules = BuildRulesFastParameterFree.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    # DumpRules(Rules, OutputRulesFile)
    print(len(Rules))
    Network = BuildNetwork.BuildNetwork(Rules)
    print(len(Network))
    DumpNetwork(Network, OutputNetworkFile)

###########################################
# Main function
###########################################
## Initialize algorithm parameters
MaxOrder = 5
MinSupport = 0
## Initialize user parameters
NumInjectData = 500 # the number of random data which will be injected into EHR
NumGenData = 100 # the number of generated random data

InputFolder = '../../data_preprocessed/tensor/Synthetic_1000/'
InputFileName = InputFolder + 'tensor_seq.pkl'
path_ADHD = '../../results/hon_tensor/Synthetic/'
path_results = path_ADHD + 'Synthetic_1000/'

path_data = path_results + 'data/'
path_network = path_results + 'network/'
os.makedirs(path_ADHD, exist_ok=True)
os.makedirs(path_results, exist_ok=True)
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_network, exist_ok=True)

# copy normal data to data folder
shutil.copy(InputFileName, path_data)

# ReadRandomData = True
ReadRandomData = False
if ReadRandomData:
    PathRandomData = InputFolder + 'SyntheticAnomalies2.csv'
    shutil.copy(PathRandomData, path_data)

LastStepsHoldOutForTesting = 0
MinimumLengthForTraining = 1
InputFileDeliminator = ' '
Verbose = False

if __name__ == "__main__":
    print('FREQ mode!!!!!!')
    RawTrajectories = ReadSequentialData(InputFileName)
    num_Raw = len(RawTrajectories)
    # print('data length ', str(num_Raw))
    # if ReadRandomData:
    #     RdData = RdDataGen(NumGenData, ReadRandomData)
    #     # RawTrajectories = RawTrajectories + RdData[:NumInjectData]
    #     RawTrajectories = RawTrajectories + RdData
    GenerateWholeGraph()
    # CalOrderNum() # calculate the number of each higher order dependencies
    GeneratePartGraph()
    GraphDiff()
    FindAbnorm()
    # CalAnoInRdm()
    PlotDistance()
    # GenerateNetworkWithoutAnomalies()
    # PlotComparison()
    print('test finished!!!')
