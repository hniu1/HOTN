#coding=utf-8
'''
build synthetic data for abnormal section detection
1. synthesizeMixOrder for 3 constraints for normal data
2. synthesizeAbnormal for 3 new constraints
3. synthesizeDynData for new constraints
'''

import random
import os
import pickle

def GenerateSteps():
    lst_actions = list(range(0, num_act))
    # lst_steps = []
    for step_id in range(0, int(len(lst_actions) / ActInStep)):
        step = []
        for x in range(0, ActInStep):
            step.append(lst_actions[ActInStep * step_id + x])
        lst_steps.append(step)

def SynthesizeMixOrder():
    lst_index = []
    counter_1st = 0
    counter_2nd = 0
    counter_3rd = 0
    for x in range(0, NumSeqData):
        action_index = []
        for i in range(seq_len):
            # add first order dependencies. when from action 0 of step 0 to step 1, 0->3 50%, 0->4 50%
            if i == 1:
                if action_index[i - 1] == 0:
                    counter_1st += 1
                    action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)
            # add second order dependencies. when from action 6->10, 50% to 12, 50% to 13
            elif i == 4:
                if action_index[i - 1] == 1 and action_index[i - 2] == 0:
                    counter_2nd += 1
                    action = 0
                    # action = random.choice([0, 1])
                else:
                        action = random.randint(0, ActInStep - 1)
            elif i == 8:
                if action_index[i-1] == 0 and action_index[i-2] == 0 and action_index[i-3] == 0:
                    counter_3rd += 1
                    # action = random.choice([0, 1])
                    action = 0
                else:
                    action = random.randint(0, ActInStep - 1)
            else:
                action = random.randint(0, ActInStep - 1)
            action_index.append(action)
        lst_index.append(action_index)
    print('Number of 1st order dependency: ', counter_1st)
    print('Number of 2nd order dependency: ', counter_2nd)
    print('Number of 3rd order dependency: ', counter_3rd)
    trajectories = GenerateTrajectories(lst_index)

    # write synthetic normal data into files
    # file_synthetic = OutputFolder + 'synthetic_MixOrder_' + str(NumSeqData) + '.csv'
    file_synthetic = OutputFolder + 'synthetic_MixOrder_' + str(NumSeqData) + '_v2.csv'
    WriteTrajectories(trajectories, file_synthetic)
    print('synthesize mix order finished!')

def SynthesizeAADHONData():
    NumSeqData = 1995
    lst_index = []
    counter_1st = 0
    counter_2nd = 0
    counter_3rd = 0
    file_synthetic = OutputFolder + 'synthetic_AADHON_6k.csv'
    # 1
    for x in range(0, NumSeqData):
        action_index = []
        for i in range(seq_len):
            # add first order dependencies. when from action 0 of step 0 to step 1, 0->3 50%, 0->4 50%
            if i == 1:
                if action_index[i - 1] == 0:
                    counter_1st += 1
                    action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)
            # add second order dependencies. when from action 6->10->12
            elif i == 4:
                if action_index[i - 1] == 1 and action_index[i - 2] == 0:
                    counter_2nd += 1
                    action = 0
                    # action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)
            elif i == 8:
                if action_index[i-1] == 0 and action_index[i-2] == 0 and action_index[i-3] == 0:
                    counter_3rd += 1
                    # action = random.choice([0, 1])
                    action = 0
                else:
                    action = random.randint(0, ActInStep - 1)
            else:
                action = random.randint(0, ActInStep - 1)
            action_index.append(action)
        lst_index.append(action_index)
    print('Number of 1st order dependency: ', counter_1st)
    print('Number of 2nd order dependency: ', counter_2nd)
    print('Number of 3rd order dependency: ', counter_3rd)
    trajectories = GenerateTrajectories(lst_index)
    # add 1st anomalies
    lst_anomalies = [
        [0, 5, 6, 9, 14, 15, 20, 23, 24, 28],
        [0, 3, 6, 10, 13, 16, 19, 22, 25, 28],
        [0, 3, 6, 10, 12, 15, 18, 21, 25, 29],
        [0, 4, 7, 9, 12, 15, 18, 21, 26, 28],
        [0, 5, 6, 10, 14, 15, 18, 21, 25, 27]
    ]
    trajectories += lst_anomalies

    # 2
    lst_index = []
    # add new patterns
    for x in range(0, NumSeqData):
        action_index = []
        for i in range(seq_len):
            # add first order dependencies. when from action 0 of step 0 to step 1, 0->3 50%, 0->4 50%
            if i == 1:
                if action_index[i - 1] == 0:
                    counter_1st += 1
                    action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)

            elif i == 4:
                # add second order dependencies. when from action 6->10->13
                if action_index[i - 1] == 1 and action_index[i - 2] == 0:
                    counter_2nd += 1
                    action = 1
                    # action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)
            elif i == 8:
                if action_index[i-1] == 0 and action_index[i-2] == 0 and action_index[i-3] == 0:
                    counter_3rd += 1
                    # action = random.choice([0, 1])
                    action = 0
                else:
                    action = random.randint(0, ActInStep - 1)
            else:
                action = random.randint(0, ActInStep - 1)
            action_index.append(action)
        lst_index.append(action_index)
    trajectories += GenerateTrajectories(lst_index)

    # add 2nd anomalies
    lst_anomalies = [
        [0, 5, 6, 9, 14, 15, 20, 23, 24, 28],
        [0, 3, 6, 10, 14, 16, 19, 22, 25, 28],
        [0, 3, 6, 10, 12, 15, 18, 21, 25, 29],
        [0, 4, 7, 9, 12, 15, 18, 21, 26, 28],
        [0, 5, 6, 10, 14, 15, 18, 21, 25, 27]
    ]
    trajectories += lst_anomalies

    print('Number of 1st order dependency: ', counter_1st)
    print('Number of 2nd order dependency: ', counter_2nd)
    print('Number of 3rd order dependency: ', counter_3rd)

    # 3
    lst_index = []
    # add new patterns
    for x in range(0, NumSeqData):
        action_index = []
        for i in range(seq_len):
            # add first order dependencies. when from action 0 of step 0 to step 1, 0->3 50%, 0->4 50%
            if i == 1:
                if action_index[i - 1] == 0:
                    counter_1st += 1
                    action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)

            elif i == 4:
                # add second order dependencies. when from action 6->10-> 13
                if action_index[i - 1] == 1 and action_index[i - 2] == 0:
                    counter_2nd += 1
                    action = 1
                    # action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)
            elif i == 8:
                # 15 -> 18 -> 21 -> 24
                if action_index[i - 1] == 0 and action_index[i - 2] == 0 and action_index[i - 3] == 0:
                    counter_3rd += 1
                    # action = random.choice([0, 1])
                    action = 0
                # 16 -> 19 -> 22 -> 25
                elif action_index[i - 1] == 1 and action_index[i - 2] == 1 and action_index[i - 3] == 1:
                    counter_3rd += 1
                    # action = random.choice([0, 1])
                    action = 1
                else:
                    action = random.randint(0, ActInStep - 1)
            else:
                action = random.randint(0, ActInStep - 1)
            action_index.append(action)
        lst_index.append(action_index)
    trajectories += GenerateTrajectories(lst_index)

    # add 3rd anomalies
    lst_anomalies = [
        [0, 5, 6, 9, 14, 15, 20, 23, 24, 28],
        [0, 3, 6, 10, 14, 16, 19, 22, 25, 28],
        [0, 3, 6, 10, 12, 16, 19, 22, 24, 29],
        [0, 4, 7, 9, 12, 16, 19, 22, 26, 28],
        [0, 5, 6, 10, 14, 15, 18, 21, 25, 27]
    ]
    trajectories += lst_anomalies

    print('Number of 1st order dependency: ', counter_1st)
    print('Number of 2nd order dependency: ', counter_2nd)
    print('Number of 3rd order dependency: ', counter_3rd)

    # # 4
    # lst_index = []
    # # add new patterns
    # for x in range(0, NumSeqData):
    #     action_index = []
    #     for i in range(seq_len):
    #         # add first order dependencies. when from action 0 of step 0 to step 1, 0->3 50%, 0->4 50%
    #         if i == 1:
    #             if action_index[i - 1] == 0:
    #                 counter_1st += 1
    #                 action = random.choice([0, 1])
    #             else:
    #                 action = random.randint(0, ActInStep - 1)
    #
    #         elif i == 4:
    #             # add second order dependencies. when from action 6->10, 100% to 13
    #             if action_index[i - 1] == 1 and action_index[i - 2] == 0:
    #                 counter_2nd += 1
    #                 action = 1
    #                 # action = random.choice([0, 1])
    #             else:
    #                 action = random.randint(0, ActInStep - 1)
    #         elif i == 8:
    #             if action_index[i - 1] == 0 and action_index[i - 2] == 0 and action_index[i - 3] == 0:
    #                 counter_3rd += 1
    #                 # action = random.choice([0, 1])
    #                 action = 2
    #             else:
    #                 action = random.randint(0, ActInStep - 1)
    #         else:
    #             action = random.randint(0, ActInStep - 1)
    #         action_index.append(action)
    #     lst_index.append(action_index)
    # trajectories += GenerateTrajectories(lst_index)
    #
    # # add anomalies
    # lst_anomalies = [
    #     [0, 5, 6, 9, 14, 15, 20, 23, 24, 28],
    #     [0, 3, 6, 10, 13, 16, 19, 22, 26, 28],
    #     [0, 3, 6, 10, 12, 15, 18, 21, 25, 29],
    #     [0, 4, 7, 9, 12, 15, 18, 21, 26, 28],
    #     [0, 5, 6, 10, 14, 15, 18, 21, 25, 27]
    # ]
    # trajectories += lst_anomalies
    #
    # print('Number of 1st order dependency: ', counter_1st)
    # print('Number of 2nd order dependency: ', counter_2nd)
    # print('Number of 3rd order dependency: ', counter_3rd)
    #
    # # 5
    # lst_index = []
    # # add new patterns
    # for x in range(0, NumSeqData):
    #     action_index = []
    #     for i in range(seq_len):
    #         # add first order dependencies. when from action 0 of step 0 to step 1, 0->3 50%, 0->4 50%
    #         if i == 1:
    #             if action_index[i - 1] == 0:
    #                 counter_1st += 1
    #                 action = random.choice([0, 1])
    #             else:
    #                 action = random.randint(0, ActInStep - 1)
    #
    #         elif i == 4:
    #             # add second order dependencies. when from action 6->10, 100% to 13
    #             if action_index[i - 1] == 1 and action_index[i - 2] == 0:
    #                 counter_2nd += 1
    #                 action = 1
    #                 # action = random.choice([0, 1])
    #             else:
    #                 action = random.randint(0, ActInStep - 1)
    #         elif i == 8:
    #             if action_index[i - 1] == 1 and action_index[i - 2] == 1 and action_index[i - 3] == 1:
    #                 counter_3rd += 1
    #                 # action = random.choice([0, 1])
    #                 action = 1
    #             else:
    #                 action = random.randint(0, ActInStep - 1)
    #         else:
    #             action = random.randint(0, ActInStep - 1)
    #         action_index.append(action)
    #     lst_index.append(action_index)
    # trajectories += GenerateTrajectories(lst_index)
    #
    # # add anomalies
    # lst_anomalies = [
    #     [0, 5, 6, 9, 14, 15, 20, 23, 24, 28],
    #     [0, 3, 6, 10, 13, 16, 19, 22, 26, 28],
    #     [0, 3, 6, 10, 12, 15, 18, 21, 25, 29],
    #     [0, 4, 7, 9, 12, 15, 18, 21, 26, 28],
    #     [0, 5, 6, 10, 14, 15, 18, 21, 25, 27]
    # ]
    # trajectories += lst_anomalies
    #
    # print('Number of 1st order dependency: ', counter_1st)
    # print('Number of 2nd order dependency: ', counter_2nd)
    # print('Number of 3rd order dependency: ', counter_3rd)

    # write synthetic normal data into files
    # file_synthetic = OutputFolder + 'synthetic_MixOrder_' + str(NumSeqData) + '.csv'
    WriteTrajectories(trajectories, file_synthetic)
    print('synthesize mix order finished!')

# inject 3 anomalies.
def SynthesizeAnomalies():
    lst_index = []
    for x in range(0,num_anom):
        action_index = []
        # insert first order dependency 0->5
        if x == 0:
            for i in range(seq_len):
                if i == 0:
                    action = 0
                elif i == 1:
                    action = 2
                else:
                    action = random.randint(0, ActInStep - 1)
                action_index.append(action)
        # insert 2nd order dependency 6->10->14
        elif x == 1:
            for i in range(seq_len):
                if i == 2:
                    action = 0
                elif i == 3:
                    action = 1
                elif i == 4:
                    action = 2
                else:
                    action = random.randint(0, ActInStep - 1)
                action_index.append(action)
        else:
            for i in range(seq_len):
                if i == 5:
                    action = 0
                elif i == 6:
                    action = 0
                elif i == 7:
                    action = 0
                elif i == 8:
                    if x == 2:
                        action = 1
                    else:
                        action = 2
                else:
                    action = random.randint(0, ActInStep - 1)
                action_index.append(action)
        lst_index.append(action_index)
    trajectories = GenerateTrajectories(lst_index)
    # write synthetic normal data into files
    file_synthetic = OutputFolder + 'synthetic_anomalies_' + str(num_anom) + '.csv'
    WriteTrajectories(trajectories, file_synthetic, True)
    print('synthesize mix order finished!')
    print('Anomalies Generated!')


def SynthesizeDynData():
    lst_index = []
    counter_1st = 0
    counter_2nd = 0
    counter_3rd = 0
    NumDynData = 100
    for x in range(0, NumDynData):
        action_index = []
        for i in range(seq_len):
            # add first order dependencies. when from action 0 of step 0 to step 1, 0->3 50%, 0->4 50%
            if i == 1:
                if action_index[i - 1] == 0:
                    counter_1st += 1
                    action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)
            # add second order dependencies. when from action 6->10, 100% to 12
            elif i == 4:
                if action_index[i - 1] == 1 and action_index[i - 2] == 0:
                    counter_2nd += 1
                    action = 0
                    # action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)
            # elif i == 8:
            #     if action_index[i - 1] == 0 and action_index[i - 2] == 0 and action_index[i - 3] == 0:
            #         counter_3rd += 1
            #         # action = random.choice([0, 1])
            #         action = 0
            #     else:
            #         action = random.randint(0, ActInStep - 1)
            # new constraint: action 17, 20, 23, 26
            elif i==5:
                action = 2
            elif i == 6:
                action = 2
            elif i == 7:
                action = 2
            elif i == 8:
                action = 2
            else:
                action = random.randint(0, ActInStep - 1)
            action_index.append(action)
        lst_index.append(action_index)
    print('Number of 1st order dependency: ', counter_1st)
    print('Number of 2nd order dependency: ', counter_2nd)
    print('Number of 3rd order dependency: ', counter_3rd)
    trajectories = GenerateTrajectories(lst_index)

    # write synthetic normal data into files
    # file_synthetic = OutputFolder + 'synthetic_MixOrder_' + str(NumSeqData) + '.csv'
    file_synthetic = OutputFolder + 'DynamicData_' + str(NumDynData) + '.csv'
    WriteTrajectories(trajectories, file_synthetic, True)
    print('synthesize dynamic data finished!')

def SynthesizeTensorData():
    lst_index = []
    counter_1st = 0
    counter_2nd = 0
    counter_3rd = 0
    for x in range(0, NumSeqData):
        action_index = []
        for i in range(seq_len):
            # add first order dependencies. when from action 0 of step 0 to step 1, 0->3 50%, 0->4 50%
            if i == 1:
                if action_index[i - 1] == 0:
                    counter_1st += 1
                    action = random.choice([0, 1])
                else:
                    action = random.randint(0, ActInStep - 1)
            # add second order dependencies. when from action 6->10, 50% to 12, 50% to 13
            elif i == 4:
                if action_index[i - 1] == 0 and action_index[i - 2] == 0:
                    counter_2nd += 1
                    action = random.choice([0, 1])
                    # action = random.choice([0, 1])
                else:
                        action = random.randint(0, ActInStep - 1)
            # add 3rd order dependencies. when from action 2->5->8, 100% to 14
            elif i == 3:
                if action_index[i-1] == 2 and action_index[i-2] == 2 and action_index[i-3] == 2:
                    counter_3rd += 1
                    # action = random.choice([0, 1])
                    action = 2
                else:
                    action = random.randint(0, ActInStep - 1)
            else:
                action = random.randint(0, ActInStep - 1)
            action_index.append(action)
        lst_index.append(action_index)
    print('Number of 1st order dependency: ', counter_1st)
    print('Number of 2nd order dependency: ', counter_2nd)
    print('Number of 3rd order dependency: ', counter_3rd)
    trajectories = GenerateTrajectories(lst_index)

    # write synthetic normal data into files
    # file_synthetic = OutputFolder + 'synthetic_MixOrder_' + str(NumSeqData) + '.csv'
    # file_synthetic = OutputFolder + 'synthetic_' + str(NumSeqData) + '_v2.csv'
    # WriteTrajectories(trajectories, file_synthetic)
    list_data = []
    list_time = ['2021-04-16 12:00:00', '2021-04-16 12:01:00', '2021-04-16 12:02:00', '2021-04-16 12:03:00', '2021-04-16 12:04:00']
    for id, t in enumerate(trajectories):
        list_order = []
        for ix, act in enumerate(t):
            list_order.append((str(act), list_time[ix]))
        list_data.append([str(id+1)] + list_order)
    # add synthetic outliers
    data_1 = [('0', list_time[0]), ('5', list_time[1]), ('6', list_time[2]), ('9', list_time[3]), ('12', list_time[4])]
    data_2 = [('0', list_time[0]), ('3', list_time[1]), ('6', list_time[2]), ('9', list_time[3]), ('14', list_time[4])]
    data_3 = [('2', list_time[0]), ('5', list_time[1]), ('8', list_time[2]), ('10', list_time[3]), ('14', list_time[4])]
    data_4 = [('0', list_time[0]), ('3', list_time[1]), ('6', list_time[2]), ('9', list_time[3]), ('12', '2021-04-16 12:13:00')]
    data_5 = [('1', list_time[0]), ('4', list_time[1]), ('7', list_time[2]), ('10', '2021-04-16 12:12:00'), ('13', '2021-04-16 12:22:00')]
    list_data.append([str(len(list_data) + 1)] + data_1)
    list_data.append([str(len(list_data) + 1)] + data_2)
    list_data.append([str(len(list_data) + 1)] + data_3)
    list_data.append([str(len(list_data) + 1)] + data_4)
    list_data.append([str(len(list_data) + 1)] + data_5)
    with open(OutputFolder + 'tensor_seq.pkl', 'wb') as f:
        pickle.dump(list_data, f)
    print('synthesize mix order finished!')


def GenerateTrajectories(lst_index):
    trajectories = []
    for indexes in lst_index:
        trajectory = []
        for step_id, index in enumerate(indexes):
            action = lst_steps[step_id][index]
            trajectory.append(action)
        # print(trajectory)
        trajectories.append(trajectory)
    print(len(trajectories))
    return trajectories

def WriteTrajectories(trajectories, file_path, anomalies=False):
    with open(file_path, 'w') as f:
        for id, seq in enumerate(trajectories):
            str_seq = [str(act) for act in seq]
            data_seq = ' '.join(str_seq)
            if anomalies:
                id = id + NumSeqData
            f.write(str(id+1) + ' ' + data_seq + '\n')
    print('write {} seq data into file {}'.format(len(trajectories), file_path))

def SynthesizeInclusiveDetection():
    with open(OutputFolder+'synthesize.csv', 'w') as f:
        for id in range(0,1000):
            f.write(str(id+1) + ' 1 7 4 2' + '\n')
################ main #################
OutputFolder = '../../data_preprocessed/tensor/Synthetic_1000/'
# OutputFolder = '../data_preprocessed/hon/synthetic_dynamic/'
os.makedirs(OutputFolder, exist_ok=True)
seq_len = 5
ActInStep = 3
num_act = seq_len * ActInStep
NumSeqData = 1000
num_anom = 4
lst_steps = []

if __name__ == '__main__':
    GenerateSteps()
    # SynthesizeAADHONData()
    # SynthesizeMixOrder()
    # SynthesizeAnomalies()
    # SynthesizeDynData()
    # SynthesizeInclusiveDetection()
    SynthesizeTensorData()
    print('test finished!!!')
