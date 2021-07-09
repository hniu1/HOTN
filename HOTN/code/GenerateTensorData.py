'''
Generate tensor data for test
'''
import pickle

def GenerateTensor():
    data_1 = [('1', '2021-04-21 11:13:00'),('3','2021-04-21 11:14:00'),('4','2021-04-21 11:15:00')]
    data_2 = [('2', '2021-04-21 11:13:00'),('3','2021-04-21 11:14:00'),('5','2021-04-21 11:15:00')]
    data_3 = [('6', '2021-04-21 11:13:00'),('3','2021-04-21 11:14:00'),('4','2021-04-21 11:15:00')]
    data_4 = [('6', '2021-04-21 11:13:00'),('3','2021-04-21 11:14:00'),('5','2021-04-21 11:15:00')]
    data_5 = [('1', '2021-04-21 11:13:00'), ('3', '2021-04-21 11:14:00'), ('4', '2021-04-22 11:15:00')]


    list_data = []
    for ix in range(1,6):
        list_data.append([str(ix)] + data_1)
    for ix in range(6, 11):
        list_data.append([str(ix)] + data_2)
    for ix in range(11, 12):
        list_data.append([str(ix)] + data_3)
    for ix in range(12, 14):
        list_data.append([str(ix)] + data_4)
    for ix in range(14, 15):
        list_data.append([str(ix)] + data_5)

    with open(path_save + 'tensor_test.pkl', 'wb') as f:
        pickle.dump(list_data, f)

    print('')

if __name__ == '__main__':
    path_save = '../../data_preprocessed/tensor/synthetic_test/'

    GenerateTensor()
    print('test finished!')
