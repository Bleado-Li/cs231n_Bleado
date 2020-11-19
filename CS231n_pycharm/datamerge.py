import numpy
import numpy as np



# 合并数据集
def merge_data(dict_list):

    dictionary = {b'batch_label': "merge data", b'labels': [], b'data': None, b'filenames': []}

    l = len(dict_list)

    for i in range(l):

        dic = dict_list[i]

        dictionary[b'labels'] += dic[b'labels']

        dictionary[b'filenames'] += dic[b'filenames']

    nparry = dict_list[0][b'data']

    for i in range(1, l):

        dic = dict_list[i]

        nparry = numpy.vstack((nparry, dic[b'data']))

    dictionary[b'data'] = nparry

    return dictionary


if __name__ == '__main__':


    dict_1 = [{b'batch_label': "merge data1", b'labels': [1,4], b'data': np.array([[1,2],[3,4]]), b'filenames': ["a"]},
              {b'batch_label': "merge data2", b'labels': [2,5], b'data': np.array([[2,2],[3,4]]), b'filenames': ["b"]},
              {b'batch_label': "merge data3", b'labels': [3,6], b'data': np.array([[3,2],[3,4]]), b'filenames': ["c"]}]
    print(merge_data(dict_1))


    # print(dict_1[0][b'data'])
    # print(type(dict_1[0][b'data']))
    # print(numpy.vstack((dict_1[0][b'data'], dict_1[1][b'data'])))


