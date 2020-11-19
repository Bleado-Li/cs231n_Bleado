import numpy
import numpy as np
import time


def process_bar(i, calculate_num, start_str='', end_str='', total_length=0):
    percent = i / calculate_num
    bar = ''.join(["\033[31m%s\033[0m" % '   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)


class NearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)

            # L1
            # distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            # L2
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))

            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

            process_bar(i, 500000000, start_str='', end_str='100%', total_length=15)

        return Ypred


def unpickle(file):

    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')

    return dictionary


# 合并数据集
def merge_data(file_path):

    dictionary = {b'batch_label': "merge data", b'labels': [], b'data': None, b'filenames': []}

    l = len(file_path)

    nparry = unpickle(file_path[0])[b'data']

    for i in range(l):

        dic = unpickle(file_path[i])

        dictionary[b'labels'] += dic[b'labels']

        dictionary[b'filenames'] += dic[b'filenames']

    for i in range(1, l):

        dic = unpickle(file_path[i])

        nparry = numpy.vstack((nparry, dic[b'data']))

    dictionary[b'data'] = nparry

    return dictionary


# 最近邻算法
def nearest_Neighbor_distance(sample_dic, test_dic):

    sample_data = sample_dic[b'data']
    test_data = test_dic[b'data']
    distance = []
    mix_distance = 0

    # L1
    for i in range():
        pass

    # L2
    for i in range():
        pass

    test_label = None

    return test_label


if __name__ == '__main__':

    dic = merge_data([".\cifar-10-python\cifar-10-batches-py\data_batch_1",
                      ".\cifar-10-python\cifar-10-batches-py\data_batch_2",
                      ".\cifar-10-python\cifar-10-batches-py\data_batch_3",
                      ".\cifar-10-python\cifar-10-batches-py\data_batch_4",
                      ".\cifar-10-python\cifar-10-batches-py\data_batch_5"])

    dic_test = unpickle(".\\cifar-10-python\\cifar-10-batches-py\\test_batch")

    dic = unpickle(".\cifar-10-python\cifar-10-batches-py\data_batch_1")
    print(dic.keys())
    print(dic[b'batch_label'])
    # print(dic[b'labels'])
    print(len(dic[b'labels']))
    print(type(dic[b'data']))
    print(dic[b'data'].shape)
    print(dic[b'data'])
    print(dic)


    # flatten out all images to be one-dimensional
    # Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
    # Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072
    Xtr_rows = dic[b'data']
    Xte_rows = dic_test[b'data']
    Ytr = np.array(dic[b'labels'])
    Yte = np.array(dic_test[b'labels'])

    # 计时器开始
    time_start = time.time()

    nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
    print("testing……")
    nn.train(Xtr_rows, Ytr)  # train the classifier on the training images and labels
    Yte_predict = nn.predict(Xte_rows)  # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print('accuracy: %f' % (np.mean(Yte_predict == Yte)))

    # 计时器停止
    time_end = time.time()
    print('totally cost: %d h %d min %f s ' % ((time_end - time_start)/3600, (time_end - time_start)/60, (time_end - time_start) % 60))



