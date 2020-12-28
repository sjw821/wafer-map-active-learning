import numpy as np
import pandas as pd
import os
import pickle
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import math

def read_resize(resize_x=64, resize_y=64, data_directory=''):

    data_directory = data_directory

    df = pd.read_pickle("{}WM_withSize.pkl".format(data_directory))

    data_x = df[df['trianTestLabel'] != "[]"]['waferMap']
    data_y = df[df['trianTestLabel'] != "[]"]['failureType']  # exclude non-labled data
    data_x.reset_index(drop=True, inplace=True)
    data_y.reset_index(drop=True, inplace=True)

    split_idx = df[df['trianTestLabel'] != "[]"]['trianTestLabel']
    split_idx.reset_index(drop=True, inplace=True)
    del df
    training_idx = split_idx == "[['Training']]"  # True if training data

    wholedata = []

    for index, row in data_x.iteritems():

        resize_image = resize(row, (resize_x, resize_y), preserve_range=True)
        wholedata.append(resize_image)

        if index % 10000 == 0:
            print(str(index) + "index complete")

    wholedata = np.asarray(wholedata)

    class_map = {"[['none']]": 0, "[['Edge-Ring']]": 1, "[['Center']]": 2, "[['Edge-Loc']]": 3, \
                 "[['Loc']]": 4, "[['Random']]": 5, "[['Scratch']]": 6, "[['Donut']]": 7, "[['Near-full']]": 8}

    data_y = data_y.map(class_map).values

    # normalize to 0-1
    wholedata = wholedata / 2
    wholedata = np.reshape(wholedata, (wholedata.shape[0], wholedata.shape[1], wholedata.shape[2], 1))

    # with open('{}wholedata_64.pickle'.format(data_directory), 'wb') as f:
    #     pickle.dump([wholedata, data_y, training_idx], f)
    pickle.dump([wholedata, data_y, training_idx], open('{}wholedata_64.pickle'.format(data_directory), 'wb'), protocol=4)
    print('wholedata_64.pickle saved')

    return wholedata, data_y, training_idx


def random_split_in_trainingset(wholedata, data_y, training_idx):
    train_x = wholedata[training_idx]
    train_y = data_y[training_idx]
    test_x = wholedata[~training_idx]
    test_y = data_y[~training_idx]

    train_x, unlabeled_x, train_y, unlabeled_y = train_test_split(train_x, train_y, test_size=0.98, random_state=1,
                                                                  stratify=train_y)

    return train_x, train_y, test_x, test_y, unlabeled_x, unlabeled_y

def make_allrandom_initial_trainingset2(wholedata, data_y, training_idx, sample_count=400): # with get some val data
    train_x = wholedata[training_idx]
    train_y = data_y[training_idx]
    test_x = wholedata[~training_idx]
    test_y = data_y[~training_idx]

    sampleidx =np.random.choice(train_y.shape[0], sample_count, replace=False)

    train_x_b = train_x[sampleidx]
    train_y_b = train_y[sampleidx]
    unlabeled_x_b = np.delete(train_x, sampleidx, axis=0)
    unlabeled_y_b = np.delete(train_y, sampleidx, axis=0)

    print(np.unique(train_y_b, return_counts=True))

    all_indices = []
    for i in np.unique(train_y_b):
        indices = np.where(train_y_b == i)
        selected_indices = np.random.choice(indices[0], math.floor(indices[0].shape[0] / 2), replace=False)
        all_indices.append(selected_indices)

    train_indices = np.concatenate(all_indices)
    train_y_c = train_y_b[train_indices]
    train_x_c = train_x_b[train_indices]

    val_x_c = np.delete(train_x_b, train_indices, axis=0)
    val_y_c = np.delete(train_y_b, train_indices, axis=0)

    temp_no = val_y_c.shape[0] - int(sample_count/2)
    temp_index = np.where(val_y_c == 0)
    move_index = np.random.choice(temp_index[0], temp_no, replace=False)
    train_x_c = np.concatenate((train_x_c, val_x_c[move_index]), axis=0)
    train_y_c = np.concatenate((train_y_c, val_y_c[move_index]), axis=0)
    val_x_c = np.delete(val_x_c, move_index, axis=0)
    val_y_c = np.delete(val_y_c, move_index, axis=0)

    print(np.unique(train_y_c, return_counts=True))
    print(np.unique(val_y_c, return_counts=True))

    return train_x_c, train_y_c, val_x_c, val_y_c, test_x, test_y, unlabeled_x_b, unlabeled_y_b


if __name__ == '__main__':

    data_directory='/home/woong/WFmap/'
    #data_directory='/home/gait/Gait/woong/WFmap/'
    # data_directory='/home/student/WFmap/'

    if os.path.exists('{}wholedata_64.pickle'.format(data_directory)):
        with open('{}wholedata_64.pickle'.format(data_directory), 'rb') as f:
            wholedata, data_y, training_idx = pickle.load(f)
        print('read data from pkl')
    else:
        wholedata, data_y, training_idx = read_resize(resize_x=64, resize_y=64, data_directory= data_directory )
        print('make data from original data. complete')

    train_x, train_y, val_x, val_y, test_x, test_y, unlabeled_x, unlabeled_y = make_allrandom_initial_trainingset2(wholedata, data_y, training_idx, sample_count=400)

    pickle.dump([train_x, train_y, val_x, val_y, test_x, test_y, unlabeled_x, unlabeled_y], open('allrandom_data_1_with_val.pickle', 'wb'), protocol=4)

    print('split complete : train shape {}, unlabeled shape {}, test shape {}'.format(train_x.shape, unlabeled_x.shape,
                                                                                      test_x.shape))