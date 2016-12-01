import numpy
import collections
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from sklearn.cross_validation import StratifiedKFold
from utils import dataprep as dp
from datasource.data import DataSet
from utils import utility as util
from sklearn.metrics import classification_report, accuracy_score, \
    confusion_matrix
import os
import time


def read_data_sets(rootDir,
                   scaler,
                   n_folds=5,
                   sampling_rate=50,
                   reshape=True,
                   scale=True,
                   normalize=True,
                   resample=True):
    # extract training data from the root directory of the ground truth
    labels, \
    data, \
    target, \
    labelsdict, \
    avg_len_emg, \
    avg_len_acc, \
    user_map, \
    user_list, \
    data_dict, \
    max_length_emg, \
    max_length_others, \
    data_path = dp.getTrainingData(rootDir)

    # scale the training data
    print('scaling data')
    if scale:
        if scaler is not None:
            data = dp.scaleData(data,
                                scaler)  # add a print statement that will
            # indicate which instance is being scaled and how many left. do
            # this for others as well

    # normalize the training instances to a common length to preserve the
    # sampling to be used later for extracting features
    if normalize:
        print('normalizing data')
        data = dp.normalizeTrainingData(data, max_length_emg, max_length_others)

    # resample all the training instances to normalize the data vectors
    # resample also calls consolidate data so there is no need to call
    # consolidate raw data again
    if resample:
        print('resample data')
        data = dp.resampleTrainingData(data, sampling_rate, avg_len_acc,
                                       emg=False, imu=True)

    skf = StratifiedKFold(target, n_folds)

    target = dp.discritizeLabels(target)
    data = dp.prepareDataset(data)
    if n_folds > 0:
        kFolds = []
        for train, test in skf:
            print('split train and validation data')
            train_x, train_y, val_x, val_y = dp.splitDataset(train, test,
                                                             target, data)
            train = DataSet(train_x, train_y, reshape=reshape)
            validation = DataSet(val_x, val_y, reshape=reshape)
            kFolds.append((train, validation))
        return kFolds, train_x.shape
    else:
        return DataSet(data, target, reshape=reshape)


def load_vds():
    return read_data_sets()
