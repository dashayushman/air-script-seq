# Import dependencies
import os, logging, json
from datasource import TrainingInstance as tri
import numpy as np
from utils import feature_extractor as fe
import pickle
import random

def generate_label_sequences(labels, n_instances, l_range=(1, 30)):
    '''
    Generates a given number of label sequences
    :param labels: a list of available unique labels
    :param n_instances: number of instances (sequences) to generate
    :param range: A tuple that holds the min and max length of sequences
    :return: list of sequences and list of their corresponding lengths
    '''
    seqs = []
    seq_lengths = []

    for i in np.arange(l_range[0], l_range[1]+1):
        logging.info('Generating sequences of length ' + str(i))
        for j in range(n_instances):
            if j%20 == 0:
                logging.info('Generating ' + str(j) + 'th sequence of length '
                             + str(i))
            seqs.append(random.sample(labels, i))
            seq_lengths.append(i)
    return seqs, seq_lengths

def generate_data_sequences(codebook, labele_seqs, l_range=(1, 30)):
    '''
    Generates a given number of data sequences
    :param codebook: A codebook (dict obj) that holds all the labels as keys
    and all the corresponding data instances as values
    :param labele_seqs: Sequence of labels for which data sequences needs to
    be generated
    :param range: A tuple that holds the min and max length of sequences
    :return:
    '''

    seqs = []
    t_seqs = []

    for j, label_seq in enumerate(labele_seqs) :
        if j % 20 == 0 :
            logging.info('Generating ' + str(j) + 'th data sequence of length '
                         + str(len(label_seq)))
        d_seq_buf = None
        for lbl in label_seq:
            if d_seq_buf is None:
                d_seq_buf = np.array(random.sample(codebook[lbl], 1))
            else:
                rand_lbl_sample = random.sample(codebook[lbl], 1)
                d_seq_buf = np.concatenate((d_seq_buf, rand_lbl_sample), axis=1)

        seqs.append(d_seq_buf)
        t_seqs.append(np.transpose(d_seq_buf))

    return seqs, t_seqs


def scaleData(data, scaler):
    '''
    Method to scale the sensor data as a preprocessing step
    :param data: (list) List of all the training instance objects
    :return: data: (list) List of a ll the scaled training data objects
    '''

    # data = np.array([ti.scaleData(scaler) for ti in data])
    d = []
    for i, ti in enumerate(data):
        if i % 50 is 0:
            print(
            str(i) + ' out of ' + str(len(data)) + 'training instances scaled')
        d.append(ti.scaleData(scaler))
    return np.array(d)


# Serielize objects to disk for future reuse to make things faster
def dumpObject(filePath, object):
    try:
        with open(filePath, 'wb') as f:
            pickle.dump(object, f)
            return True
    except IOError as e:
        return False


def loadObject(filePath):
    if os.path.isfile(filePath):
        with open(filePath, 'rb') as f:
            object = pickle.load(f)
            return object
    else:
        return None


def read_json_file(filepath):
    with open(filepath) as data_file:
        data = json.load(data_file)
        return data


# Get training data from the root directory where the ground truth exists

def getTrainingData(rootdir):
    '''
    This method gets all the training data from the root directory of the
    ground truth
    The following is the directory structure for the ground truth
    Root_Dir
        |_Labels
            |_Participants
                |_data_files
    :param rootdir (string): path to the rood directory where the ground
    truth exists
    :return:    labels      (list),                 A list of class labels
                data        (list),                 A list of training instances
                target      (list),                 A list of class labels
                corresponding to the training instances in the in 'data'
                labelsdict  (dictionary),           A dictionary for
                converting class labels from string to integer and vice versa
                avg_len     (float),                The average length of the
                sensor data (emg, accelerometer, gyroscope and orientation)
                which would later be used for normalization
                user_map    (dictionary),           A dictionary of all
                participants and their corresponding file list to be used for
                leave one out test later
                user_list   (list),                 A list of all participants
                data_dict   (dictionary)            A dictionary containing a
                mapping of all the class labels, participants, and files of
                the participants which can be used later for transforming the
                data for leave one out test
                max_len     (integer)               the maximum length of the
                sensor data
                data_path   (list)                  A list that will hold the
                path to every training instance in the 'data list'
    '''

    # List of all training labels
    training_class_dirs = os.walk(rootdir)

    labels = []  # Empty list to hold all the class labels
    labelsdict = {}  # Dictionary to store the labels and their correspondig
    # interger values
    labeldirs = []  # Directory paths of all labels
    target = []  # List that will hold class labels of the training instances
    #  in 'data list'
    data = []  # List that will hold all the training/validation instances
    sample_len_vec_emg = []  # List that holds that length of all the sensor
    # data. It will be used later for calculating average length
    sample_len_vec_others = []  # List that holds that length of all the
    # sensor data. It will be used later for calculating average length
    data_dict = {}  # The dictionary that will hold that mappings for all
    # labels, participants of the the label and data files corresponding to
    # all the participants. This will be used later for leave one out test
    user_map = {}  # A dictionary that will hold the mappings for all
    # participants and their corresponding ids
    user_list = []  # A list of all participants
    user_ids = np.arange(
        100).tolist()  # A pre generated list of userids for assigning a
    # unique id to every user
    data_path = []  # A list that will hold the path to every training
    # instance in the 'data list'

    # A codebook of all the labels and their corresponding
    codebook = {}

    # Get the list of labels by walking the root directory
    for trclass in training_class_dirs:
        labels = trclass[1]
        break

    # extracting list of participants for each label
    for i, label in enumerate(labels):
        dict = {}  # dictionary to store participant information
        lbl_users_lst = []  # list of participants per label
        labelsdict[label] = i
        labeldir = os.path.join(rootdir, label)

        # list of users for the respective label
        lbl_usrs = os.walk(labeldir)

        # enumerating all the users of the respective label
        for usr in lbl_usrs:
            # print(usr)
            lbl_users_lst = usr[1]

            # assigning unique ids to all the users
            for i, user in enumerate(lbl_users_lst):
                if user not in user_map:
                    id = user_ids.pop()
                    user_map[user] = id
                    user_list.append(id)
            break

        # extracting data file list for every  participant
        for usr in lbl_users_lst:
            usrdir = os.path.join(labeldir, usr)
            filewalk = os.walk(usrdir)
            file_list = []
            for fls in filewalk:
                file_list = fls[2]
                break
            dict[usr] = (usrdir, file_list)

        dict['users'] = lbl_users_lst
        data_dict[label] = dict  # add all meta information to data_dict

    # Extracting data from the data files from all participants
    for key, value in data_dict.items():
        tar_val = int(key)
        users = value['users']
        for user in users:
            user_dir = value[user]
            dirPath = user_dir[0]
            filelist = user_dir[1]
            for file in filelist:
                fp = os.path.join(dirPath, file)

                data_path.append(fp)

                fileData = read_json_file(fp)
                # extract data from the dictionary
                # emg
                emg = np.array(fileData['emg']['data'])
                emgts = np.array(fileData['emg']['timestamps'])

                # accelerometer
                acc = np.array(fileData['acc']['data'])
                accts = np.array(fileData['acc']['timestamps'])

                # gyroscope
                gyr = np.array(fileData['gyr']['data'])
                gyrts = np.array(fileData['gyr']['timestamps'])

                # orientation
                ori = np.array(fileData['ori']['data'])
                orits = np.array(fileData['ori']['timestamps'])

                # create training instance
                ti = tri.TrainingInstance(key, emg, acc, gyr, ori, emgts, accts,
                                          gyrts, orits)

                # add length for resampling later to the sample length vector
                sample_len_vec_emg.append(emg.shape[0])
                sample_len_vec_others.append(acc.shape[0])

                # split raw data
                ti.separateRawData()
                ti.consolidateData(None, False, True)
                # append training instance to data list
                data.append(ti)

                # append class label to target list
                target.append(tar_val)

                if codebook.has_key(key): codebook[key].append(
                    ti.getConsolidatedDataMatrix())
                else: codebook[key] = [ti.getConsolidatedDataMatrix()]


    avg_len_emg = int(np.mean(sample_len_vec_emg))
    avg_len_acc = int(np.mean(sample_len_vec_others))
    max_length_emg = np.amax(sample_len_vec_emg)
    max_length_others = np.amax(sample_len_vec_others)
    return labels, data, target, labelsdict, avg_len_emg, avg_len_acc, \
           user_map, user_list, data_dict, max_length_emg, max_length_others,\
           data_path, codebook


def normalizeTrainingData(data, max_length_emg, max_len_others):
    '''
    Method to normalize the training data to fixed length
    :param data: (list) List of all the training instance objects
    :param max_length_emg: (int) Normalized length for EMG signals
    :param max_len_others: (int) Normalized length of IMU signals
    :return: data (list) List of all the normalized training instance objects
    '''
    # data = np.array([ti.normalizeData(max_length_emg,max_len_others) for ti
    #  in data])
    d = []
    for i, ti in enumerate(data):
        if i % 50 is 0:
            print(str(i) + ' out of ' + str(
                len(data)) + 'training instances normalized')
        if i is 2 or i is 4:
            print('case')
        d.append(ti.normalizeData(max_length_emg, max_len_others))
    return np.array(d)


def resampleTrainingData(data, sampling_rate, avg_len, emg=True, imu=True):
    '''
    Method to resample the training instances to a given sampling frequency
    in HZ.
    It calls consolidate data implicitly.
    Can remove the consolidation to a different method.
    :param data: (list) List of all the training instance objects
    :param sampling_rate: (int) The new sampling rate in Hz
    :param avg_len: (int) Average length of vectors in case both EMG and IMU
    needs to be resampled and consolidated
    :param emg: (boolean) Flag to indicate that we need to consider emg
    signals for consolidating the data after resampling
    :param imu: (boolean) Flag to indicate that we need to consider IMU
    signals for consolidating the data after resampling
    :return: data : resampled data
    '''
    # data = np.array([ti.resampleData(sampling_rate,avg_len,emg,imu) for ti
    # in data])
    d = []
    for i, ti in enumerate(data):
        if i % 50 is 0:
            print(str(i) + ' out of ' + str(
                len(data)) + 'training instances normalized')
        d.append(ti.resampleData(sampling_rate, avg_len, emg, imu))
    return np.array(d)


def extractFeatures(data, scaler=None, window=True, rms=False, f_mfcc=False,
                    emg=True, imu=True):
    '''
    method to loop through all training instances and extract features from
    the signals
    @params: data (list)                : list of training instances
             scaler (sclaer object)     : scaler object to scale features if
             necessary
             window (Boolean)           : To get overlapping window features
             rms (Boolean)              : To get features from the rms value
             of the signals in all directions
             f_mfcc (Boolean)           :  to get the MFCC features as well
    @return: data (list)                : list of training instances with
    extracted features
    '''
    # data = np.array([ti.extractFeatures(window,scaler,rms,f_mfcc,emg,
    # imu) for ti in data])
    d = []
    for i, ti in enumerate(data):
        if i % 20 is 0:
            print('features extracted from ' + str(i) + ' out of ' + str(
                len(data)) + 'training instances')
        d.append(ti.extractFeatures(window, scaler, rms, f_mfcc, emg, imu))
    return np.array(d)


'''
def prepareDataPC(target, data):
    consolidate = zip(target,data)
    for lbl,d in consolidate:

        con_mat = d.getConsolidatedFeatureMatrix()
        if train_x is None:
            train_x = con_mat
        else:
            train_x = np.append(train_x,con_mat,axis=0)
        train_y.append(int(key))


    return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)
'''


def prepareTrainingDataSvm(trainingIndexes, testingIndexes, target, data):
    train_x = None  # training data
    train_y = []  # training labels

    test_x = None  # testing data
    test_y = []  # testing labels

    for tid in trainingIndexes:
        key = target[tid]
        ti = data[tid]
        con_mat = ti.getConsolidatedFeatureMatrix()
        if train_x is None:
            train_x = con_mat
        else:
            train_x = np.append(train_x, con_mat, axis=0)
        train_y.append(int(key))

    for tid in testingIndexes:
        key = target[tid]
        ti = data[tid]
        con_mat = ti.getConsolidatedFeatureMatrix()
        if test_x is None:
            test_x = con_mat
        else:
            test_x = np.append(test_x, con_mat, axis=0)
        test_y.append(int(key))

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(
        test_y)


def prepareTrainingDataHmmFeatures(trainingIndexes, target, data):
    trainingData = {}
    for l, tid in enumerate(trainingIndexes):

        # printing corrent status so that the wait is not too boring :-P
        if l % 50 is 0:
            print(str(l) + ' out of ' + str(
                len(trainingIndexes)) + 'training instances prepared')

        key = target[tid]
        ti = data[tid]
        # con_data = ti.getConsolidatedDataMatrix()
        if key in trainingData:

            # get data from existing dictionary
            trld = trainingData.get(key)
            lbl_data = trld.get('data')
            n_data = trld.get('datal')
            # extract data from the training instance

            # get consolidated data matrix
            con_mat = ti.getConsolidatedFeatureMatrix()

            # append
            lbl_data = np.append(lbl_data, con_mat, axis=0)
            n_data.append(con_mat.shape[0])

            # replace in the existing dict
            trld['data'] = lbl_data
            trld['datal'] = n_data

            trainingData[key] = trld

        else:
            trld = {}
            # extract others and get features for creating an svm model
            con_mat = ti.getConsolidatedFeatureMatrix()

            trld['data'] = con_mat
            trld['datal'] = [con_mat.shape[0]]

            trainingData[key] = trld

    return trainingData


def discritizeLabels(target):
    n_classes = np.unique(target)
    d_labels = []
    for t in target:
        d_l = np.zeros(n_classes.size, dtype=np.int)
        d_l[t] = 1
        d_labels.append(d_l)
    return np.array(d_labels)


def splitDataset(train, test, target, data):
    train_x = np.take(data, train, axis=0)
    train_y = np.take(target, train, axis=0)

    val_x = np.take(data, test, axis=0)
    val_y = np.take(target, test, axis=0)

    return train_x, train_y, val_x, val_y


def prepareDataset(data):
    d = []
    for i, ti in enumerate(data):
        if i % 20 is 0:
            print(
            'prepared ' + str(i) + ' out of ' + str(len(data)) + 'instances')
        d.append(ti.consolidatedDataMatrix)
    return np.array(d)


def prepareTrainingDataHmmRaw(trainingIndexes, target, data):
    trainingData = {}
    for l, tid in enumerate(trainingIndexes):
        # printing corrent status so that the wait is not too boring :-P
        if l % 50 is 0:
            print(str(l) + ' out of ' + str(
                len(trainingIndexes)) + 'training instances prepared')

        key = target[tid]
        ti = data[tid]
        # con_data = ti.getConsolidatedDataMatrix()
        if key in trainingData:

            # get data from existing dictionary
            trld = trainingData.get(key)
            lbl_data = trld.get('data')
            n_data = trld.get('datal')
            # extract data from the training instance

            # get consolidated data matrix
            con_mat = ti.getConsolidatedDataMatrix()

            # append
            lbl_data = np.append(lbl_data, con_mat, axis=0)
            n_data.append(con_mat.shape[0])

            # replace in the existing dict
            trld['data'] = lbl_data
            trld['datal'] = n_data

            trainingData[key] = trld

        else:
            trld = {}
            # extract others and get features for creating an svm model
            con_mat = ti.getConsolidatedDataMatrix()

            trld['data'] = con_mat
            trld['datal'] = [con_mat.shape[0]]

            trainingData[key] = trld

    return trainingData


def prepareTrainingData(trainingIndexes, target, data):
    # dictionary that holds all the consolidated training data
    trainingDict = {}

    for tid in trainingIndexes:
        key = target[tid]
        ti = data[tid]
        # call separate raw data to create models for the others but for now
        # use raw data
        if key in trainingDict:

            # get data from existing dictionary
            trld = trainingDict.get(key)
            emg = trld.get('emg')
            emgl = trld.get('emgl')

            acc = trld.get('acc')
            accl = trld.get('accl')

            gyr = trld.get('gyr')
            gyrl = trld.get('gyrl')

            ori = trld.get('ori')
            oril = trld.get('oril')

            # extract data from the training instance
            emg_t, acc_t, gyr_t, ori_t = ti.getRawData()

            # append
            emg = np.append(emg, emg_t, axis=0)
            emgl.append(len(emg_t))

            acc = np.append(acc, acc_t, axis=0)
            accl.append(len(acc_t))

            gyr = np.append(gyr, gyr_t, axis=0)
            gyrl.append(len(gyr_t))

            ori = np.append(ori, ori_t, axis=0)
            oril.append(len(ori_t))

            # replace in the existing dict
            trld['emg'] = emg
            trld['emgl'] = emgl

            trld['acc'] = acc
            trld['accl'] = accl

            trld['gyr'] = gyr
            trld['gyrl'] = gyrl

            trld['ori'] = ori
            trld['oril'] = oril

            trainingDict[key] = trld

        else:
            trld = {}
            # extract others and get features for creating an svm model
            emg_t, acc_t, gyr_t, ori_t = ti.getRawData()

            trld['emg'] = emg_t
            trld['emgl'] = [len(emg_t)]

            trld['acc'] = acc_t
            trld['accl'] = [len(acc_t)]

            trld['gyr'] = gyr_t
            trld['gyrl'] = [len(gyr_t)]

            trld['ori'] = ori_t
            trld['oril'] = [len(ori_t)]

            trainingDict[key] = trld

    return trainingDict
