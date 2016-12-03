import argparse, logging, sys, os
from utils import dataprep as dp
from datasource.data import DataSet
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def process_args(args) :
    '''
    Parses the commandline arguments
    :param args: the commandline arguments
    :return: parsed args
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
                        type = str, default = 'of_model',
                        help = ('Name of the '
                                'dataset'))

    parser.add_argument('--root_dir',
                        type = str, default = '/tmp',
                        help = ('Root directory of '
                                'the raw data'))
    parser.add_argument('--log_dir',
                        type = str, default = '/tmp',
                        help = ('Directory to dump '
                                'log files'))

    parser.add_argument('--min_len', type = int, default = 1,
                        help = ('Min sequence length'))

    parser.add_argument('--max_len', type = int, default = 5,
                        help = ('max sequence length'))

    parser.add_argument('--n_seq', type = int, default = 500,
                        help = ('Number of '
                                'sequences to '
                                'generate'))

    parser.add_argument('--output_dir',
                        type = str, default = '/tmp',
                        help = ('directory location '
                                'to output the '
                                'training and testing '
                                'instances'))

    parser.add_argument('--train_size',
                        type = float, default = 0.7,
                        help = ('proportion of the '
                                'total number of '
                                'sequences to keep '
                                'for training and the '
                                'rest will be for '
                                'testing. For eg, 0.7'))
    parameters = parser.parse_args(args)
    return parameters


def initialize_logger(log_dir) :
    '''
    This initializes the logger
    :param log_dir: Path to the log file
    :return:
    '''
    logging.basicConfig(
        level = logging.DEBUG,
        format = '%(asctime)-15s %(name)-5s %(levelname)-8s %('
                 'message)s',
        filename = log_dir)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def dump_sequences(dir, data_seqs, lbl_seqs) :
    '''
    Dumps the data sequences into a
    :param dir: dir to dump data
    :param data_seqs: data sequences
    :param lbl_seqs: label sequences
    :return: satus (True / False)
    '''
    #create directory if does not exist
    data_dir = os.path.join(dir, 'data')
    file_list_f = os.path.join(dir, 'meta.txt')
    if not os.path.exists(data_dir) :
        os.makedirs(data_dir)

    file_list = []
    for i, (lbl, data) in enumerate(zip(lbl_seqs, data_seqs)):
        lbl_str = ''.join(lbl)
        data_file_name = str(i) + '_' + lbl_str
        if i % 50 == 0 :
            logging.info('dumping ' + ''.join(lbl) + ' into ' + data_dir)

        np.save(os.path.join(data_dir, data_file_name), data)

        with open(file_list_f, "a") as myfile :
            myfile.write(data_file_name + '.npy ' + lbl_str + '\n')

    return False


def generator(args) :
    '''
    Generates random data sequences
    :param args: commandline args
    :return:
    '''

    # Parsed commandline args
    parameters = process_args(args)

    # Some boilerplate code for logging and stuff
    initialize_logger(parameters.log_dir)

    labels, data, target, labelsdict, avg_len_emg, avg_len_acc, \
    user_map, user_list, data_dict, max_length_emg, max_length_others, \
    data_path, codebook = dp.getTrainingData(parameters.root_dir)

    # Number of instances per sequence length
    n_instances_per_seq_len = (parameters.n_seq / (parameters.max_len -
                                                   parameters.min_len + 1))

    label_seqs, label_seq_lengths = dp.generate_label_sequences(labels,
                                                                n_instances_per_seq_len,
                                                                l_range = (parameters.min_len,
                                                                    parameters.max_len))

    data_seqs, t_data_seq = dp.generate_data_sequences(codebook,
                                                       label_seqs,
                                                       l_range = (parameters.min_len,
                                                           parameters.max_len))
    sss = StratifiedShuffleSplit(n_splits = 1,
                                 #test_size = (1 - parameters.train_size),
                                 train_size = parameters.train_size,
                                 random_state = 0)

    train_idx = []
    test_idx = []

    for train_index, test_index in sss.split(np.zeros(len(label_seq_lengths)),
                                             label_seq_lengths) :
        train_idx = train_index
        test_idx = test_index

    train_x, train_y, test_x, test_y = dp.splitDataset(train_idx, test_idx,
                                                       label_seqs, data_seqs)

    dataset_root_dir = os.path.join(parameters.output_dir, parameters.name)
    #dump training data
    dump_sequences(os.path.join(dataset_root_dir, 'training'),
                   train_x,
                   train_y)

    # dump testing data
    dump_sequences(os.path.join(dataset_root_dir, 'testing'),
                   train_x,
                   train_y)

    logging.info('Finished generating dataset ' + parameters.name)

if __name__ == '__main__' :
    generator(sys.argv[1 :])
