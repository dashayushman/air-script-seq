import argparse, logging, sys
from utils import dataprep as dp
from datasource.data import DataSet


def process_args(args):
    '''
    Parses the commandline arguments
    :param args: the commandline arguments
    :return: parsed args
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
                        type=str, default='of_model', help=('Name of the '
                                                            'dataset'))

    parser.add_argument('--root_dir',
                        type=str, default='/tmp', help=('Root directory of '
                                                        'the raw data'))
    parser.add_argument('--log_dir',
                        type=str, default='/tmp', help=('Directory to dump '
                                                        'log files'))

    parser.add_argument('--min_len', type=int, default=1,
                        help=('Min sequence length'))

    parser.add_argument('--max_len', type=int, default=5,
                        help=('max sequence length'))

    parser.add_argument('--n_seq', type=int, default=500, help=('Number of '
                                                                'sequences to '
                                                                'generate'))

    parser.add_argument('--output_dir',
                        type=str, default='/tmp', help=('directory location '
                                                        'to output the '
                                                        'training and testing '
                                                        'instances'))

    parser.add_argument('--train_size_ratio',
                        type=float, default=0.7, help=('proportion of the '
                                                       'total number of '
                                                       'sequences to keep '
                                                       'for training and the '
                                                       'rest will be for '
                                                       'testing. For eg, 0.7'))
    parameters = parser.parse_args(args)
    return parameters


def generator(args):
    '''
    Generates random data sequences
    :param args: commandline args
    :return:
    '''

    # Parsed commandline args
    parameters = process_args(args)

    # Some boilerplate code for logging and stuff
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_dir)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    labels, data, target, labelsdict, avg_len_emg, avg_len_acc, user_map, \
    user_list, data_dict, max_length_emg, max_length_others, data_path, \
    codebook = dp.getTrainingData(parameters.root_dir)

    # Number of instances per sequence length
    n_instances_per_seq_len = (parameters.n_seq / (parameters.max_len -
                                                   parameters.min_len))

    label_seqs, label_seq_lengths = dp.generate_label_sequences(labels,
                                                                n_instances_per_seq_len,
                                                                range=(parameters.min_len,
                                                                    parameters.max_len))

    data_seqs, data_seq_lengths = dp.generate_data_sequences(codebook,
                                                             label_seqs,
                                                             n_instances_per_seq_len,
                                                             range=(parameters.min_len,
                                                                 parameters.max_len))

    train_label_seqs = dp.generate_label_sequences()
    test_label_seqs = dp.generate_label_sequences()


if __name__ == '__main__':
    generator(sys.argv[1:])
