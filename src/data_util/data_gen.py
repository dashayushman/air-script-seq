__author__ = 'Ayushman Dash'

import os
import numpy as np
from PIL import Image
import random
from bucketdata import BucketData
from scipy import signal


class DataGen(object):
    GO = 1
    EOS = 2

    def __init__(self,
                 data_root, annotation_fn,
                 evaluate=False,
                 valid_target_len=float('inf'),
                 seq_width_range=(20, 1900),
                 word_len=13):
        """
        This initializes the data generator object
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param signal_width_range: only needed for training set
        :return:
        """
        seq_features = 10
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)

        if evaluate:
            self.bucket_specs = [(240 / 16, word_len + 2),
                                 (400 / 16, word_len + 2),
                                 (600 / 16, word_len + 2),
                                 (800 / 16, word_len + 2),
                                 (1000 / 16, word_len + 2),
                                 (1200 / 16, word_len + 2),
                                 (1800 / 16, word_len + 2),
                                 (seq_width_range[1] / 16, word_len + 2)]
        else:
            self.bucket_specs = [(230 / 16, 2 + 2),
                                 (400 / 16, 4 + 2),
                                 (600 / 16, 6 + 2),
                                 (800 / 16, 8 + 2),
                                 (1000 / 16, 8 + 2),
                                 (1200 / 16, 9 + 2),
                                 (1800 / 16, 10 + 2),
                                 (seq_width_range[1] / 16, word_len + 2)]
        self.bins = [230, 400, 600, 800, 1000, 1200, 1800, seq_width_range[1]]

        self.bucket_min_width, self.bucket_max_width = seq_width_range
        self.seq_features = seq_features
        self.valid_target_len = valid_target_len

        self.bucket_data = {i: BucketData()
                                for i in range(len(self.bins))}

    def clear(self):
        self.bucket_data = {i : BucketData()
                            for i in range(len(self.bins))}

    def gen(self, batch_size):
        '''
        The batch generator that yields a batch of data instances and their
        corresponding metadata
        :param batch_size: batch size
        :return: A batch of sequences
        '''
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'rb') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                seq_path, lex, _, _ = l.strip().split()
                try:
                    norm_seq, word = self.read_data_seq(seq_path, lex)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = norm_seq.shape[-1]

                    # TODO:resize if > 320
                    b_idxs = self.get_bin_id([width], self.bins)
                    b_idx = b_idxs[0] #just for debugging purpose.[remove later]
                    norm_seq = signal.resample(norm_seq, self.bins[b_idx], axis=2)
                    bs = self.bucket_data[b_idx].append(norm_seq, word,
                                                        os.path.join(
                                                            self.data_root,
                                                            seq_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                            self.bucket_specs,
                            valid_target_length=valid_target_len,
                            go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d' % width
                except IOError:
                    pass

        self.clear()

    def get_bin_id(self, data, bins):
        '''

        :param data:
        :param bins:
        :return:
        '''
        ids = []
        for d in data:
            for i, b in enumerate(bins):
                if b >= d:
                    ids.append(i)
                    break
        return ids

    def read_data_seq(self, seq_path, lex):
        '''
        This reads the data file and normalizes it to create batches
        :param seq_path: Path to the sequence file
        :param lex: The corresponding output sequence string
        :return: Input and output sequence
        '''
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        seq_file = os.path.join(self.data_root, seq_path)
        seq = np.load(seq_file)
        w, h = seq.shape
        if w < self.bucket_min_width:
            seq = signal.resample(seq, self.bucket_min_width)

        elif w > self.bucket_max_width:
            seq = signal.resample(seq, self.bucket_max_width)

        elif h != self.seq_features:
            raise Exception('Invalid number of channels')

        norm_seq = seq.transpose()
        norm_seq = np.asarray(norm_seq, dtype=np.float32)
        norm_seq = norm_seq[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        for c in lex:
            assert 96 < ord(c) < 123 or 47 < ord(c) < 58
            word.append(
                ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3)
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)

        return norm_seq, word

def test_gen():
    print 'testing gen_valid'
    s_gen = DataGen('../../data/voice/v005_hist_1_10_1000000/training/data',
                    '../../data/voice/v005_hist_1_10_1000000/training/dataset'
                    '.txt')
    count = 0
    for batch in s_gen.gen(10) :
        count += 1
        print batch['bucket_id'], batch['data'].shape[2 :]
        print count


if __name__ == '__main__':
    test_gen()
