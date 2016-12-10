__author__ = 'dash'

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
                 img_width_range=(20, 1900),
                 word_len=13):
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """

        # img_height = 32
        img_height = 10
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)

        if evaluate:
            self.bucket_specs = [(400 / 16, word_len + 2),
                                 (800 / 16, word_len + 2),
                                 (1200 / 16, word_len + 2),
                                 (1800 / 16, word_len + 2),
                                 (img_width_range[1] / 16, word_len + 2)]
        else:
            self.bucket_specs = [(400 / 16, 4 + 2),
                                 (800 / 16, 6 + 2),
                                 (1200 / 16, 9 + 2),
                                 (1800 / 16, 10 + 2),
                                 (img_width_range[1] / 16, word_len + 2)]
        self.bins = [400, 800, 1200, 1800, img_width_range[1]]
        #for bucket in self.bucket_specs:
        #    self.bins.append(bucket[0])

        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        #self.bucket_data = {i: BucketData()
        #                    for i in range(self.bucket_max_width + 1)}
        self.bucket_data = {i: BucketData()
                                for i in range(len(self.bins))}

    def clear(self):
        self.bucket_data = {i : BucketData()
                            for i in range(len(self.bins))}

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'rb') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                img_path, lex, _, _ = l.strip().split()
                try:
                    img_bw, word = self.read_data_seq(img_path, lex)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-1]

                    # TODO:resize if > 320
                    #b_idx = min(width, self.bucket_max_width)
                    b_idxs = self.get_bin_id([width], self.bins)
                    b_idx = b_idxs[0] #just for debugging purpose.[remove later]
                    img_bw = signal.resample(img_bw, self.bins[b_idx], axis=2)
                    bs = self.bucket_data[b_idx].append(img_bw, word,
                                                        os.path.join(
                                                            self.data_root,
                                                            img_path))
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
                    pass  # ignore error images
                    # with open('error_img.txt', 'a') as ef:
                    #    ef.write(img_path + '\n')
        self.clear()

    def get_bin_id(self, data, bins):
        ids = []
        for d in data:
            for i, b in enumerate(bins):
                if b >= d:
                    ids.append(i)
                    break
        return ids

    def read_data_seq(self, img_path, lex):
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        # with open(os.path.join(self.data_root, img_path), 'rb') as img_file :
        img_file = os.path.join(self.data_root, img_path)
        img = np.load(img_file)
        w, h = img.shape
        # aspect_ratio = float(w) / float(h)
        if w < self.bucket_min_width:
            img = signal.resample(img, self.bucket_min_width)

        elif w > self.bucket_max_width:
            img = signal.resample(img, self.bucket_max_width)

        elif h != self.image_height:
            raise Exception('Invalid number of channels')

        img_bw = img.transpose()
        img_bw = np.asarray(img_bw, dtype=np.float32)
        img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        for c in lex:
            assert 96 < ord(c) < 123 or 47 < ord(c) < 58
            word.append(
                ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3)
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word

    def read_data(self, img_path, lex):
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.float32)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        for c in lex:
            assert 96 < ord(c) < 123 or 47 < ord(c) < 58
            word.append(
                ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3)
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word


def test_gen():
    print 'testing gen_valid'
    # s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
    s_gen = DataGen('../../data/voice/v005_hist_1_10_1000000/training/data',
                    '../../data/voice/v005_hist_1_10_1000000/training/dataset'
                    '.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar13', 'test.txt')
    count = 0
    for batch in s_gen.gen(10) :
        count += 1
        # print batch['bucket_id'], batch['data'].shape[2:]
        #print batch
        print batch['bucket_id'], batch['data'].shape[2 :]
        print count


if __name__ == '__main__':
    test_gen()
