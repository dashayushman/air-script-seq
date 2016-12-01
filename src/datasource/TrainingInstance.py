import numpy as np
from scipy import signal
from utils import feature_extractor as fe
from utils import utility as util
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
from sklearn import preprocessing


class TrainingInstance:
    scaler = preprocessing.MinMaxScaler()

    def __init__(self, label, emg, acc, gyr, ori, emgts=None, accts=None,
                 gyrts=None, orits=None):

        self.m_label = label
        # raw data
        self.emg = emg
        self.acc = acc
        self.gyr = gyr
        self.ori = ori

        # time stamps
        self.emgts = emgts
        self.accts = accts
        self.gyrts = gyrts
        self.orits = orits

        self.sr_emg = 200
        self.sr_other = 50

        # splitted flag
        self.splitted = False
        self.consolidated = False
        self.consolidatedFeatures = False

    def separateRawData(self):
        if self.emg is not None:
            self.emgList = np.array(
                [np.array(self.emg[:, 0]), np.array(self.emg[:, 1]),
                 np.array(self.emg[:, 2]), np.array(self.emg[:, 3]),
                 np.array(self.emg[:, 4]), np.array(self.emg[:, 5]),
                 np.array(self.emg[:, 6]), np.array(self.emg[:, 7])])

        if self.acc is not None:
            self.accList = np.array(
                [np.array(self.acc[:, 0]), np.array(self.acc[:, 1]),
                 np.array(self.acc[:, 2])])

        if self.gyr is not None:
            self.gyrList = np.array(
                [np.array(self.gyr[:, 0]), np.array(self.gyr[:, 1]),
                 np.array(self.gyr[:, 2])])

        if self.ori is not None:
            self.oriList = np.array(
                [np.array(self.ori[:, 0]), np.array(self.ori[:, 1]),
                 np.array(self.ori[:, 2]), np.array(self.ori[:, 3])])

        self.splitted = True

    # scale data
    def scaleData(self, scaler):
        if self.splitted == True:
            norm_emgs = []
            norm_accs = []
            norm_gyrs = []
            norm_oris = []

            for x in self.emgList:
                x = x.reshape(-1, 1)
                x = scaler.fit_transform(x)
                reshaped = x.reshape(x.shape[0])
                norm_emgs.append(reshaped)

            for a, b in zip(self.accList, self.gyrList):
                a = a.reshape(-1, 1)
                a = scaler.fit_transform(a)
                reshaped_a = a.reshape(a.shape[0])
                norm_accs.append(reshaped_a)
                b = b.reshape(-1, 1)
                b = scaler.fit_transform(b)
                reshaped_b = b.reshape(a.shape[0])
                norm_gyrs.append(reshaped_b)

            for x in self.oriList:
                x = x.reshape(-1, 1)
                x = scaler.fit_transform(x)
                reshaped = x.reshape(x.shape[0])
                norm_oris.append(reshaped)

            self.emgList = np.array(norm_emgs)
            self.accList = np.array(norm_accs)
            self.gyrList = np.array(norm_gyrs)
            self.oriList = np.array(norm_oris)
        return self

    # normalize data to common length
    def normalizeData(self, max_len_emg, max_len_others):
        if self.splitted == True:
            norm_emgs = []
            norm_accs = []
            norm_gyrs = []
            norm_oris = []

            for x in self.emgList:
                if (x.shape[0] == max_len_emg):
                    norm_emgs.append(x)
                    continue
                if (x.shape[0] < max_len_emg):
                    half = (float(max_len_emg - x.shape[0])) / 2
                    back = ceil(half)
                    front = floor(half)
                    norm_emgs.append(util.padVector(x, front, back, True))

            for a, b in zip(self.accList, self.gyrList):
                if (a.shape == max_len_others):
                    norm_accs.append(a)
                    norm_gyrs.append(b)
                    continue
                if (a.shape[0] < max_len_others):
                    half_a = (float(max_len_others - a.shape[0])) / 2
                    back_a = ceil(half_a)
                    front_a = floor(half_a)

                    half_b = (float(max_len_others - b.shape[0])) / 2
                    back_b = ceil(half_b)
                    front_b = floor(half_b)

                    norm_accs.append(util.padVector(a, front_a, back_a))
                    norm_gyrs.append(util.padVector(b, front_b, back_b))

            for x in self.oriList:
                if (x.shape[0] == max_len_others):
                    norm_oris.append(x)
                    continue
                if (x.shape[0] < max_len_others):
                    half = (float(max_len_others - x.shape[0])) / 2
                    back = ceil(half)
                    front = floor(half)
                    norm_oris.append(util.padVector(x, front, back))

            '''
            # Four axes, returned as a 2-d array
            f, axarr = plt.subplots(2, 2)
            axarr[0, 0].plot(np.arange(len(self.emgList[0])),self.emgList[0])
            axarr[0, 0].set_title('Raw EMG')
            axarr[0, 1].plot(np.arange(len(norm_emgs[0])),norm_emgs[0])
            axarr[0, 1].set_title('Normalized Emg')
            axarr[1, 0].plot(np.arange(len(self.accList[1])),self.accList[1])
            axarr[1, 0].set_title('Raw ACC X')
            axarr[1, 1].plot(np.arange(len(norm_accs[1])),norm_accs[1])
            axarr[1, 1].set_title('Normalized ACC X')
            plt.show()
            '''
            self.emgList = np.array(norm_emgs)
            self.accList = np.array(norm_accs)
            self.gyrList = np.array(norm_gyrs)
            self.oriList = np.array(norm_oris)
        return self

    def resampleData(self, sr, avg_len, emg=True, imu=True):
        '''
        Method for resampling the all the signals and bringing them to the
        same sampling rate
        :param sr: (int): sampling rate
        :return: self with all resampled data
        '''
        if self.splitted == True:

            # Calculate the new length of vectors given the new sampling
            # frequency/rate

            sample_len_emg = int((sr * self.emgList[0].size) / self.sr_emg)
            sample_len_emg_others = int(
                (sr * self.accList[0].size) / self.sr_other)

            self.sr_emg = sr
            self.sr_other = sr
            '''
            self.emgList_r = self.emgList
            self.accList_r = self.accList
            self.gyrList_r = self.gyrList
            self.oriList_r = self.oriList
            '''
            # resampling the normalized data
            self.emgList = np.array(
                [signal.resample(x, sample_len_emg) for x in self.emgList])
            self.accList = np.array(
                [signal.resample(x, sample_len_emg_others) for x in
                 self.accList])
            self.gyrList = np.array(
                [signal.resample(x, sample_len_emg_others) for x in
                 self.gyrList])
            self.oriList = np.array(
                [signal.resample(x, sample_len_emg_others) for x in
                 self.oriList])

            self.consolidateData(avg_len, emg, imu)
        return self

    def extractFeatures(self, window=True, scaler=None, rms=False, f_mfcc=False,
                        emg=True, imu=True):
        '''
        This method extracts features from the training instance and
        consolidates into one meature matrix according to the parameters
        provided
        :param window: (Boolean)                            : To get
        overlapping windowed features
        :param scaler: (Scaler Object as in scikit-learn)   : Scalar object
        to scale the features
        :param rms: (Boolean)                               : To extract
        features from the Root Mean Square of the signals in all dimensions
        :param f_mfcc: (Boolean)                            : To extract MFCC
        features
        :param emg: (Boolean)                               : To extract
        features from EMG signals
        :param imu: (Boolean)                               : To extract
        features from IMU signals
        :return: self
        '''
        # print(self.m_label)
        if self.splitted == True:
            # For RMS
            if rms:
                all_emg = zip(self.emgList[0], self.emgList[1], self.emgList[2],
                              self.emgList[3], self.emgList[4], self.emgList[5],
                              self.emgList[6], self.emgList[7])
                all_acc = zip(self.accList[0], self.accList[1], self.accList[2])
                all_gyr = zip(self.gyrList[0], self.gyrList[1], self.gyrList[2])
                all_ori = zip(self.oriList[0], self.oriList[1], self.oriList[2],
                              self.oriList[3])

                rms_emg = []
                rms_acc = []
                rms_gyr = []
                rms_ori = []

                # calculating RMS for all the signals
                for _0, _1, _2, _3, _4, _5, _6, _7 in all_emg:
                    vec = [_0, _1, _2, _3, _4, _5, _6, _7]
                    rms_val = sqrt(sum(n * n for n in vec) / len(vec))
                    rms_emg.append(rms_val)
                for _0, _1, _2 in all_acc:
                    vec = [_0, _1, _2]
                    rms_val = sqrt(sum(n * n for n in vec) / len(vec))
                    rms_acc.append(rms_val)
                for _0, _1, _2 in all_gyr:
                    vec = [_0, _1, _2]
                    rms_val = sqrt(sum(n * n for n in vec) / len(vec))
                    rms_gyr.append(rms_val)
                for _0, _1, _2, _3 in all_ori:
                    vec = [_0, _1, _2, _3]
                    rms_val = sqrt(sum(n * n for n in vec) / len(vec))
                    rms_ori.append(rms_val)

                # Extracting features for all the signals
                self.emgRmsFeatures = fe.getFeatures(rms_emg, self.sr_emg,
                                                     window, f_mfcc)
                self.accRmsFeatures = fe.getFeatures(rms_acc, self.sr_other,
                                                     window, f_mfcc)
                self.gyrRmsFeatures = fe.getFeatures(rms_gyr, self.sr_other,
                                                     window, f_mfcc)
                self.oriRmsFeatures = fe.getFeatures(rms_ori, self.sr_other,
                                                     window, f_mfcc)

            # for extracting features from raw data
            else:
                self.emgFeatures = np.array(
                    [fe.getFeatures(x, self.sr_emg, window, f_mfcc) for x in
                     self.emgList])
                self.accFeatures = np.array(
                    [fe.getFeatures(x, self.sr_other, window, f_mfcc) for x in
                     self.accList])
                self.gyrFeatures = np.array(
                    [fe.getFeatures(x, self.sr_other, window, f_mfcc) for x in
                     self.gyrList])
                self.oriFeatures = np.array(
                    [fe.getFeatures(x, self.sr_other, window, f_mfcc) for x in
                     self.oriList])

            self.consolidateFeatures(scaler, rms, emg, imu)
        return self

    def consolidateFeatures(self, scaler=None, rms=False, emg=True, imu=True):
        '''
        Method to consolidate the features of all the sensor data in all
        dimensions to a single feature matrix
        :param scaler: (Scaler Object)      : A scaler object to scale the
        features
        :param rms: (Boolean)               : Flag for consolidating RMS
        features
        :param emg: (Boolean)               : Flas to consider features from
        EMG signals
        :param imu: (Boolean)               : Flag to consider IMU Signals
        :return: consolidated_feature_Matrix (ndarray) : with columns as
        features and rows as overlapping window frames. If window was false
        then it just has one row.
        '''
        if self.splitted == True:
            con_emg_feat = None
            con_acc_feat = None
            con_gyr_feat = None
            con_ori_feat = None
            consolidatedFeatureMatrix = None
            if rms:
                if emg:
                    con_emg_feat = self.emgRmsFeatures
                if imu:
                    con_acc_feat = self.accRmsFeatures
                    con_gyr_feat = self.gyrRmsFeatures
                    con_ori_feat = self.oriRmsFeatures
            else:
                if emg:
                    n_emg_rows = self.emgFeatures[0].shape[0]
                    n_emg_columns = self.emgFeatures[0].shape[1]
                    new_n_emg_columns = self.emgFeatures.shape[
                                            0] * n_emg_columns
                if imu:
                    n_acc_rows = self.accFeatures[0].shape[0]
                    n_acc_columns = self.accFeatures[0].shape[1]
                    new_n_acc_columns = self.accFeatures.shape[
                                            0] * n_acc_columns

                    n_gyr_rows = self.gyrFeatures[0].shape[0]
                    n_gyr_columns = self.gyrFeatures[0].shape[1]
                    new_n_gyr_columns = self.gyrFeatures.shape[
                                            0] * n_gyr_columns

                    n_ori_rows = self.oriFeatures[0].shape[0]
                    n_ori_columns = self.oriFeatures[0].shape[1]
                    new_n_ori_columns = self.oriFeatures.shape[
                                            0] * n_ori_columns

                if emg:
                    con_emg_feat = np.reshape(self.emgFeatures,
                                              (n_emg_rows, new_n_emg_columns))
                if imu:
                    con_acc_feat = np.reshape(self.accFeatures,
                                              (n_acc_rows, new_n_acc_columns))
                    con_gyr_feat = np.reshape(self.gyrFeatures,
                                              (n_gyr_rows, new_n_gyr_columns))
                    con_ori_feat = np.reshape(self.oriFeatures,
                                              (n_ori_rows, new_n_ori_columns))
            if emg and imu:
                consolidatedFeatureMatrix = np.concatenate(
                    (con_emg_feat, con_acc_feat), axis=1)
                consolidatedFeatureMatrix = np.concatenate(
                    (consolidatedFeatureMatrix, con_gyr_feat), axis=1)
                consolidatedFeatureMatrix = np.concatenate(
                    (consolidatedFeatureMatrix, con_ori_feat), axis=1)
            elif emg and (not imu):
                consolidatedFeatureMatrix = con_emg_feat
            elif (not emg) and imu:
                consolidatedFeatureMatrix = con_acc_feat
                consolidatedFeatureMatrix = np.concatenate(
                    (consolidatedFeatureMatrix, con_gyr_feat), axis=1)
                consolidatedFeatureMatrix = np.concatenate(
                    (consolidatedFeatureMatrix, con_ori_feat), axis=1)
            else:
                return None
            '''
            consolidatedFeatureMatrix = np.concatenate((con_emg_feat,
            con_acc_feat), axis=1)
            consolidatedFeatureMatrix = np.concatenate((
            consolidatedFeatureMatrix, con_gyr_feat), axis=1)
            consolidatedFeatureMatrix = np.concatenate((
            consolidatedFeatureMatrix, con_ori_feat), axis=1)
            '''
            self.consolidatedFeatureMatrix = consolidatedFeatureMatrix
            self.consolidatedFeatures = True
            if scaler is not None:
                consolidatedFeatureMatrix = scaler.fit_transform(
                    consolidatedFeatureMatrix)
            return consolidatedFeatureMatrix
        else:
            return None

    def consolidateData(self, avg_len, emg, imu):
        consolidatedDataMatrix = None
        if self.splitted == True:
            if emg and imu:
                emg_r = np.array(
                    [signal.resample(x, avg_len) for x in self.emgList_r])
                acc_r = np.array(
                    [signal.resample(x, avg_len) for x in self.accList_r])
                gyr_r = np.array(
                    [signal.resample(x, avg_len) for x in self.gyrList_r])
                ori_r = np.array(
                    [signal.resample(x, avg_len) for x in self.oriList_r])
                consolidatedDataMatrix = np.concatenate(
                    (emg_r, acc_r, gyr_r, ori_r), axis=0)
            elif emg and (not imu):
                consolidatedDataMatrix = self.emgList
            elif (not emg) and imu:
                consolidatedDataMatrix = np.concatenate(
                    (self.accList, self.gyrList, self.oriList), axis=0)
            else:
                emg_r = np.array(
                    [signal.resample(x, avg_len) for x in self.emgList_r])
                acc_r = np.array(
                    [signal.resample(x, avg_len) for x in self.accList_r])
                gyr_r = np.array(
                    [signal.resample(x, avg_len) for x in self.gyrList_r])
                ori_r = np.array(
                    [signal.resample(x, avg_len) for x in self.oriList_r])
                consolidatedDataMatrix = np.concatenate(
                    (emg_r, acc_r, gyr_r, ori_r), axis=0)
            self.consolidatedDataMatrix = consolidatedDataMatrix.transpose()
            self.consolidated = True
            return consolidatedDataMatrix
        else:
            return None

    def getConsolidatedFeatureMatrix(self):
        if self.consolidatedFeatures:
            return self.consolidatedFeatureMatrix

    def getConsolidatedDataMatrix(self):
        if self.consolidated:
            return self.consolidatedDataMatrix

    def getRawData(self):
        return self.emg, self.acc, self.gyr, self.ori

    def getData(self):
        if self.splitted is True:
            return self.emg, self.acc, self.gyr, self.ori, self.emgList, \
                   self.accList, self.gyrList, self.oriList
        else:
            return self.emg, self.acc, self.gyr, self.ori

    def getIndevidualFeatures(self, meanNormalized=False):
        emg_0_feat = None
        emg_1_feat = None
        emg_2_feat = None
        emg_3_feat = None
        emg_4_feat = None
        emg_5_feat = None
        emg_6_feat = None
        emg_7_feat = None

        acc_x_feat = None
        acc_y_feat = None
        acc_z_feat = None

        gyr_x_feat = None
        gyr_y_feat = None
        gyr_z_feat = None

        ori_x_feat = None
        ori_y_feat = None
        ori_z_feat = None
        ori_w_feat = None

        if self.splitted and self.consolidatedFeatures:
            if meanNormalized:
                for i, feat in enumerate(self.emgFeatures):
                    if i is 0:
                        emg_0_feat = self.scaler.fit_transform(feat)
                        emg_0_feat = np.insert(emg_0_feat, len(emg_0_feat[0]),
                                               self.m_label)
                    elif i is 1:
                        emg_1_feat = self.scaler.fit_transform(feat)
                        emg_1_feat = np.insert(emg_1_feat, len(emg_1_feat[0]),
                                               self.m_label)
                    elif i is 2:
                        emg_2_feat = self.scaler.fit_transform(feat)
                        emg_2_feat = np.insert(emg_2_feat, len(emg_2_feat[0]),
                                               self.m_label)
                    elif i is 3:
                        emg_3_feat = self.scaler.fit_transform(feat)
                        emg_3_feat = np.insert(emg_3_feat, len(emg_3_feat[0]),
                                               self.m_label)
                    elif i is 4:
                        emg_4_feat = self.scaler.fit_transform(feat)
                        emg_4_feat = np.insert(emg_4_feat, len(emg_4_feat[0]),
                                               self.m_label)
                    elif i is 5:
                        emg_5_feat = self.scaler.fit_transform(feat)
                        emg_5_feat = np.insert(emg_5_feat, len(emg_5_feat[0]),
                                               self.m_label)
                    elif i is 6:
                        emg_6_feat = self.scaler.fit_transform(feat)
                        emg_6_feat = np.insert(emg_6_feat, len(emg_6_feat[0]),
                                               self.m_label)
                    elif i is 7:
                        emg_7_feat = self.scaler.fit_transform(feat)
                        emg_7_feat = np.insert(emg_7_feat, len(emg_7_feat[0]),
                                               self.m_label)
                for i, feat in enumerate(self.accFeatures):
                    if i is 0:
                        acc_x_feat = self.scaler.fit_transform(feat)
                        acc_x_feat = np.insert(acc_x_feat, len(acc_x_feat[0]),
                                               self.m_label)
                    elif i is 1:
                        acc_y_feat = self.scaler.fit_transform(feat)
                        acc_y_feat = np.insert(acc_y_feat, len(acc_y_feat[0]),
                                               self.m_label)
                    elif i is 2:
                        acc_z_feat = self.scaler.fit_transform(feat)
                        acc_z_feat = np.insert(acc_z_feat, len(acc_z_feat[0]),
                                               self.m_label)
                for i, feat in enumerate(self.gyrFeatures):
                    if i is 0:
                        gyr_x_feat = self.scaler.fit_transform(feat)
                        gyr_x_feat = np.insert(gyr_x_feat, len(gyr_x_feat[0]),
                                               self.m_label)
                    elif i is 1:
                        gyr_y_feat = self.scaler.fit_transform(feat)
                        gyr_y_feat = np.insert(gyr_y_feat, len(gyr_y_feat[0]),
                                               self.m_label)
                    elif i is 2:
                        gyr_z_feat = self.scaler.fit_transform(feat)
                        gyr_z_feat = np.insert(gyr_z_feat, len(gyr_z_feat[0]),
                                               self.m_label)
                for i, feat in enumerate(self.oriFeatures):
                    if i is 0:
                        ori_x_feat = self.scaler.fit_transform(feat)
                        ori_x_feat = np.insert(ori_x_feat, len(ori_x_feat[0]),
                                               self.m_label)
                    elif i is 1:
                        ori_y_feat = self.scaler.fit_transform(feat)
                        ori_y_feat = np.insert(ori_y_feat, len(ori_y_feat[0]),
                                               self.m_label)
                    elif i is 2:
                        ori_z_feat = self.scaler.fit_transform(feat)
                        ori_z_feat = np.insert(ori_z_feat, len(ori_z_feat[0]),
                                               self.m_label)
                    elif i is 3:
                        ori_w_feat = self.scaler.fit_transform(feat)
                        ori_w_feat = np.insert(ori_w_feat, len(ori_w_feat[0]),
                                               self.m_label)
            else:
                for i, feat in enumerate(self.emgFeatures):
                    if i is 0:
                        emg_0_feat = feat
                        emg_0_feat = np.insert(emg_0_feat, len(emg_0_feat[0]),
                                               self.m_label)
                    elif i is 1:
                        emg_1_feat = feat
                        emg_1_feat = np.insert(emg_1_feat, len(emg_1_feat[0]),
                                               self.m_label)
                    elif i is 2:
                        emg_2_feat = feat
                        emg_2_feat = np.insert(emg_2_feat, len(emg_2_feat[0]),
                                               self.m_label)
                    elif i is 3:
                        emg_3_feat = feat
                        emg_3_feat = np.insert(emg_3_feat, len(emg_3_feat[0]),
                                               self.m_label)
                    elif i is 4:
                        emg_4_feat = feat
                        emg_4_feat = np.insert(emg_4_feat, len(emg_4_feat[0]),
                                               self.m_label)
                    elif i is 5:
                        emg_5_feat = feat
                        emg_5_feat = np.insert(emg_5_feat, len(emg_5_feat[0]),
                                               self.m_label)
                    elif i is 6:
                        emg_6_feat = feat
                        emg_6_feat = np.insert(emg_6_feat, len(emg_6_feat[0]),
                                               self.m_label)
                    elif i is 7:
                        emg_7_feat = feat
                        emg_7_feat = np.insert(emg_7_feat, len(emg_7_feat[0]),
                                               self.m_label)
                for i, feat in enumerate(self.accFeatures):
                    if i is 0:
                        acc_x_feat = feat
                        acc_x_feat = np.insert(acc_x_feat, len(acc_x_feat[0]),
                                               self.m_label)
                    elif i is 1:
                        acc_y_feat = feat
                        acc_y_feat = np.insert(acc_y_feat, len(acc_y_feat[0]),
                                               self.m_label)
                    elif i is 2:
                        acc_z_feat = feat
                        acc_z_feat = np.insert(acc_z_feat, len(acc_z_feat[0]),
                                               self.m_label)
                for i, feat in enumerate(self.gyrFeatures):
                    if i is 0:
                        gyr_x_feat = feat
                        gyr_x_feat = np.insert(gyr_x_feat, len(gyr_x_feat[0]),
                                               self.m_label)
                    elif i is 1:
                        gyr_y_feat = feat
                        gyr_y_feat = np.insert(gyr_y_feat, len(gyr_y_feat[0]),
                                               self.m_label)
                    elif i is 2:
                        gyr_z_feat = feat
                        gyr_z_feat = np.insert(gyr_z_feat, len(gyr_z_feat[0]),
                                               self.m_label)
                for i, feat in enumerate(self.oriFeatures):
                    if i is 0:
                        ori_x_feat = feat
                        ori_x_feat = np.insert(ori_x_feat, len(ori_x_feat[0]),
                                               self.m_label)
                    elif i is 1:
                        ori_y_feat = feat
                        ori_y_feat = np.insert(ori_y_feat, len(ori_y_feat[0]),
                                               self.m_label)
                    elif i is 2:
                        ori_z_feat = feat
                        ori_z_feat = np.insert(ori_z_feat, len(ori_z_feat[0]),
                                               self.m_label)
                    elif i is 3:
                        ori_w_feat = feat
                        ori_w_feat = np.insert(ori_w_feat, len(ori_w_feat[0]),
                                               self.m_label)
            return emg_0_feat, emg_1_feat, emg_2_feat, emg_3_feat, emg_4_feat, emg_5_feat, emg_6_feat, emg_7_feat, acc_x_feat, acc_y_feat, acc_z_feat, gyr_x_feat, gyr_y_feat, gyr_z_feat, ori_x_feat, ori_y_feat, ori_z_feat, ori_w_feat
        else:
            return None
