from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
import numpy as np
from python_speech_features import mfcc
from python_speech_features import sigproc
from scipy import signal

# concatnate methods for extracting features in a single loop
# can be done later but the present approach will cause a huge drop in
# performance

max_abs_scaler = preprocessing.MaxAbsScaler()


def meanScale(data):
    return preprocessing.scale(data)


def absScale(data):
    nor_data = max_abs_scaler.fit_transform(data)
    return nor_data


def absScale(data):
    return preprocessing.scale(data)


def getFeatures(data, samplingRate, window=False, f_mfcc=False):
    '''
    Method to get the time domain and frequency domain features for a given
    signal
    :param data: (list)                 : The signal whos features will be
    extracted
    :param samplingRate: (int)          : The sampling rate of the signal
    :param window: (Boolean)            : Flag to extract overlapping windows
    :param f_mfcc: (Boolean)            : Flag to extract MFCC features
    :return: feature_matrix (ndarray)   : A feature matrix for the input signal
    '''

    windowed_frames = None

    # Create overlapping windows
    if window == False:
        ovlp_windows = [data]
        windowed_frames = ovlp_windows
    else:
        window_size = int(samplingRate * 1)  # a window length of 1 seconds
        overlap_size = int(
            samplingRate * 0.5)  # a window overlap of 0.5 seconds
        ovlp_windows = get_sliding_windows(data, window_size, overlap_size)
        # apply hamming window function
        windowed_frames = windowfn(ovlp_windows)

    # MFCC Features
    if f_mfcc:
        mfcc_feat = get_mfcc_features(data, samplingRate)

    # Time domain features
    window_gc, \
    window_rms, \
    window_mean, \
    window_var, \
    window_ssi, \
    window_iemg, \
    window_peaks, \
    window_minima, \
    window_maxima = get_time_features(windowed_frames)

    # Frequency domain features
    window_mean_pwr, \
    window_pow_peaks, \
    window_tot_pw, \
    window_pow_var, \
    window_max_fr, \
    window_dominating_frequencies = get_freq_features(windowed_frames, 512,
                                                      samplingRate)

    # get 1st order derivative of the signal
    d1 = get_derivative(windowed_frames)

    # get 2nd order derivative
    d2 = get_derivative(d1)

    # create feature vector
    feature_matrix = np.array([window_gc,
                               window_rms,
                               window_mean,
                               window_var,
                               window_ssi,
                               window_iemg,
                               window_peaks,
                               window_minima,
                               window_maxima,
                               window_mean_pwr,
                               window_pow_peaks,
                               window_tot_pw,
                               window_pow_var,
                               window_max_fr,
                               window_dominating_frequencies])

    feature_matrix = np.transpose(feature_matrix)

    # adding MFCC features to the feature matrix
    if f_mfcc and (window is False):
        mfcc_feat = [np.ravel(mfcc_feat)]
        feature_matrix = np.concatenate((feature_matrix, mfcc_feat), axis=1)
    elif f_mfcc and window:
        feature_matrix = np.concatenate((feature_matrix, mfcc_feat), axis=1)

    # adding derivatives to the feature matrix
    feature_matrix = np.concatenate((feature_matrix, d1), axis=1)
    feature_matrix = np.concatenate((feature_matrix, d2), axis=1)

    return feature_matrix


def get_derivative(frames):
    windowed_derivatives = []
    for frame in frames:
        windowed_derivatives.append(np.gradient(frame))
    return np.array(windowed_derivatives)


def get_mfcc_features(data, sampleRate):
    mfcc_feat = mfcc(data, sampleRate, 1, 0.5)
    # mfcc_feat = np.ravel(mfcc_feat)
    return mfcc_feat


def get_time_features(frames):
    '''
    Method to extract time domain features of the windowed frames
    :param frames: Windoed frames of the imput signal
    :return:window_gc (list)            : The number of zero crossings per frame
           window_rms (list)            : The Root Mean Squared value for frame
           window_mean (list)           : Mean of every frame
           window_var (list)            : Variance of every frame
           window_ssi (list)            : The integral of the square of
           signal strength at every time step in every frame
           window_iemg (list)           : The sum of absolute values of the
           signal strengths at every time step in every frame
           window_peaks (list)          : the number of peaks in every frame
           window_minima (list)         : Maxima in every frame
           window_maxima (list)         : Minima in every frame
    '''
    window_gc = []
    # window_zc = []
    # window_len = []
    window_rms = []
    window_mean = []
    window_var = []
    window_ssi = []
    window_iemg = []
    window_peaks = []
    # window_auto_coor = []
    window_minima = []
    window_maxima = []

    for frame in frames:
        # gradient
        gs = np.gradient(frame)
        signs = np.sign(gs)
        sign_change = 0
        last_sign = 0

        for sign in signs:
            if last_sign == 1 and sign == -1:
                sign_change += 1
                last_sign = sign
            elif last_sign == -1 and sign == 1:
                sign_change += 1
                last_sign = sign
            elif last_sign == 0:
                last_sign = sign
        window_gc.append(sign_change)

        # zero crossing
        '''
        zero_crossings = np.where(np.diff(np.sign(frame)))[0]
        window_zc.append(len(zero_crossings))
        '''
        # window length
        '''
        sum = 0
        for x in range(0, len(frame) - 2):
            sum += np.absolute(frame[x + 1] - frame[x])
        window_len.append(sum)
        '''
        # rms
        rms = math.sqrt(np.sum(np.square(frame)) / len(frame))
        window_rms.append(rms)

        # mean
        m = np.mean(frame)
        window_mean.append(m)

        # variance
        var = np.var(frame)
        window_var.append(var)

        # ssi
        ssi = np.sum(np.square(frame))
        window_ssi.append(ssi)

        # iemg
        sum = np.sum(np.absolute(frame))
        window_iemg.append(sum)

        # peaks
        peakind = signal.find_peaks_cwt(frame, np.arange(1, 10))
        window_peaks.append(len(peakind))

        # auto coorealtion
        '''
        freqs = np.fft.rfft(frame)
        auto1 = freqs * np.conj(freqs)
        auto2 = auto1 * np.conj(auto1)
        result = np.fft.irfft(auto2)
        window_auto_coor.append(result)
        '''
        # minima
        minima = np.amin(frame)
        window_minima.append(minima)

        # maxima
        maxima = np.amax(frame)
        window_maxima.append(maxima)

    return np.array(window_gc), \
           np.array(window_rms), \
           np.array(window_mean), \
           np.array(window_var), \
           np.array(window_ssi), \
           np.array(window_iemg), \
           np.array(window_peaks), \
           np.array(window_minima), \
           np.array(window_maxima)


def get_freq_features(frames, NFFT, samplingRate):
    '''
    This method extracts the frequency domain features from the signal
    :param frames: (list)                           : List of windowed frames
    :param NFFT: (integer)                          : The NFFT value for
    calculating FFT
    :param samplingRate: (integer)                  : The sampling rate of
    the signal
    :return:window_mean_pwr (list)                  : Mean power of every frame
            window_n_peaks (list)                   : Number of peaks in
            every frame
            window_tot_pw (list)                    : Total power in every frame
            window_pow_var (list)                   : Variance in power in
            every frame
            window_max_fr (list)                    : Maximum frequency in
            every frame
            window_dominating_frequencies (list)    : The dominating
            frequency in every frame
    '''
    window_mean_pwr = []
    window_n_peaks = []
    window_tot_pw = []
    window_pow_var = []
    window_max_fr = []
    window_dominating_frequencies = []

    frames_pw_spec = sigproc.powspec(frames, NFFT)
    for frame in frames_pw_spec:
        # n_peaks
        peakind = signal.find_peaks_cwt(frame, np.arange(1, 10))
        window_n_peaks.append(len(peakind))

        # mean
        m = np.mean(frame)
        window_mean_pwr.append(m)

        # total power
        sum = np.sum(np.absolute(frame))
        window_tot_pw.append(sum)

        # power variance
        var = np.var(frame)
        window_pow_var.append(var)

        # min and max frequencies
        '''
        w = np.fft.fft(frame)
        freqs = np.fft.fftfreq(len(w))
        #print(freqs.min(), freqs.max())

        window_max_fr.append(freqs.min())
        window_min_fr.append(freqs.max())
        idx = np.argmax(np.abs(w))
        freq = freqs[idx]
        freq_in_hertz = abs(freq * samplingRate)
        if(freq_in_hertz > 0.0):
            print(freq_in_hertz)
        window_dominating_frequencies.append(freq_in_hertz)
        '''

        # Find the peak in the coefficients
        fourier = np.fft.fft(frame)
        frequencies = np.fft.fftfreq(len(frame))
        positive_frequencies = frequencies[np.where(frequencies > 0)]
        magnitudes = abs(
            fourier[np.where(frequencies > 0)])  # magnitude spectrum
        peak_frequency = np.argmax(magnitudes)
        peak_fr = positive_frequencies[peak_frequency]
        window_dominating_frequencies.append(peak_fr)
        window_max_fr.append(positive_frequencies.max())

    return np.array(window_mean_pwr), \
           np.array(window_n_peaks), \
           np.array(window_tot_pw), \
           np.array(window_pow_var), \
           np.array(window_max_fr), \
           np.array(window_dominating_frequencies)


def gr_change(frames):
    window_gc = []
    for frame in frames:
        gs = np.gradient(frame)
        signs = np.sign(gs)
        sign_change = 0
        last_sign = 0
        for sign in signs:
            if last_sign == 1 and sign == -1:
                sign_change += 1
                last_sign = sign
            elif last_sign == -1 and sign == 1:
                sign_change += 1
                last_sign = sign
            elif last_sign == 0:
                last_sign = sign
        # print(sign_change)
        window_gc.append(sign_change)
    return np.array(window_gc)


def zero_crossings(frames):
    window_zc = []
    for frame in frames:
        zero_crossings = np.where(np.diff(np.sign(frame)))[0]
        window_zc.append(len(zero_crossings))
    return np.array(window_zc)


def find_waveform_length(frames):
    window_len = []
    for frame in frames:
        sum = 0
        for x in range(0, len(frame) - 2):
            sum += np.absolute(frame[x + 1] - frame[x])
        window_len.append(sum)
    return np.array(window_len)


def find_rms(frames):
    window_rms = []
    for frame in frames:
        rms = math.sqrt(np.sum(np.square(frame)) / len(frame))
        window_rms.append(rms)
    return np.array(window_rms)


def find_mean(frames):
    window_mean = []
    for frame in frames:
        m = np.mean(frame)
        window_mean.append(m)
    return np.array(window_mean)


def find_var(frames):
    window_var = []
    for frame in frames:
        var = np.var(frame)
        window_var.append(var)
    return np.array(window_var)


def find_ssi(frames):
    window_ssi = []
    for frame in frames:
        ssi = np.sum(np.square(frame))
        window_ssi.append(ssi)
    return np.array(window_ssi)


def iemg(frames):
    window_iemg = []
    for frame in frames:
        sum = np.sum(np.absolute(frame))
        window_iemg.append(sum)
    return np.array(window_iemg)


def find_peaks(frames):
    window_peaks = []
    for frame in frames:
        peakind = signal.find_peaks_cwt(frame, np.arange(1, 10))
        window_peaks.append(len(peakind))
    return np.array(window_peaks)


def get_sliding_windows(data, window_size, overlap_size):
    ovlp_windows = sigproc.framesig(data, window_size, overlap_size)
    return ovlp_windows


def windowfn(frames):
    window_norm_frames = []
    for frame in frames:
        window = signal.hamming(len(frame))
        window_norm_frames.append(window * frame)
    # print(np.array(window_norm_frames))
    return np.array(window_norm_frames)


def plot_frames(frames):
    fig, axs = plt.subplots(frames.shape[0], sharex=True, sharey=True)
    for i, (ax, frame) in enumerate(zip(axs, frames)):
        ax.set_title("{0}th frame".format(i))
        ax.plot(frame)
        ax.grid(True)
        # for frame in frames:


def estimated_autocorrelation(frames):
    window_auto_coor = []
    for x in frames:
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode='full')[-n:]
        # assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
        result = r / (variance * (np.arange(n, 0, -1)))
        window_auto_coor.append(result)
    return np.array(window_auto_coor)
