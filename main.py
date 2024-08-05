import os
import math
import random
import argparse

import cupy as cp
import numpy as np
from scipy.signal import chirp
from scipy.fft import fft
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import colorsys

# random seeds
np.random.seed(10)
random.seed(10)

bw = 125e3  # bandwidth
fs = 1e6  # sampling frequency
data_dir = '/path/to/NeLoRa_Dataset/'  # directory for training dataset

snr_range = list(range(-40, -10))  # range of SNR for training
sfrange = list(range(7, 11))
method_names = ['LoRaPhy', 'LoRaTrimmer']
linestyles = dict(zip(method_names, ['--', '-']))
colors = dict(zip(sfrange, [colorsys.hls_to_rgb((hue + 0.5) / len(sfrange), 0.4, 1) for hue in range(len(sfrange))]))


# decoding symbols using loraphy, as baseline method
# note: this method only works with upsampling (FS >= BW*2)
def decode_loraphy(data_in, num_classes, downchirp):
    upsampling = 100  # up-sampling rate for loraphy, default 100
    # upsamping can counter possible frequency misalignments, finding the highest position of the signal peak, but higher upsampling lead to more noise

    # dechirp
    chirp_data = data_in * downchirp

    # compute FFT
    fft_raw = fft(chirp_data, len(chirp_data) * upsampling)

    # cut the FFt results to two (due to upsampling)
    target_nfft = num_classes * upsampling
    cut1 = np.array(fft_raw[:target_nfft])
    cut2 = np.array(fft_raw[-target_nfft:])

    # add absolute values of cut1 and cut2 to merge two peaks into one
    return round(np.argmax(abs(cut1) + abs(cut2)) / upsampling) % num_classes


# adding noise for data
def add_noise(dataY, truth_idx, sf, snr):
    num_classes = 2 ** sf  # number of codes per symbol == 2 ** sf
    num_samples = int(num_classes * fs / bw)  # number of samples per symbol
    # add noise of a certain SNR, chosen from snr_range
    amp = math.pow(0.1, snr / 20) * np.mean(np.abs(dataY))
    noise = (amp / math.sqrt(2) * np.random.randn(num_samples) + 1j * amp / math.sqrt(2) * np.random.randn(num_samples))
    dataX = dataY + noise  # dataX: data with noise
    return dataX


# load the whole dataset
def load_data(sf, downchirp):

    num_classes = 2 ** sf  # number of codes per symbol == 2 ** sf
    num_samples = int(num_classes * fs / bw)  # number of samples per symbol

    # read all file paths
    files = [[] for i in range(num_classes)]
    for root, dirs, file_in_dir in tqdm(os.walk(os.path.join(data_dir, str(sf)))):
        for filename in file_in_dir:
            if filename.endswith('.csv') or 'wrong' in filename: continue
            truth_idx = int(filename.split('_')[1])
            files[truth_idx].append(os.path.join(root, filename))

    # read file contents
    datax = []  # chirp symbols
    datay = []  # truth indexes for each symbol
    for truth_idx, filelist in tqdm(enumerate(files), desc='Reading Files'):
        for filepath in filelist:
            with open(filepath, 'rb') as fid:
                # read file
                chirp_raw = np.fromfile(fid, np.complex64, num_samples)
                assert len(chirp_raw) == num_samples
                # check if code is correct
                if decode_loraphy(chirp_raw, num_classes, downchirp) == truth_idx:
                    # append data
                    datax.append(chirp_raw)
                    datay.append(truth_idx)

    return datax, datay


def gen_constants(sf):
    num_classes = 2 ** sf  # number of codes per symbol == 2 ** sf
    num_samples = int(num_classes * fs / bw)  # number of samples per symbol
    # generate downchirp
    t = np.linspace(0, num_samples / fs, num_samples + 1)[:-1]
    chirpI1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    chirpQ1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    downchirp = chirpI1 + 1j * chirpQ1

    # two DFT matrices
    dataE1 = np.zeros((num_classes, num_samples), dtype=np.cfloat)
    dataE2 = np.zeros((num_classes, num_samples), dtype=np.cfloat)
    for symbol_index in range(num_classes):
        time_shift = int(symbol_index / num_classes * num_samples)
        time_split = num_samples - time_shift
        dataE1[symbol_index][:time_split] = downchirp[time_shift:]
        if symbol_index != 0: dataE2[symbol_index][time_split:] = downchirp[:time_shift]
    dataE1 = cp.array(dataE1)
    dataE2 = cp.array(dataE2)
    return downchirp, dataE1, dataE2


def decode_ours(dataX, dataE1, dataE2):
    dataX = cp.array(dataX).T
    data1 = cp.matmul(dataE1, dataX)
    data2 = cp.matmul(dataE2, dataX)
    vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
    est = cp.argmax(vals).item()
    return est


# Plot
def generate_plot(vals):
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(8, 6))
    plt.axhline(y=0.9, linestyle='--', color='black')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')

    for name in method_names:
        for sf in sfrange:
            mean_val = [np.mean(vals[name][sf][snr]) if len(vals[name][sf][snr]) else 0 for snr in snr_range]
            plt.plot(snr_range, mean_val, label=f'{name}_SF{sf}', color=colors[sf], linestyle=linestyles[name])

    plt.legend()
    plt.savefig(f'result.png')
    plt.close()


# Test main function
def main():
    vals = dict([(k, dict([(j, dict([(i, []) for i in snr_range])) for j in sfrange])) for k in method_names])
    for sf in sfrange:
        num_classes = 2 ** sf  # number of codes per symbol == 2 ** sf
        downchirp, dataE1, dataE2 = gen_constants(sf)

        datax, datay = load_data(sf, downchirp)

        for dataY_test, truth_test in tqdm(zip(datax, datay)):
            for snr in snr_range:
                dataX = add_noise(dataY_test, truth_test, sf, snr)

                est_loraphy = decode_loraphy(dataX, num_classes, downchirp)
                est_ours = decode_ours(dataX, dataE1, dataE2)

                vals['LoRaPhy'][sf][snr].append(int(est_loraphy == truth_test))
                vals['LoRaTrimmer'][sf][snr].append(int(est_ours == truth_test))
    with open("vals.pkl", "wb") as f:
        pickle.dump(vals, f)
    # Plot
    generate_plot(vals)


if __name__ == '__main__':
    main()  # For training





