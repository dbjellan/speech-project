#!/usr/bin/python2

import math

from scikits.audiolab import Sndfile
import numpy as np
from matplotlib import pyplot as plt


def process_recording(filename, window_width=.03, window_spacing=.02, num_coeffs=40, mel_encode=True):
    f = Sndfile(filename, 'r')
    fs = f.samplerate
    nc = f.channels
    enc = f.encoding
    n = f.nframes
    data = f.read_frames(n)

    samples = int(fs*window_width)
    num_windows = int((len(data)/(fs*window_spacing)))-1
    freqs = np.fft.rfftfreq(samples, d=1./fs)

    if len(freqs) % 2 == 1:
        idx = len(freqs)/2
        pos_freqs = freqs[len(freqs)/2:]
    else:
        idx = len(freqs)/2-1
        pos_freqs = freqs[len(freqs)/2-1:]

    spectragram = np.empty((num_windows, len(pos_freqs)))

    for i in range(num_windows):
        left = ((i+1)*fs*window_spacing)-int(samples/2)
        right = ((i+1)*fs*window_spacing)+int(math.ceil(samples/2))
        window = data[left:right]
        spectragram[i] = np.abs(np.fft.rfft(window)[idx:])

    edges = np.linspace(0, fs / 2., num=(num_coeffs+2))
    if mel_encode:
        edges = mel_transform(edges)
    filter_bank = np.matrix(np.empty((num_coeffs, len(pos_freqs))))
    for i in range(num_coeffs):
        for j in range(len(pos_freqs)):
            if edges[i] <= pos_freqs[j] <= edges[i+2]:
                filter_bank[i, j] = triangle(edges[i], edges[i+1], edges[i+2], pos_freqs[j])

    coeffs = np.empty((num_windows, num_coeffs))
    for i in range(num_windows):
        coeffs[i] = np.transpose(filter_bank * np.transpose(np.matrix(spectragram[i])))

    return np.transpose(coeffs), np.transpose(spectragram)


def triangle(left, middle, right, x):
    """Returns array containing the cepstrum and spectrum of the provided wav files
     at window_width intervals.

    """
    if left <= x <= middle:
        return x/(middle-left)
    elif middle <= x <= right:
        return 1 - (x-middle)/(right-middle)
    else:
        return 0


def cepstrum(data):
    spectrum = np.fft.fft(data)
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real

    return ceps


def mel_transform(data):
    return 2595 * np.log10(1 + np.abs(data)/700.0)


def graph_spectragram(data, window_width):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(abs(data))
    plt.show()
