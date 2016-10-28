import os
import math

from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt

project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_directory = os.path.join(project_directory, 'corpus/extracted')
sources = os.listdir(data_directory)
sources.sort()
sources = map(lambda x: os.path.join(data_directory, x), sources)

def parse_source(directory):
    wavfiles = []
    promptfile = open(os.path.join(directory, 'etc/PROMPTS'), 'r')
    lines = promptfile.readlines()
    for line in lines:
        tokens = line.split(' ')
        path = tokens[0]
        name = path.split('/')[-1]
        filename = os.path.join(directory, 'wav/' + name + '.wav')
        transcript = ' '.join(tokens[1:])
        wavfiles.append((filename, transcript))
    return wavfiles


"""Returns array containing the cepstrum and spectrum of the provided wav files
 at window_width intervals.

"""


def process_recording(filename, window_width=.1):
    rate, data = wavfile.read(filename)
    samples = int(rate*window_width)

    num_windows = int(math.ceil(len(data)/samples))
    ceps = np.empty((num_windows, samples))
    spectragram = np.empty((num_windows, samples))
    for i in range(num_windows):
        window = data[i*samples:(i+1)*samples]
        ceps[i] = np.fft.fft(window)
        spectragram[i] = cepstrum(window)

    return ceps, spectragram


def graph_spectragram(data, window_width):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(abs(data))
    plt.show()


def cepstrum(data):
    spectrum = np.fft.fft(data)
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real

    return ceps

def process_data():
    pass

def main():
    pass

if __name__ == '__main__':
    main()