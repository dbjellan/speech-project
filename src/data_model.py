import os
import json

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# loads files names of the speech samples
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir, 'corpus', 'extracted')
processed_dir = os.path.join(project_dir, 'corpus', 'processed')


class Model():
    def __init__(self, max_timesteps, num_classes, num_features, num_samples, dir):
        self.max_timesteps = max_timesteps
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_samples = num_samples
        self.dir = dir

    def pad(self, x):
        pad_num = self.max_timesteps - x.shape[1]
        return np.pad(x, ((0, pad_num), (0, 0)), 'constant', constant_values=0)

    def sparse_y(self, batch_y):
        idxs = []
        vals = []
        for target_i, target in enumerate(batch_y):
            for seq_i, val in enumerate(target):
                idxs.append([target_i, seq_i])
                vals.append(val)
        shape = [len(batch_y), np.asarray(idxs).max(0)[1]+1]
        return np.array(idxs), np.array(vals), np.array(shape)

    def get_batch(self, batch_idx, batch_size):
        start_indx = batch_idx*batch_size
        batch_x = np.zeros((self.max_timesteps, self.batch_size, self.num_features))
        batch_y = []
        batch_lengths = np.zeros(batch_size)
        for i in range(batch_size):
            with open(os.path.join(self.dir, fname_to_idx(start_indx+i) + '-x')) as x_file:
                x = np.load(x_file)
                batch_x[:, i, :] = self.pad(x)
            with open(os.path.join(self.dir, fname_to_idx(start_indx+i) + '-y')) as y_file:
                y = np.load(y_file)
                batch_y.append(y)
                batch_lengths[i] = y.shape[-1]

        sparse_y = self.sparse_y(batch_y)
        return batch_x, sparse_y, batch_lengths


# returns list of tuples ('sample.wav', transcript) from VoxForge data directory
def parse_samples(directory):
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


def build_phonetic_dict():
    # Assign each phoneme an id
    phoneme_ids = {}
    f = open(os.path.join(project_dir, 'corpus/cmudict-0.7b.symbols'), 'r')
    symbols = f.readlines()
    symbols.sort()
    symbols = map(lambda x: x.strip('\n'), symbols)
    f.close()
    for i in range(len(symbols)):
        phoneme_ids[symbols[i]] = i

    # Create dict mapping English words to sequence of ids from cmu dictionary
    phonetic_dict = {}
    f = open(os.path.join(project_dir, 'corpus/cmudict-0.7b'), 'r')
    while True:
        line = f.readline()
        if not line:
            break
        # skip commented out lines
        if line.find(';;;') != -1:
            continue
        line = line.strip('\n')
        parts = line.split(' ')
        word = parts[0]
        id_seq = map(lambda x: phoneme_ids[x], parts[2:])
        phonetic_dict[word] = id_seq

    f.close()

    return phonetic_dict, phoneme_ids


# processes each transcript into a sequence of phoneme ids
def process_text(transcripts, phonetic_dict):
    words = transcripts.split(' ')
    return map(lambda x: phonetic_dict[x], words)


# processes audio file into np array (num_frames x num_features) using python_speech_features defaults
def process_recording(audio_file, num_features):
    with wav(audio_file) as f:
        fs, audio = wav.read(f)

    coeffs = mfcc(audio, samplerate=fs, nfilt=num_features)

    return coeffs


def fname_to_idx(i, dir):
    return os.path.join(dir, "%05d" % (i, ))


def process_data(output_dir):
    phonetic_dict, phoneme_ids = build_phonetic_dict()

    num_classes = len(phoneme_ids)+1
    num_features = 26

    source_dirs = os.listdir(data_dir)
    source_dirs = map(lambda x: os.path.join(data_dir, x), source_dirs)
    sources = map(parse_samples, source_dirs)

    max_timesteps = 0
    num_samples = 0
    for i in range(len(sources)):
        source = sources[i]
        for recording in source:
            num_samples += 1
            with open(os.path.join(output_dir, fname_to_idx(num_samples) + '-x'), 'wb') as input_file:
                coeffs = process_recording(recording[0], num_features)
                np.save(input_file, coeffs)
                if coeffs.shape[1] > max_timesteps:
                    max_timesteps = coeffs.shape[1]

            with open(os.path.join(output_dir, fname_to_idx(num_samples, output_dir) + '-y'), 'wb') as transcript_file:
                encoded_transcript = process_text(recording[1], phonetic_dict)
                np.save(transcript_file, encoded_transcript)
        print('Processing data: source %d of %d, ~%f percent complete' % (i, len(sources), float(i)/(len(sources))))


    data = {
        'max_timesteps': max_timesteps,
        'num_classes': num_classes,
        'num_features': num_features,
        'num_samples': num_samples
    }
    with open(os.path.join(output_dir, 'data.json')) as f:
        json.dump(f, data)


model = None


def load_data(dir):
    global model
    metadata_file = os.path.join(dir, 'data.json')
    if os.path.exists(metadata_file):
        print('Loading data model...')
        f = open(metadata_file)
        metadata = json.load(f)
        max_timesteps = metadata['max_timesteps']
        num_classes = metadata['num_classes']
        num_features = metadata['num_features']
        num_samples = metadata['num_samples']
        model = Model(max_timesteps, num_classes, num_features, num_samples, dir)
    else:
        print('Processing data...')
        process_data(dir)
        load_data()


load_data(processed_dir)


def main():
    pass

if __name__ == '__main__':
    main()
