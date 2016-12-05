#Processes raw data into a usable form

import os
import errno
import json
import string

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# loads files names of the speech samples
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir, 'corpus', 'extracted')
processed_dir = os.path.join(project_dir, 'corpus', 'processed')

try:
    os.makedirs(processed_dir)
except OSError as exc:
    pass


class Model():
    def __init__(self, dir, max_timesteps, num_classes, num_features, num_samples):
        self.max_timesteps = int(max_timesteps)
        self.num_classes = int(num_classes)
        self.num_features = int(num_features)
        self.num_samples = int(num_samples)
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
        batch_x = np.zeros((batch_size, self.max_timesteps, self.num_features))
        print(batch_x.size)
        batch_y = []
        batch_lengths = np.zeros(batch_size)
        for i in range(batch_size):
            with open(os.path.join(self.dir, 'x-' + fname_to_idx(start_indx+i) + '.dat')) as x_file:
                x = np.load(x_file)
                batch_x[i, :, :] = self.pad(x)
            with open(os.path.join(self.dir, 'y-' + fname_to_idx(start_indx+i) + '.dat')) as y_file:
                y = np.load(y_file)
                batch_y.append(y)
                batch_lengths[i] = y.shape[-1]

        sparse_y = self.sparse_y(batch_y)
        return batch_x, sparse_y, batch_lengths

def get_prompt_files(dir):
    # If contains a single prompt files, return that
    for fname in os.listdir(dir):
        if 'prompts' in fname.lower():
            return [os.path.join(dir, fname)]
    prompts = []
    for fname in os.listdir(dir):
        if fname not in ['GPL_License', 'README']:
            prompts.append(os.path.join(dir, fname))
    return prompts


# returns list of tuples ('sample.wav', transcript) from VoxForge data directory
def parse_samples(directory):
    wavfiles = []
    prompt_dir = os.path.join(directory, 'etc')
    promptfiles = get_prompt_files(prompt_dir)
    for promptfile in promptfiles:
        with open(promptfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                tokens = line.split(' ')
                if len(tokens) > 2:
                    path = tokens[0]
                    name = path.split('/')[-1]
                    filename = os.path.join(directory, 'wav/' + name + '.wav')
                    if os.path.exists(filename):
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
        word = parts[0].lower()
        id_seq = map(lambda x: phoneme_ids[x], parts[2:])
        phonetic_dict[word] = id_seq

    f.close()

    return phonetic_dict, phoneme_ids


# processes each transcript into a sequence of phoneme ids
def process_text(transcript, phonetic_dict):
    words = transcript.split(' ')
    phonetic_seq = []
    unrecognized_word = False
    for word in words:
        w_stripped = filter(lambda x: x in string.letters, word.lower())
        if len(w_stripped) > 0:
            seq = phonetic_dict.get(w_stripped)
            if seq:
                phonetic_seq.extend(seq)
            else:
                unrecognized_word = True
                print('unrecognized word: %s' % (w_stripped,))
    return np.asarray(phonetic_seq), unrecognized_word


# processes audio file into np array (num_frames x num_features) using python_speech_features defaults
def process_recording(audio_file, num_features=26):
    fs, audio = wav.read(audio_file)
    coeffs = mfcc(audio, samplerate=fs, nfilt=num_features)

    return coeffs


def fname_to_idx(i):
    return "%05d" % (i, )

#Runs through the data directory, parsing and cleaning data, and saving it for future use
def process_data(output_dir):
    phonetic_dict, phoneme_ids = build_phonetic_dict()

    #add extra id for null phoneme
    num_classes = len(phoneme_ids)+1
    num_features = 13

    source_dirs = os.listdir(data_dir)
    source_dirs = map(lambda x: os.path.join(data_dir, x), source_dirs)
    sources = map(parse_samples, source_dirs)

    max_timesteps = 0
    num_samples = 0
    for i in range(len(sources)):
        source = sources[i]
        for recording in source:
            encoded_transcript, missing_word = process_text(recording[1], phonetic_dict)
            if not missing_word:
                with open(os.path.join(output_dir, 'y-' + fname_to_idx(num_samples + 1) + '.dat'), 'wb') as yf:
                    np.save(yf, encoded_transcript)

                coeffs = process_recording(recording[0])
                if coeffs.shape[0] > max_timesteps:
                    max_timesteps = coeffs.shape[0]

                with open(os.path.join(output_dir, 'x-' + fname_to_idx(num_samples+1) + '.dat'), 'wb') as input_file:
                    np.save(input_file, coeffs)

                num_samples += 1

        if i % 10 == 0:
            print('Approximately %2.3f percent complete' % (100*float(i+1)/(len(sources)), ))


    data = {
        'max_timesteps': max_timesteps,
        'num_classes': num_classes,
        'num_features': num_features,
        'num_samples': num_samples
    }
    with open(os.path.join(output_dir, 'data.json'), 'w') as f:
        json.dump(data, f)


model = None

#Initializes data
def load_data(dir):
    global model
    metadata_file = os.path.join(dir, 'data.json')
    if os.path.exists(metadata_file):
        print('Loading data model...')
        f = open(metadata_file)
        metadata = json.load(f)
        num_classes = metadata['num_classes']
        num_features = metadata['num_features']
        num_samples = metadata['num_samples']
        model = Model(dir, max_timesteps=metadata['max_timesteps'], num_classes=metadata['num_classes'],
                      num_features=metadata['num_features'], num_samples=metadata['num_samples'])
    else:
        print('Processing data...')
        process_data(dir)
        load_data(dir)


load_data(processed_dir)


def main():
    pass

if __name__ == '__main__':
    main()
