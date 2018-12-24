import librosa
import h5py
import os
import numpy as np
import sys
from tqdm import tqdm
# need to explicit import display
import librosa.display
import matplotlib.pyplot as plt
from sklearn import preprocessing
import configparser

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


class Dcase18TaskbData:

    def __init__(self):
        self.verbose = True
        config = configparser.ConfigParser()
        config.read(os.path.join(ROOT_DIR, 'data_manager.cfg'))
        self.dev_path = config['dcase18_taskb']['dev_path']
        data_h5 = os.path.join(ROOT_DIR, 'data_h5')
        if not os.path.exists(data_h5):
            os.makedirs(data_h5)
        self.dev_h5_path = os.path.join(data_h5, 'TaskbDev.h5')
        self.dev_matrix_h5_path = os.path.join(data_h5, 'TaskbDevMatrix.h5')
        self.dev_matrix_fnames_h5_path = os.path.join(data_h5, 'TaskbDevMatrixFnames.h5')
        self.lb_h5_path = os.path.join(data_h5, 'TaskbLB.h5')
        self.train_path = os.path.join(self.dev_path, 'evaluation_setup/fold1_train.txt')
        self.test_path = os.path.join(self.dev_path, 'evaluation_setup/fold1_evaluate.txt')
        self.meta_path = os.path.join(self.dev_path, 'meta.csv')
        # fname_encoder encode audio names to int, vice versa.
        self.fname_encoder = preprocessing.LabelEncoder()
        self.set_fname_encoder()

    def extract_logmel(self, wav_path):
        """        self.fname_encoder = preprocessing.LabelEncoder()
        Give a wav, extract logmel feature
        :param wav_path:
        :return: fea of dim (1, frequency, time), first dim is added
        """
        x, sr = librosa.load(wav_path, sr=None, mono=False)
        assert (x.shape == (441000,) and sr == 44100)

        # 40ms winlen, half overlap
        y = librosa.feature.melspectrogram(x, sr=44100, n_fft=1764, hop_length=882, n_mels=40)
        # about 1e-7
        EPS = np.finfo(np.float32).eps
        fea = np.log(y+EPS)
        # add a new axis
        return np.expand_dims(fea[:, :-1], axis=0)

    def extract_fea_for_datagroup(self, data_group, mode='train'):
        """
        Loop through train/test setup file, and extract logmel for each audio, sore in a h5 group
        :param data_group: a hdf5 group(group works like dictionary)
        :param mode: train or test
        :return:
        """
        if mode == 'train':
            fp = open(self.train_path, 'r')
        else:
            fp = open(self.test_path, 'r')

        for i, line in tqdm(enumerate(fp)):
            audio_name, label = line.split()
            audio_path = os.path.join(self.dev_path, audio_name)
            fea = self.extract_logmel(wav_path=audio_path)

            wav_name = os.path.basename(audio_path)
            data_group[wav_name] = fea
            data_group[wav_name].attrs['label'] = label
            data_group[wav_name].attrs['venue'] = wav_name.split('-')[1]
            data_group[wav_name].attrs['device'] = wav_name.split('-')[4][0]
            # label could be extracted by :data_group[u'airport-barcelona-0-0-a.wav'].attrs['label']

    def set_fname_encoder(self):
        """
        meta_path store all dev files and labels, fit all wav_names to get an encoder
        the encoder could transform string to int vice versa.
        :return:
        """

        fp = open(self.meta_path, 'r')
        wav_names = []
        next(fp)
        for i, line in tqdm(enumerate(fp)):
            audio_name, _, _, _ = line.split()
            wav_name = os.path.basename(audio_name)
            wav_names.append(wav_name)
        self.fname_encoder.fit(wav_names)

    def extract_npy(self, mode='train', devices='abc'):
        """
        Extract data and label as numpy array from dev.h5
        :param mode:
        :param devices: a, b, c
        :return: data and label as numpy array
        """

        data = []
        label = []
        with h5py.File(self.dev_h5_path, 'r') as f:
            audios = f[mode].keys()
            for audio in audios:
                # extract according to device
                if f[mode][audio].attrs['device'] in devices:
                    data.append(np.array(f[mode][audio].value))
                    label.append(np.array(f[mode][audio].attrs['label']))
        # concat data along existing axis 0
        data = np.concatenate(data, axis=0)
        le = preprocessing.LabelEncoder()
        label_ids = le.fit_transform(label)
        return data, label_ids

    def extrac_para_npy(self, mode='train'):
        data = []
        label = []
        with h5py.File(self.dev_h5_path, 'r') as f:
            audios = f[mode].keys()
            for audio in audios:
                if audio[-5] == 'b':
                    para_audio = audio.replace('-b.wav', '-a.wav')
                    data.append(np.array(f[mode][para_audio].value))
                    label.append(np.array(f[mode][para_audio].attrs['label']))
        # concat data along existing axis 0
        data = np.concatenate(data, axis=0)
        le = preprocessing.LabelEncoder()
        label_ids = le.fit_transform(label)
        return data, label_ids

    def extract_neg_para_npy(self, mode='train'):
        data = []
        label = []
        para_audios = []
        with h5py.File(self.dev_h5_path, 'r') as f:
            audios = f[mode].keys()
            # get list of para audios
            for audio in audios:
                if audio[-5] == 'b':
                    para_audio = audio.replace('-b.wav', '-a.wav')
                    para_audios.append(para_audio)
            for audio in audios:
                if audio[-5] == 'a' and audio not in para_audios:
                    neg_para_audio = audio
                    data.append(np.array(f[mode][neg_para_audio].value))
                    label.append(np.array(f[mode][neg_para_audio].attrs['label']))
        # concat data along existing axis 0
        data = np.concatenate(data, axis=0)
        le = preprocessing.LabelEncoder()
        label_ids = le.fit_transform(label)
        return data, label_ids

    def extract_npy_fnames(self, mode='train', devices='abc'):
        """
        Extract data, label, fnames(encoded as int) as numpy array from dev.h5
        :param mode:
        :param devices: a, b, c
        :return: data and label as numpy array
        """

        data = []
        label = []
        fnames = []
        with h5py.File(self.dev_h5_path, 'r') as f:
            audios = f[mode].keys()
            for audio in audios:
                # extract according to device
                if f[mode][audio].attrs['device'] in devices:
                    data.append(np.array(f[mode][audio].value))
                    label.append(np.array(f[mode][audio].attrs['label']))
                    fnames.append(audio)
        # concat data along existing axis 0
        data = np.concatenate(data, axis=0)
        le = preprocessing.LabelEncoder()
        label_ids = le.fit_transform(label)
        fnames_codes = self.fname_encoder.transform(fnames)
        return data, label_ids, fnames_codes

    def create_devh5(self):
        """
        Extract LogMel and Store in h5 File, index by wav name
        :return:
        """
        if os.path.exists(self.dev_h5_path):
            print("[LOGGING]: " + self.dev_h5_path + " exists!")
            return

        with h5py.File(self.dev_h5_path, 'w') as f:

            # create a group: f['train']
            train = f.create_group('train')
            self.extract_fea_for_datagroup(train, mode='train')

            # f['test']
            test = f.create_group('test')
            self.extract_fea_for_datagroup(test, mode='test')

        f.close()

    def create_dev_matrix(self):
        """
        Store train and test data in h5, ready to be load by tensorflow
        :return: None
        """
        if os.path.exists(self.dev_matrix_h5_path):
            print("[LOGGING]: " + self.dev_matrix_h5_path + " exists!")
            return

        with h5py.File(self.dev_matrix_h5_path, 'w') as f:

            for mode in ['train', 'test']:
                for device in ['a', 'b', 'c']:
                    grp = f.create_group(mode + '/' + device)
                    grp['data'], grp['label'] = self.extract_npy(mode=mode, devices=device)
                # add parallel data as separate device p
                grp = f.create_group(mode + '/p')
                grp['data'], grp['label'] = self.extrac_para_npy(mode=mode)

                # add neg parallel data as device A
                grp = f.create_group(mode + '/A')
                grp['data'], grp['label'] = self.extract_neg_para_npy(mode=mode)
        f.close()

    def create_dev_matrix_fnames(self):
        """
        Store train and test data, labels, fnames in h5, ready to be load by tensorflow
        :return: None
        """
        if os.path.exists(self.dev_matrix_fnames_h5_path):
            print("[LOGGING]: " + self.dev_matrix_fnames_h5_path + " exists!")
            return

        with h5py.File(self.dev_matrix_fnames_h5_path, 'w') as f:

            for mode in ['train', 'test']:
                for device in ['a', 'b', 'c']:
                    grp = f.create_group(mode + '/' + device)
                    grp['data'], grp['label'], grp['fnames'] = self.extract_npy_fnames(mode=mode, devices=device)
        f.close()

    def load_dev(self, mode='train', devices='abc'):
        """
        Give mode and device, load data and label as numpy array
        :param mode:
        :param devices: could be combination of 'a', 'b', 'c'
        :return: data, label as np array
        """
        if not os.path.exists(self.dev_matrix_h5_path):
            print(self.dev_matrix_h5_path + "not exists!")
            sys.exit()

        with h5py.File(self.dev_matrix_h5_path, 'r') as f:
            data = []
            label = []
            for device in devices:
                data.append(np.array(f[mode][device]['data'].value))
                label.append(np.array(f[mode][device]['label'].value))
            # concat data and label from multi devices as required, along "batch" axis
            datas = np.concatenate(data, axis=0)
            labels = np.concatenate(label, axis=0)
            if self.verbose:
                print("[LOGGING]: Loading", mode, devices, "of shape: ", datas.shape)
            return datas, labels

    def load_dev_with_fnames(self, mode='train', devices='abc'):
        """
        :param mode:adhv
        :param devices:
        :return: datas, labels, wav file names int coded, numpy arrays
        """
        if not os.path.exists(self.dev_matrix_fnames_h5_path):
            print("[LOGGING]: " + self.dev_matrix_fnames_h5_path + "not exists!")
            sys.exit()
        with h5py.File(self.dev_matrix_fnames_h5_path, 'r') as f:
            data = []
            label = []
            fnames = []
            for device in devices:
                data.append(np.array(f[mode][device]['data'].value))
                label.append(np.array(f[mode][device]['label'].value))
                fnames.append(np.array(f[mode][device]['fnames'].value))
            # concat data and label from multi devices as required, along "batch" axis
            datas = np.concatenate(data, axis=0)
            labels = np.concatenate(label, axis=0)
            fnames = np.concatenate(fnames, axis=0)
            if self.verbose:
                print("[LOGGING]: Loading", mode, devices, "of shape: ", datas.shape)
            return datas, labels, fnames

    def show_spec_by_name(self, wav_name):
        """
        Given wav name, plot LogMel Spectrogram
        :param wav_name:
        :return:
        """
        with h5py.File(self.dev_h5_path, 'r') as f:
            if wav_name in f['train'].keys():
                spec_data = np.array(f['train'][wav_name].value)
                print(wav_name, "in train split and label is", f['train'][wav_name].attrs['label'], \
                    "device is", f['train'][wav_name].attrs['device'])
            else:
                spec_data = np.array(f['test'][wav_name].value)
                print(wav_name, "in test split and label is", f['test'][wav_name].attrs['label'], "device is", \
                    f['test'][wav_name].attrs['device'])

            plt.figure()
            librosa.display.specshow(spec_data[0])

            plt.title(wav_name)
            plt.show()

    def load_spec_by_name(self, wav_name):
        with h5py.File(self.dev_h5_path, 'r') as f:
            if wav_name in f['train'].keys():
                spec_data = np.array(f['train'][wav_name].value)
            else:
                spec_data = np.array(f['test'][wav_name].value)
            return spec_data


if __name__ == '__main__':
    data_manager = Dcase18TaskbData()
    data_manager.create_devh5()
    data_manager.create_dev_matrix()
    data_manager.create_dev_matrix_fnames()

    # list of the number of train/test set
    for mode in ['train', 'test']:
        for device in ['a', 'b', 'c', 'p', 'A', 'abc']:
            data, label = data_manager.load_dev(mode=mode, devices=device)
            # print(mode + '/' + device + ': ', len(label))
    # dataset_b = data_manager.load_dev(mode='train', devices='b')
