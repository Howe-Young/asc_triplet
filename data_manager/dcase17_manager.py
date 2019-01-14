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


class Dcase17Data:

    def __init__(self):
        self.verbose = True
        config = configparser.ConfigParser()
        config.read(os.path.join(ROOT_DIR, 'data_manager.cfg'))
        self.config = config
        self.dev_path = config['dcase17']['dev_path']
        self.eva_path = config['dcase17']['eva_path']
        data_h5 = os.path.join(ROOT_DIR, 'data17_h5')
        if not os.path.exists(data_h5):
            os.makedirs(data_h5)
        self.dev_h5_path = os.path.join(data_h5, 'Dev.h5')
        self.eva_h5_path = os.path.join(data_h5, 'Eva.h5')
        self.dev_matrix_h5_path = os.path.join(data_h5, 'DevMatrix.h5')
        self.eva_matrix_h5_path = os.path.join(data_h5, 'EvaMatrix.h5')
        self.dev_matrix_fnames_h5_path = os.path.join(data_h5, 'DevMatrixFnames.h5')
        self.eva_matrix_fnames_h5_path = os.path.join(data_h5, 'EvaMatrixFanmes.h5')
        self.evaluation_setup_path = os.path.join(self.dev_path, 'evaluation_setup')
        self.dev_meta_path = os.path.join(self.dev_path, 'meta.txt')
        self.eva_meta_path = os.path.join(self.eva_path, 'meta.txt')
        # fname_encoder encode audio names to int, vice versa.
        self.fname_encoder = preprocessing.LabelEncoder()
        self.set_fname_encoder()

    def _get_wavelist_by_fold(self, fold_str='fold1', mode='train'):
        if mode == 'train':
            fold_meta = os.path.join(self.evaluation_setup_path, fold_str + '_train.txt')
        elif mode == 'test':
            fold_meta = os.path.join(self.evaluation_setup_path, fold_str + '_evaluate.txt')
        wave_list = []
        with open(fold_meta, 'r') as fp:
            for l in fp.readlines():
                audio_name, _ = l.split()
                wave_list.append(os.path.basename(audio_name))

        return wave_list

    def _get_wavelist_by_split(self, split='dev'):
        if split == 'dev':
            split_meta = self.dev_meta_path
        else:
            split_meta = self.eva_meta_path

        wave_list = []
        with open(split_meta, 'r') as fp:
            for l in fp.readlines():
                audio_name, _, _ = l.split()
                wave_list.append(os.path.basename(audio_name))

        return wave_list

    def extract_logmel(self, wav_path):
        """        self.fname_encoder = preprocessing.LabelEncoder()
        Give a wav, extract logmel feature
        :param wav_path:
        :return: fea of dim (1, frequency, time), first dim is added
        """
        x, sr = librosa.load(wav_path, sr=None, mono=False)
        assert (x.shape == (2, 441001) and sr == 44100)

        # 40ms winlen, half overlap
        y = librosa.feature.melspectrogram(x[0],
                                           sr=int(self.config['logmel']['sr']),
                                           n_fft=int(self.config['logmel']['n_fft']),
                                           hop_length=int(self.config['logmel']['hop_length']),
                                           n_mels=int(self.config['logmel']['n_mels'])
                                           )
        # about 1e-7
        EPS = np.finfo(np.float32).eps
        fea = np.log(y+EPS)
        # add a new axis
        return np.expand_dims(fea[:, :-1], axis=0)

    def extract_fea_for_datagroup(self, data_group, is_dev):
        """
        Loop through meta file, and extract logmel for each audio, sore in a h5 group
        :param data_group: a hdf5 group(group works like dictionary)
        :param is_dev
        :return:
        """
        if is_dev:
            fp = open(self.dev_meta_path, 'r')
            audio_dir = self.dev_path
        else:
            fp = open(self.eva_meta_path, 'r')
            audio_dir = self.eva_path

        for i, line in tqdm(enumerate(fp)):
            audio_name, label, _ = line.split()
            audio_path = os.path.join(audio_dir, audio_name)
            fea = self.extract_logmel(wav_path=audio_path)

            wav_name = os.path.basename(audio_path)
            data_group[wav_name] = fea
            data_group[wav_name].attrs['label'] = label

    def set_fname_encoder(self):
        """
        TODO: This may change API!!!!
        meta_path store all dev files and labels, fit all wav_names to get an encoder
        the encoder could transform string to int vice versa.
        :return:
        """
        wav_names = []

        for meta_path in [self.dev_meta_path, self.eva_meta_path]:
            fp = open(meta_path, 'r')

            for i, line in tqdm(enumerate(fp)):
                audio_name, _, _ = line.split()
                wav_name = os.path.basename(audio_name)
                wav_names.append(wav_name)

        self.fname_encoder.fit(wav_names)

    def extract_npy(self, fold='fold1', mode='train'):
        """
        Extract data and label as numpy array from dev.h5
        :param fold in 1 to 4
        :param mode: 'train' or 'test'
        :return: data and label as numpy array
        """

        data = []
        label = []
        with h5py.File(self.dev_h5_path, 'r') as f:
            audios = self._get_wavelist_by_fold(fold_str=fold, mode=mode)
            for audio in audios:
                # extract according to fold and mode
                data.append(np.array(f['dev'][audio].value))
                label.append(np.array(f['dev'][audio].attrs['label']))
        # concat data along existing axis 0
        data = np.concatenate(data, axis=0)
        le = preprocessing.LabelBinarizer()
        label_onehot = le.fit_transform(np.array(label))
        return data, label_onehot

    def extract_npy_fnames(self, fold='fold1', mode='train'):
        """
        Extract data, label, fnames(encoded as int) as numpy array from dev.h5
        :param fold in 1 to 4
        :param mode: 'train' or 'test'
        :return: data and label as numpy array
        """

        data = []
        label = []
        fnames = []
        with h5py.File(self.dev_h5_path, 'r') as f:
            audios = self._get_wavelist_by_fold(fold_str=fold, mode=mode)
            for audio in audios:
                # extract according to fold and mode
                data.append(np.array(f['dev'][audio].value))
                label.append(np.array(f['dev'][audio].attrs['label']))
                fnames.append(audio)
        # concat data along existing axis 0
        data = np.concatenate(data, axis=0)
        le = preprocessing.LabelBinarizer()
        label_onehot = le.fit_transform(np.array(label))
        fnames_codes = self.fname_encoder.transform(fnames)
        return data, label_onehot, fnames_codes

    def extract_npy_fnames_eva(self, split='dev'):

        if split == 'dev':
            h5_path = self.dev_h5_path
        else:
            h5_path = self.eva_h5_path

        data = []
        label = []
        fnames = []
        with h5py.File(h5_path, 'r') as f:
            audios = self._get_wavelist_by_split(split=split)
            for audio in audios:
                # extract according to fold and mode
                data.append(np.array(f[split][audio].value))
                label.append(np.array(f[split][audio].attrs['label']))
                fnames.append(audio)
        # concat data along existing axis 0
        data = np.concatenate(data, axis=0)
        le = preprocessing.LabelBinarizer()
        label_onehot = le.fit_transform(np.array(label))
        fnames_codes = self.fname_encoder.transform(fnames)
        return data, label_onehot, fnames_codes

    def create_devh5(self):
        """
        Extract LogMel and Store in h5 File, index by wav name
        :return:
        """
        if os.path.exists(self.dev_h5_path):
            print("[LOGGING]: " + self.dev_h5_path + " exists!")
            return

        with h5py.File(self.dev_h5_path, 'w') as f:

            dev = f.create_group('dev')
            self.extract_fea_for_datagroup(dev, is_dev=True)

        f.close()

    def create_evah5(self):
        if os.path.exists(self.eva_h5_path):
            print("[LOGGING]: " + self.eva_h5_path + " exists!")
            return

        with h5py.File(self.eva_h5_path, 'w') as f:

            eva = f.create_group('eva')
            self.extract_fea_for_datagroup(eva, is_dev=False)

        f.close()

    def create_dev_matrix(self):
        """
        Store train and test data in h5
        :return: None
        """
        if os.path.exists(self.dev_matrix_h5_path):
            print("[LOGGING]: " + self.dev_matrix_h5_path + " exists!")
            return

        with h5py.File(self.dev_matrix_h5_path, 'w') as f:

            for fold_idx in range(1, 5):
                for mode in ['train', 'test']:
                    fold_str = 'fold' + str(fold_idx)
                    grp = f.create_group(fold_str + '/' + mode)
                    grp['data'], grp['label'] = self.extract_npy(fold=fold_str, mode=mode)

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

            for fold_idx in range(1, 5):
                for mode in ['train', 'test']:
                    fold_str = 'fold' + str(fold_idx)
                    grp = f.create_group(fold_str + '/' + mode)
                    grp['data'], grp['label'], grp['fnames'] = self.extract_npy_fnames(fold=fold_str, mode=mode)
        f.close()

    def create_eva_matrix_fnames(self):
        if os.path.exists(self.eva_matrix_fnames_h5_path):
            print("[LOGGING]: " + self.eva_matrix_fnames_h5_path + " exists!")
            return

        with h5py.File(self.eva_matrix_fnames_h5_path, 'w') as f:
            for split in ['dev', 'eva']:
                grp = f.create_group(split)
                grp['data'], grp['label'], grp['fnames'] = self.extract_npy_fnames_eva(split=split)
        f.close()

    def load_dev(self, mode='train', fold_idx=1):
        """
        Give mode and device, load data and label as numpy array
        :param mode:
        :param fold_idx: 1,2 ,3,4
        :return: data, label as np array
        """
        if not os.path.exists(self.dev_matrix_h5_path):
            print(self.dev_matrix_h5_path + "not exists!")
            sys.exit()

        with h5py.File(self.dev_matrix_h5_path, 'r') as f:
            fold_str = 'fold' + str(fold_idx)
            data = np.array(f[fold_str][mode]['data'].value)
            label = np.array(f[fold_str][mode]['label'].value)

            if self.verbose:
                print("[LOGGING]: Loading", fold_str, mode, "of shape: ", data.shape)
            return data, label

    def load_dev_with_fnames(self, mode='train', fold_idx=1):
        """
        :param mode:'train' / 'test'
        :param fold_idx: 1, 2, 3,4
        :return: data, label, wav file names int coded, numpy arrays
        """
        if not os.path.exists(self.dev_matrix_fnames_h5_path):
            print("[LOGGING]: " + self.dev_matrix_fnames_h5_path + "not exists!")
            sys.exit()
        with h5py.File(self.dev_matrix_fnames_h5_path, 'r') as f:
            fold_str = 'fold' + str(fold_idx)
            data = np.array(f[fold_str][mode]['data'].value)
            label = np.array(f[fold_str][mode]['label'].value)
            fname = np.array(f[fold_str][mode]['fnames'].value)

            if self.verbose:
                print("[LOGGING]: Loading", fold_str, mode, "of shape: ", data.shape)
            return data, label, fname

    def load_eva_with_fnames(self, split='dev'):
        if not os.path.exists(self.eva_matrix_fnames_h5_path):
            print("[LOGGING]: " + self.eva_matrix_fnames_h5_path + "not exists!")
            sys.exit()
        with h5py.File(self.eva_matrix_fnames_h5_path, 'r') as f:
            data = np.array(f[split]['data'].value)
            label = np.array(f[split]['label'].value)
            fname = np.array(f[split]['fnames'].value)

            if self.verbose:
                print("[LOGGING]: Loading", split, "of shape: ", data.shape)
            return data, label, fname

    def show_spec_by_name(self, wav_name):
        """
        Given wav name, plot LogMel Spectrogram
        :param wav_name:
        :return:
        """
        with h5py.File(self.dev_h5_path, 'r') as f:

            spec_data = np.array(f['dev'][wav_name].value)

            plt.figure()
            librosa.display.specshow(spec_data[0])

            plt.title(wav_name)
            plt.show()

    def load_spec_by_name(self, wav_name):
        with h5py.File(self.dev_h5_path, 'r') as f:
            spec_data = np.array(f['dev'][wav_name].value)
            return spec_data

    def get_label_by_name(self, wav_name):
        with h5py.File(self.dev_h5_path, 'r') as f:
            label = str(f['dev'][wav_name].attrs['label'])
            return label


if __name__ == '__main__':

    data_manager = Dcase17Data()
    data_manager.create_devh5()
    data_manager.create_dev_matrix()
    data_manager.create_dev_matrix_fnames()
    data, label = data_manager.load_dev(fold_idx=1, mode='test')

    data_manager.create_evah5()
    data_manager.create_eva_matrix_fnames()
    data_manager.load_eva_with_fnames(split='dev')

