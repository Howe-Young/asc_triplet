import os
import h5py
import numpy as np


class Dcase17Standarizer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.dev_scaler_h5 = os.path.dirname(data_manager.dev_h5_path) + '/DevMatrixScaler.h5'
        self.eva_scaler_h5 = os.path.dirname(data_manager.eva_h5_path) + '/EvaMatrixScaler.h5'

    def calc_mu_sigma(self, data):
        """
        calc mu and sigma give data numpy
        :param data: (batch, frequency, time)
        :return: mu, sigma numpy vector
        """
        # transpose to (batch, time, frequency)
        data = np.transpose(data, [0, 2, 1])
        # mean, std
        mu = np.mean(data, axis=(0, 1))
        sigma = np.std(data, axis=(0, 1))

        return mu, sigma

    def mu_sigma_by_fold(self, mode='train', fold_idx=1):
        """
        given mode and device, return mu and sigma
        :param mode:
        :param fold_idx:
        :return:
        """

        data, _ = self.data_manager.load_dev(mode=mode, fold_idx=fold_idx)
        return self.calc_mu_sigma(data)

    def mu_sigma_by_split(self, split='dev'):
        data, _, _ = self.data_manager.load_eva_with_fnames(split=split)
        return self.calc_mu_sigma(data)

    def create_scaler_h5(self):
        """
        create scaler h5 file, data is f[fold][mode]['mu'] and f[fold][mode]['sigma']
        :return:
        """
        if os.path.exists(self.dev_scaler_h5):
            print("[LOGGING]: " + self.dev_scaler_h5 + " exists!")
            return
        with h5py.File(self.dev_scaler_h5, 'w') as f:
            for fold_idx in range(1, 5):
                fold_str = 'fold' + str(fold_idx)
                for mode in ['train', 'test']:
                    grp = f.create_group(fold_str + '/' + mode)
                    grp['mu'], grp['sigma'] = self.mu_sigma_by_fold(mode=mode, fold_idx=fold_idx)
        f.close()

    def create_eva_scaler_h5(self):
        if os.path.exists(self.eva_scaler_h5):
            print("[LOGGING]: " + self.eva_scaler_h5 + " exists!")
            return
        with h5py.File(self.eva_scaler_h5, 'w') as f:
            for split in ['dev', 'eva']:
                grp = f.create_group(split)
                grp['mu'], grp['sigma'] = self.mu_sigma_by_split(split=split)

    def load_scaler(self, mode='train', fold_idx=1):
        """
        load mu and sigma given mode and device
        :param mode:
        :param fold_idx:
        :return:
        """
        with h5py.File(self.dev_scaler_h5, 'r') as f:
            fold_str = 'fold' + str(fold_idx)
            mu = f[fold_str][mode]['mu'].value
            sigma = f[fold_str][mode]['sigma'].value
            return mu, sigma

    def load_eva_scaler(self, split='dev'):
        with h5py.File(self.eva_scaler_h5, 'r') as f:
            mu = f[split]['mu'].value
            sigma = f[split]['sigma'].value
            return mu, sigma

    def load_dev_standrized(self, fold_idx=1, mode='train'):
        """
        given mode and device, scale using train scaler , and return data and label
        :param mode:
        :param fold_idx:
        :return:
        """
        data, label = self.data_manager.load_dev(mode=mode, fold_idx=fold_idx)
        # Take Care!!! only scale using train mu and sigma
        mu, sigma = self.load_scaler(mode='train', fold_idx=fold_idx)

        data = np.transpose(data, [0, 2, 1])
        data = (data - mu) / sigma
        # transpose back to (batch, frequency, time)
        data = np.transpose(data, [0, 2, 1])

        print("[LOGGING] Normalize using fold {}".format(str(fold_idx)))

        return data, label

    def load_dev_fname_standrized(self, mode='train', fold_idx=1):
        """
        given mode and device, scale using train scaler , and return data and label
        :param mode:
        :param fold_idx:
        :return:
        """
        data, label, fnames = self.data_manager.load_dev_with_fnames(mode=mode, fold_idx=fold_idx)
        # Take Care!!! only scale using train mu and sigma
        mu, sigma = self.load_scaler(mode='train', fold_idx=fold_idx)

        data = np.transpose(data, [0, 2, 1])
        data = (data - mu) / sigma
        # transpose back to (batch, frequency, time)
        data = np.transpose(data, [0, 2, 1])

        print("[LOGGING] Normalize using fold {}".format(str(fold_idx)))

        return data, label, fnames

    def load_eva_fname_standrized(self, split='dev'):
        data, label, fnames = self.data_manager.load_eva_with_fnames(split=split)
        # Take Care!!! only scale using train mu and sigma
        mu, sigma = self.load_eva_scaler(split='dev')

        data = np.transpose(data, [0, 2, 1])
        data = (data - mu) / sigma
        # transpose back to (batch, frequency, time)
        data = np.transpose(data, [0, 2, 1])

        print("[LOGGING] Normalize using split {}".format('dev'))

        return data, label, fnames

    def load_normed_spec_by_name(self, wav_name=None, fold_idx=1):
        spec_data = self.data_manager.load_spec_by_name(wav_name)
        # Take Care!!! only scale using train mu and sigma

        mu, sigma = self.load_scaler(mode='train', fold_idx=fold_idx)

        spec_data = np.transpose(spec_data, [0, 2, 1])
        spec_data = (spec_data - mu) / sigma
        # transpose back to (batch, frequency, time)
        spec_data = np.transpose(spec_data, [0, 2, 1])

        return spec_data

    def get_label_by_name(self, wav_name=None):
        return self.data_manager.get_label_by_name(wav_name)


if __name__ == '__main__':
    from data_manager.dcase17_manager import Dcase17Data
    ori_data_manager = Dcase17Data()
    ori_standarizer = Dcase17Standarizer(data_manager=ori_data_manager)
    ori_standarizer.create_scaler_h5()
    ori_standarizer.create_eva_scaler_h5()

