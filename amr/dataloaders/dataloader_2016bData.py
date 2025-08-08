import random
import numpy as np
import pickle

__all__ = ["SignalDataLoader"]


class SignalDataLoader(object):
    def __init__(self, mod_type=[],snr_type=[],scal=0):
        mods = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']  # 10类

        data = pickle.load(open(r'/home/sangruijie_qyh/Data/Signal/RML2016.10b/RML2016.10b.dat', 'rb'),encoding='iso-8859-1')
        # read data
        data_keys=data.keys()

        X = []
        Y = []
        Z = []
        for idx in data_keys:                               #Reorganize the data structure
            if idx[0] in mod_type and idx[1] in snr_type:
                X.extend(data[idx])                         #data_shape[?,2,128]
                Y.extend([mod_type.index(idx[0])]*data[idx].shape[0])       #label[?,1]
                Z.extend([idx[1]]*data[idx].shape[0])                   #snr[?,1]

        #to numpy
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        self.snrs = snr_type
        self.mods = mod_type

        n_examples = X.shape[0]
        n_train = int(0.1 * n_examples)
        n_valid = int(0.4 * n_examples)

        allnum = list(range(0, n_examples))
        random.shuffle(allnum)

        train_idx = allnum[0:n_train]
        valid_idx = allnum[n_train:n_train + n_valid]
        test_idx = allnum[n_train + n_valid:]

        # Define the proportion factor and control the relationship between noise intensity and data size
        scaling_factor = scal

        # Calculate the standard deviation of each data sample
        noise_std_per_sample = np.std(X[valid_idx], axis=(1, 2), keepdims=True) * scaling_factor

        noise_v = np.random.normal(0, noise_std_per_sample, X[valid_idx].shape)

        noise_std_per_sample = np.std(X[test_idx], axis=(1, 2), keepdims=True) * scaling_factor

        noise_t = np.random.normal(0, noise_std_per_sample, X[test_idx].shape)

        if scal != 0:
            # add noise
            X[valid_idx] = X[valid_idx] + noise_v
            X[test_idx] =X[test_idx] + noise_t

        self.X_train = X[train_idx]
        self.Y_train = Y[train_idx]
        self.Z_train = Z[train_idx]
        self.X_valid = X[valid_idx]
        self.Y_valid = Y[valid_idx]
        self.Z_valid = Z[valid_idx]
        self.X_test = X[test_idx]
        self.Y_test = Y[test_idx]
        self.Z_test = Z[test_idx]
        del X
        del Y
        del Z

    def __call__(self):
        return self.X_train, self.Y_train, self.Z_train, self.X_valid, self.Y_valid, self.Z_valid, self.X_test, self.Y_test, self.Z_test, self.snrs, self.mods

if __name__ == '__main__':
    SignalDataLoader()