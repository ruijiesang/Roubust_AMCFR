import random
import numpy as np
import h5py

__all__ = ["SignalDataLoader"]


class SignalDataLoader(object):
    def __init__(self, mod_type=[],snr_type=[]):
        mods = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC','FM', 'GMSK', 'OQPSK']  # "CPFSK"
        with h5py.File(r'/home/sangruijie_qyh/Data/Signal/GOLD_XYZ_OSC.0001_1024.hdf5', 'r+') as h5file:
            allX = np.asarray(h5file['X'][:])
            allY = np.asarray(h5file['Y'][:])   #独热编码
            allZ = np.asarray(h5file['Z'][:])
        mod_type = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK',
                '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
                'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        snr_type=[0,2,4,6,8,10,12,14,16,18]
        allY=np.argmax(allY,axis=1)
        allZ=allZ.reshape(-1)
        X = []
        Y = []
        Z = []

        #将mod_type转化为number
        mod_type_number=[]
        for i in mod_type:
            mod_type_number.append(mods.index(i))

        for idx in range(allX.shape[0]):
            if allY[idx] in mod_type_number and allZ[idx] in snr_type:
                X.append(allX[idx])
                Y.append(allY[idx])
                Z.append(allZ[idx])

        del allX
        del allY
        del allZ
        X = np.asarray(X)
        X = np.transpose(X, (0, 2, 1))#对数据进行变换
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        self.snrs = np.unique(Z).tolist()
        self.mods = mod_type

        n_examples = X.shape[0]
        n_train = int(0.5 * n_examples)
        n_valid = int(0.25 * n_examples)

        allnum = list(range(0, n_examples))
        random.shuffle(allnum)

        train_idx = allnum[0:n_train]
        valid_idx = allnum[n_train:n_train + n_valid]
        test_idx = allnum[n_train + n_valid:]
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