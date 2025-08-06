import random
import numpy as np
import h5py

__all__ = ["SignalDataLoader"]


class SignalDataLoader(object):
    def __init__(self, mod_type=[],snr_type=[]):
        mods = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC','FM', 'GMSK', 'OQPSK']  # "CPFSK"
        with h5py.File(r'/home/sangruijie_qyh/Data/Signal/Data2018_5.hdf5', 'r+') as h5file:
            allX = np.asarray(h5file['X'][:])
            allY = np.asarray(h5file['Y'][:])   #独热编码
            allZ = np.asarray(h5file['Z'][:])

        X = np.asarray(allX)
        X = np.transpose(X, (0, 2, 1))#对数据进行变换
        Y = np.asarray(allY)
        Z = np.asarray(allZ)

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

        # 定义比例因子，控制噪声强度与数据大小的关系
        scaling_factor = 0.2  # 可调整比例因子

        # 计算每个数据样本的标准差，以此为依据动态调整噪声强度
        noise_std_per_sample = np.std(X[valid_idx], axis=(1, 2), keepdims=True) * scaling_factor

        # 生成与 IQ_data 形状相同的噪声，每个样本有不同的噪声强度
        noise = np.random.normal(0, noise_std_per_sample, X[valid_idx].shape)

        # 将噪声添加到 IQ 数据中
        X[valid_idx] = X[valid_idx] + noise

        # 定义比例因子，控制噪声强度与数据大小的关系
        scaling_factor = 0.2  # 可调整比例因子

        # 计算每个数据样本的标准差，以此为依据动态调整噪声强度
        noise_std_per_sample = np.std(X[test_idx], axis=(1, 2), keepdims=True) * scaling_factor

        # 生成与 IQ_data 形状相同的噪声，每个样本有不同的噪声强度
        noise = np.random.normal(0, noise_std_per_sample, X[test_idx].shape)

        # 将噪声添加到 IQ 数据中
        X[test_idx] = X[test_idx] + noise

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