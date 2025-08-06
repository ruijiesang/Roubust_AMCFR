import os
from datetime import datetime
import sys
import traceback
import h5py
import numpy as np
import torch
import pickle
from torch.onnx.symbolic_opset9 import new_zeros

DEBUG = -1
INFO = 0
EMPH = 1
WARNING = 2
ERROR = 3
FATAL = 4

log_level = INFO
line_seg = ''.join(['*'] * 65)


class LoggerFatalError(SystemExit):
    pass


def _format(level, messages):
    timestr = datetime.strftime(datetime.now(), '%m.%d/%H:%M')
    father = traceback.extract_stack()[-4]
    func_info = f'{father[0].split("/")[-1]}:{str(father[1]).ljust(4, " ")}'
    m = ' '.join(map(str, messages))
    msg = f'{level} {timestr} {func_info}] {m}'
    return msg


_log_file = None
_log_buffer = []
_RED = '\033[0;31m'
_GREEN = '\033[1;32m'
_LIGHT_RED = '\033[1;31m'
_ORANGE = '\033[0;33m'
_YELLOW = '\033[1;33m'
_NC = '\033[0m'  # No Color


def set_file(fname):
    global _log_file
    global _log_buffer
    if _log_file is not None:
        warning("Change log file to %s" % fname)
        _log_file.close()
    _log_file = open(fname, 'w')
    if len(_log_buffer):
        for s in _log_buffer:
            _log_file.write(s)
        _log_file.flush()


def debug(*messages, file=None):
    if log_level > DEBUG:
        return
    msg = _format('D', messages)

    if file is None:
        sys.stdout.write(_YELLOW + msg + _NC + '\n')
        sys.stdout.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def info(*messages, file=None):
    if log_level > INFO:
        return
    msg = _format('I', messages)
    if file is None:
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()
    else:
        path = file + '/test_info.csv'
        with open(path, 'a+') as f:
            print(msg, file=f)


def emph(*messages, file=None):
    if log_level > EMPH:
        return
    msg = _format('EM', messages)
    if file is None:
        sys.stdout.write(_GREEN + msg + _NC + '\n')
        sys.stdout.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def warning(*messages, file=None):
    if log_level > WARNING:
        return
    msg = _format('W', messages)
    if file is None:
        sys.stderr.write(_ORANGE + msg + _NC + '\n')
        sys.stderr.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def error(*messages, file=None):
    if log_level > ERROR:
        return
    msg = _format('E', messages)
    if file is None:
        sys.stderr.write(_RED + msg + _NC + '\n')
        sys.stderr.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def fatal(*messages, file=None):
    if log_level > FATAL:
        return
    msg = _format('F', messages)
    if file is None:
        sys.stderr.write(_LIGHT_RED + msg + _NC + '\n')
        sys.stderr.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)

    raise LoggerFatalError(-1)

def save_output_to_hdf5(epoch, predictions, labels, path='model_output.h5'):
    file_name = os.path.join(path, 'model_output.h5')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # 确保张量在 CPU 上，并使用 detach 取消梯度追踪
    predictions = [torch.tensor(pred) if isinstance(pred, np.ndarray) else pred for pred in predictions]
    labels = [torch.tensor(label) if isinstance(label, np.ndarray) else label for label in labels]

    # 将 predictions 和 labels 转换为张量
    predictions_tensor = torch.stack(predictions)#看prediction的格式，如果是只有一列直接tensor，否则需要stack
    labels_tensor = torch.tensor(labels)

    # 将 epoch 扩展维度并重复，使其大小与 predictions 和 labels 匹配
    epoch_tensor = torch.tensor(epoch).repeat(predictions_tensor.size(0), 1)

    # 合并 epoch、predictions 和 labels 为一个大的张量
    combined_tensor = torch.cat((epoch_tensor, predictions_tensor, labels_tensor.unsqueeze(1)), dim=1)#看prediction的格式，如果是只有一列需要unsqueeze

    with h5py.File(file_name, 'a') as hdf_file:
        if 'combined_output' not in hdf_file:
            # 如果数据集不存在，创建并存储
            hdf_file.create_dataset('combined_output', data=combined_tensor.numpy(), maxshape=(None, combined_tensor.size(1)))
        else:
            # 如果存在，则扩展并追加新数据
            hdf_file['combined_output'].resize((hdf_file['combined_output'].shape[0] + combined_tensor.size(0)), axis=0)
            hdf_file['combined_output'][-combined_tensor.size(0):] = combined_tensor.numpy()



def save_output_to_hdf5_TSNE(epoch, predictions, labels, path='model_output.h5'):
    file_name = os.path.join(path, 'TSNE_model_output.h5')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # 确保张量在 CPU 上，并使用 detach 取消梯度追踪
    predictions = [torch.tensor(pred).detach().cpu() if isinstance(pred, np.ndarray) else pred.detach().cpu() for pred in predictions]
    labels = [torch.tensor(label).detach().cpu() if isinstance(label, np.ndarray) else label.detach().cpu() for label in labels]

    # 将 predictions 和 labels 转换为张量
    predictions_tensor = torch.stack(predictions)
    labels_tensor = torch.stack(labels)

    # 将 epoch 扩展维度并重复，使其大小与 predictions 和 labels 匹配
    epoch_tensor = torch.tensor(epoch).repeat(predictions_tensor.size(0), 1)

    # 合并 epoch、predictions 和 labels 为一个大的张量
    combined_tensor = torch.cat([epoch_tensor, predictions_tensor, labels_tensor.unsqueeze(1)], dim=1)

    with h5py.File(file_name, 'a') as hdf_file:
        if 'combined_output' not in hdf_file:
            # 如果数据集不存在，创建并存储
            hdf_file.create_dataset('combined_output', data=combined_tensor.numpy(), maxshape=(None, combined_tensor.size(1)))
        else:
            # 如果存在，则扩展并追加新数据
            hdf_file['combined_output'].resize((hdf_file['combined_output'].shape[0] + combined_tensor.size(0)), axis=0)
            hdf_file['combined_output'][-combined_tensor.size(0):] = combined_tensor.numpy()


def save_data_noshink(X,Y,Y_pre,path):

    # 在第二个维度（dim=1）上求最大值和对应的索引
    max_values, max_indices = torch.max(Y_pre, dim=1)

    # 找到最大值不超过 0.8 的索引
    indices_below_threshold = (max_values <= 0.8).nonzero(as_tuple=True)[0]

    Y=Y.unsqueeze(1)
    # 转换为 NumPy 数组
    new_X_np = X[indices_below_threshold].cpu().numpy()

    new_Y_np = Y[indices_below_threshold].cpu().numpy()

    new_z = torch.ones(new_X_np.shape[0], 1)

    new_Z_np = new_z.cpu().numpy()

    file_path=path+'/Noshink_data.h5'
    # 打开现有的 HDF5 文件并追加数据
    with h5py.File(file_path, 'a') as hdf_file:
        # 追加 X
        if 'X' in hdf_file:
            # 获取现有数据集的形状
            old_shape_X = hdf_file['X'].shape  # 例如 (1000, 2, 128)
            # 扩展数据集的第一个维度（批次大小）
            new_shape_X = (old_shape_X[0] + new_X_np.shape[0], old_shape_X[1], old_shape_X[2])
            hdf_file['X'].resize(new_shape_X)  # 修改数据集的形状
            # 追加新的 X 数据，注意 [-new_X_np.shape[0]:] 表示从最后的空位开始填充
            hdf_file['X'][-new_X_np.shape[0]:] = new_X_np
        else:
            # 创建新的数据集，指定 maxshape 以允许第一个维度扩展
            hdf_file.create_dataset('X', data=new_X_np, maxshape=(None, new_X_np.shape[1], new_X_np.shape[2]))

        # 追加 Y
        if 'Y' in hdf_file:
            old_shape_Y = hdf_file['Y'].shape
            new_shape_Y = (old_shape_Y[0] + new_Y_np.shape[0], old_shape_Y[1])
            hdf_file['Y'].resize(new_shape_Y)
            hdf_file['Y'][-new_Y_np.shape[0]:] = new_Y_np  # 追加新的 Y 数据
        else:
            hdf_file.create_dataset('Y', data=new_Y_np, maxshape=(None, new_Y_np.shape[1]))

        # 追加 Z
        if 'Z' in hdf_file:
            old_shape_Z = hdf_file['Z'].shape
            new_shape_Z = (old_shape_Z[0] + new_Z_np.shape[0], old_shape_Z[1])
            hdf_file['Z'].resize(new_shape_Z)
            hdf_file['Z'][-new_Z_np.shape[0]:] = new_Z_np  # 追加新的 Z 数据
        else:
            hdf_file.create_dataset('Z', data=new_Z_np, maxshape=(None, new_Z_np.shape[1]))

def save_data_to_pkl(X,Y,Z,Y_soft, path='model_output.pkl'):
    # 假设 X, Y, Z, Y_soft 已经是 tensor
    # 示例数据
    # X, Y, Z 维度应一致，Y_soft 的维度应为 [N, num_classes]

    # 计算 Y_soft 的最大值和第二大值
    max_scores, _ = torch.max(Y_soft, dim=1)  # 每个样本的最大值
    sorted_scores, _ = torch.sort(Y_soft, dim=1, descending=True)  # 对每个样本排序
    second_max_scores = sorted_scores[:, 1]  # 第二大值

    # 计算最大值和第二大值之间的差值
    score_diff = max_scores - second_max_scores

    # 筛选出差值不超过0.25的样本
    valid_samples = score_diff <= 0.25

    # 根据 valid_samples 筛选出对应的 X, Y, Z 样本
    X_valid = X[valid_samples]
    Y_valid = Y[valid_samples]
    Z_valid = Z[valid_samples]

    # 保存筛选后的 X, Y, Z 到一个 pkl 文件
    output_data = {'X': X_valid, 'Y': Y_valid, 'Z': Z_valid}

    # 保存为 pkl 文件
    with open(path+'/filtered_data.pkl', 'wb') as f:
        pickle.dump(output_data, f)

    print("Filtered data saved to 'filtered_data.pkl'")