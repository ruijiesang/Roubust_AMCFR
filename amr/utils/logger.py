import os
from datetime import datetime
import sys
import traceback
import h5py
import numpy as np
import torch
import pickle


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

    # Make sure the tensor is on the CPU and use detach to cancel gradient tracking.
    predictions = [torch.tensor(pred) if isinstance(pred, np.ndarray) else pred for pred in predictions]
    labels = [torch.tensor(label) if isinstance(label, np.ndarray) else label for label in labels]

    # Convert the predictions and labels into tensors
    predictions_tensor = torch.stack(predictions)#Look at the format of the prediction. If there is only one column, it is a direct tensor. Otherwise, it needs to be stacked.
    labels_tensor = torch.tensor(labels)

    # Expand the dimension of epoch and repeat it to make its size match that of predictions and labels.
    epoch_tensor = torch.tensor(epoch).repeat(predictions_tensor.size(0), 1)

    # Merge the epoch, predictions and labels into a single large tensor
    combined_tensor = torch.cat((epoch_tensor, predictions_tensor, labels_tensor.unsqueeze(1)), dim=1)#Looking at the format of the prediction, if there is only one column, then you need to use `unsqueeze`.

    with h5py.File(file_name, 'a') as hdf_file:
        if 'combined_output' not in hdf_file:
            # If the dataset does not exist, create and store it.
            hdf_file.create_dataset('combined_output', data=combined_tensor.numpy(), maxshape=(None, combined_tensor.size(1)))
        else:
            # If it exists, then expand and append the new data
            hdf_file['combined_output'].resize((hdf_file['combined_output'].shape[0] + combined_tensor.size(0)), axis=0)
            hdf_file['combined_output'][-combined_tensor.size(0):] = combined_tensor.numpy()



def save_output_to_hdf5_TSNE(epoch, predictions, labels, path='model_output.h5'):
    file_name = os.path.join(path, 'TSNE_model_output.h5')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Make sure the tensor is on the CPU and use detach to cancel gradient tracking.
    predictions = [torch.tensor(pred).detach().cpu() if isinstance(pred, np.ndarray) else pred.detach().cpu() for pred in predictions]
    labels = [torch.tensor(label).detach().cpu() if isinstance(label, np.ndarray) else label.detach().cpu() for label in labels]

    # Convert the predictions and labels into tensors
    predictions_tensor = torch.stack(predictions)
    labels_tensor = torch.stack(labels)

    # Expand the dimension of epoch and repeat it to make its size match that of predictions and labels.
    epoch_tensor = torch.tensor(epoch).repeat(predictions_tensor.size(0), 1)

    #  Merge the epoch, predictions and labels into a single large tensor
    combined_tensor = torch.cat([epoch_tensor, predictions_tensor, labels_tensor.unsqueeze(1)], dim=1)

    with h5py.File(file_name, 'a') as hdf_file:
        if 'combined_output' not in hdf_file:
            # If the dataset does not exist, create and store it.
            hdf_file.create_dataset('combined_output', data=combined_tensor.numpy(), maxshape=(None, combined_tensor.size(1)))
        else:
            # If it exists, then expand and append the new data
            hdf_file['combined_output'].resize((hdf_file['combined_output'].shape[0] + combined_tensor.size(0)), axis=0)
            hdf_file['combined_output'][-combined_tensor.size(0):] = combined_tensor.numpy()


def save_data_noshink(X,Y,Y_pre,path):

    # Find the maximum value and the corresponding index on the second dimension (dim = 1)
    max_values, max_indices = torch.max(Y_pre, dim=1)

    # Find the index where the maximum value does not exceed 0.8
    indices_below_threshold = (max_values <= 0.8).nonzero(as_tuple=True)[0]

    Y=Y.unsqueeze(1)
    # Convert to a NumPy array
    new_X_np = X[indices_below_threshold].cpu().numpy()

    new_Y_np = Y[indices_below_threshold].cpu().numpy()

    new_z = torch.ones(new_X_np.shape[0], 1)

    new_Z_np = new_z.cpu().numpy()

    file_path=path+'/Noshink_data.h5'

    with h5py.File(file_path, 'a') as hdf_file:

        if 'X' in hdf_file:

            old_shape_X = hdf_file['X'].shape  # 例如 (1000, 2, 128)

            new_shape_X = (old_shape_X[0] + new_X_np.shape[0], old_shape_X[1], old_shape_X[2])
            hdf_file['X'].resize(new_shape_X)

            hdf_file['X'][-new_X_np.shape[0]:] = new_X_np
        else:

            hdf_file.create_dataset('X', data=new_X_np, maxshape=(None, new_X_np.shape[1], new_X_np.shape[2]))


        if 'Y' in hdf_file:
            old_shape_Y = hdf_file['Y'].shape
            new_shape_Y = (old_shape_Y[0] + new_Y_np.shape[0], old_shape_Y[1])
            hdf_file['Y'].resize(new_shape_Y)
            hdf_file['Y'][-new_Y_np.shape[0]:] = new_Y_np
        else:
            hdf_file.create_dataset('Y', data=new_Y_np, maxshape=(None, new_Y_np.shape[1]))


        if 'Z' in hdf_file:
            old_shape_Z = hdf_file['Z'].shape
            new_shape_Z = (old_shape_Z[0] + new_Z_np.shape[0], old_shape_Z[1])
            hdf_file['Z'].resize(new_shape_Z)
            hdf_file['Z'][-new_Z_np.shape[0]:] = new_Z_np
        else:
            hdf_file.create_dataset('Z', data=new_Z_np, maxshape=(None, new_Z_np.shape[1]))

def save_data_to_pkl(X,Y,Z,Y_soft, path='model_output.pkl'):
    # Assume that X, Y, Z, Y_soft are tensors.
    # Example data
    # The dimensions of X, Y, and Z should be the same, and the dimension of Y_soft should be [N, num_classes]

    # Calculate the maximum and the second value of Y_soft
    max_scores, _ = torch.max(Y_soft, dim=1)  # The maximum value of each sample
    sorted_scores, _ = torch.sort(Y_soft, dim=1, descending=True)  # Sort each sample
    second_max_scores = sorted_scores[:, 1]  # the second value

    # Calculate the difference between the maximum value and the second value.
    score_diff = max_scores - second_max_scores

    # Select the samples whose deviation values do not exceed 0.25.
    valid_samples = score_diff <= 0.25

    # Select the corresponding X, Y, and Z samples based on the "valid_samples" criteria.
    X_valid = X[valid_samples]
    Y_valid = Y[valid_samples]
    Z_valid = Z[valid_samples]

    # Save the filtered X, Y, and Z data into a pkl file
    output_data = {'X': X_valid, 'Y': Y_valid, 'Z': Z_valid}

    # save to a pkl file
    with open(path+'/filtered_data.pkl', 'wb') as f:
        pickle.dump(output_data, f)

    print("Filtered data saved to 'filtered_data.pkl'")