from yacs.config import CfgNode as CN
import argparse

__all__ = ['get_cfgs','get_cfgs_Cen']#Used for being called by other codes

_C = CN()

_C.method = 'Model'
_C.train = True
_C.dataset = 'cfo'
_C.mod_type = ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM32", "QAM64", "QAM128", "QAM256", "GFSK", "WBFM", "AM-DSB", "AM-SSB", "OOK", "4ASK", "8ASK", "16PSK", "32PSK","8APSK","GMSK", "DQPSK","16APSK","32APSK","64APSK","128APSK"]
_C.snr_type = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
_C.scal = 0
_C.workers = 8
_C.seed = 1
_C.gpu = 0
_C.cpu = False


_C.params = CN()
_C.params.network = 'Model'
_C.params.loss = "loss_CE"
_C.params.codename = "2016a"
_C.params.loss_c= 11
_C.params.loss_t= 2
_C.params.loss_alp= 0.01
_C.params.batch_size = 1024
_C.params.epochs = 200
_C.params.lr = 5e-3
_C.params.lr_decay = 0.8
_C.params.weight_decay = 5e-2
_C.params.early_stop = False
_C.params.Xmode = [{"type": "APF", "options": {"IQ_norm": False, "zero_mask": False}}]


def get_cfg_defaults():
    return _C.clone()


def get_cfgs():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-FR/Roubust_AMCFR/train_Resnet/Resnet_configs/Resnet_train.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_Cen():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-FR/Roubust_AMCFR/train_Resnet/Resnet_configs/Resnet_train_Cen.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

