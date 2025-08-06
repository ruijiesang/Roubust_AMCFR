from yacs.config import CfgNode as CN
import argparse
import importlib

__all__ = ['get_cfgs','get_cfgs_2016atest','get_cfgs_2016aCen',
           'get_cfgs_2016aCen_noisydata','get_cfgs_noisydata','get_cfgs_2016atest_noisydata',
           'get_cfgs_noisydatatrain','get_cfgs_2016atest_noisydatatrain','get_cfgs_2016aCen_noisydatatrain',
           'get_cfgs_noisydatatest','get_cfgs_2016atest_noisydatatest','get_cfgs_2016aCen_noisydatatest',
           'get_cfgs_noisydatatest_2','get_cfgs_2016atest_noisydatatest_2','get_cfgs_2016aCen_noisydatatest_2',
           'get_cfgs_noisydatatest_6','get_cfgs_2016atest_noisydatatest_6','get_cfgs_2016aCen_noisydatatest_6',
           'get_cfgs_noisydata_onlytest2', 'get_cfgs_noisydata_onlytest4', 'get_cfgs_noisydata_onlytest6',
           'get_cfgs_2016aCen_noisydata_onlytest2', 'get_cfgs_2016aCen_noisydata_onlytest4', 'get_cfgs_2016aCen_noisydata_onlytest6',
           'get_cfgs_2016atest_noisydata_onlytest2', 'get_cfgs_2016atest_noisydata_onlytest4', 'get_cfgs_2016atest_noisydata_onlytest6',
           'get_cfgs_noisydatatest_2TSNE','get_cfgs_2016a_Notrain','get_cfgs_origin']#用于被别的代码调用

_C = CN()

_C.method = 'ours'
_C.train = True
_C.dataset = 'cfo'
_C.mod_type = ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM32", "QAM64", "QAM128", "QAM256", "GFSK", "WBFM", "AM-DSB", "AM-SSB", "OOK", "4ASK", "8ASK", "16PSK", "32PSK","8APSK","GMSK", "DQPSK","16APSK","32APSK","64APSK","128APSK"]
_C.snr_type = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
_C.workers = 8
_C.seed = 1
_C.gpu = 0
_C.cpu = False


_C.params = CN()#创建一个子节点，便于参数的管理
_C.params.network = 'TransGroupNet'
_C.params.loss = "loss_CE_test2"
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
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs


def get_cfgs_origin():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_origin.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-FR/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_noisydata():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydata.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest_noisydata():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test_noisydata.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen_noisydata():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen_noisydata.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs
def get_cfgs_noisydatatrain():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydatatrain.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest_noisydatatrain():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test_noisydatatrain.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen_noisydatatrain():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen_noisydatatrain.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_noisydatatest():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydatatest.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest_noisydatatest():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test_noisydatatest.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen_noisydatatest():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen_noisydatatest.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_noisydatatest_2():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydatatest_2.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest_noisydatatest_2():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test_noisydatatest_2.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen_noisydatatest_2():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen_noisydatatest_2.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_noisydatatest_6():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydatatest_6.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest_noisydatatest_6():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test_noisydatatest_6.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen_noisydatatest_6():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen_noisydatatest_6.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_noisydata_onlytest2():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydata_onlytest2.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest_noisydata_onlytest2():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test_noisydata_onlytest2.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen_noisydata_onlytest2():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen_noisydata_onlytest2.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_noisydata_onlytest4():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydata_onlytest4.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest_noisydata_onlytest4():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test_noisydata_onlytest4.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen_noisydata_onlytest4():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen_noisydata_onlytest4.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_noisydata_onlytest6():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydata_onlytest6.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016atest_noisydata_onlytest6():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_test_noisydata_onlytest6.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016aCen_noisydata_onlytest6():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016aCen_noisydata_onlytest6.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_noisydatatest_2TSNE():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_noisydatatest_2TSNE.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs

def get_cfgs_2016a_Notrain():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/train_ThreeStream/ThreeStream_configs/ThreeStream_data2016a_Notrain.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs