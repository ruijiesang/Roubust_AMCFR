import torch
import torch.nn as nn
from amr.dataloaders.dataloader import *
from amr.utils import *
from amr.utils.solver1 import *
from DAELSTM_config import *
import os
from amr.utils.log_train_info import train_info


def main(cfgs,c,t,f):
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(cfgs.seed, cfgs.cpu, cfgs.gpu)
    print(device, pin_memory)

    path = '/train_DAELSTM/Past_result/results_0/DAELSTM/2016b/loss_CE/0t'
    train_loader, valid_loader, test_loader, snrs, mods = AMRDataLoader(dataset=cfgs.dataset,
                                                                        Xmode=cfgs.params["Xmode"][0],
                                                                        batch_size=cfgs.params["batch_size"],
                                                                        num_workers=cfgs.workers,
                                                                        pin_memory=pin_memory,
                                                                        mod_type=cfgs.mod_type,
                                                                        snr_type=cfgs.snr_type )()


    criterion = init_loss_FG(cfgs.params["loss"],len(cfgs.mod_type),int(c),float(t))


    # 单个模型测试加载，测试前重新加载最优模型
    cfgs.train = False
    model = init_model(cfgs, cfgs.params["network"],path)
    model.to(device)

    # 单个模型测试
    test_loss, test_acc, f1,test_conf, test_conf_snr, test_acc_snr = Tester1(model=model, device=device,
                                                                         criterion=criterion,
                                                                         classes=len(cfgs.mod_type),
                                                                         snrs=snrs)(train_loader)
    print("Done!!!")
if __name__ == '__main__':
    c = [6]
    t = [ 0.0000001]
    f=[0]
    best_acc=0
    for f1 in f:
        for c1 in c:
            for t1 in t:
                cfgs = get_cfgs_2016a_Notrain()
                cfgs.seed = f1 + cfgs.seed
                main(cfgs,c1,t1,f1)