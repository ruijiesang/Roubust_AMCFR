import sys
sys.path.append('/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master')
import torch
import torch.nn as nn
from amr.dataloaders.dataloader import *
from amr.utils import *
from FEAT_config import *
from amr.utils.log_train_info import train_info

#将test_loader changse train_loasder
def main(cfgs,t):
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(cfgs.seed, cfgs.cpu, cfgs.gpu)
    print(device, pin_memory)

    path = './results/' + cfgs.method + '/' + cfgs.dataset + '/'  + cfgs.params["loss"]+'/' + str(t)+'t'

    train_loader, valid_loader, test_loader, snrs, mods = AMRDataLoader(dataset=cfgs.dataset,
                                                                        Xmode=cfgs.params["Xmode"][0],
                                                                        batch_size=cfgs.params["batch_size"],
                                                                        num_workers=cfgs.workers,
                                                                        pin_memory=pin_memory,
                                                                        mod_type=cfgs.mod_type,
                                                                        snr_type=cfgs.snr_type )()

    # 单个模型网络加载
    model = init_model(cfgs,cfgs.params["network"],path)
    model.to(device)

    # 单个模型损失函数
    if cfgs.params["loss"]=="loss_FG_test2":
        criterion = init_loss_FG(cfgs.params["loss"],len(cfgs.mod_type),float(t))
    else:
        criterion = init_loss(cfgs.params["loss"])




    # 单个模型训练
    if cfgs.train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfgs.params["lr"]), weight_decay=cfgs.params["weight_decay"])
        trainer = Trainer(model=model, device=device, optimizer=optimizer, lr_decay=cfgs.params["lr_decay"], criterion=criterion,
                          save_path=path,
                          early_stop=cfgs.params["early_stop"])
        train_loss, train_acc, valid_loss, valid_acc = trainer.loop(cfgs.params["epochs"], train_loader, valid_loader)

        # 训练下训练曲线
        draw_train(train_loss, train_acc, valid_loss, valid_acc,
                   save_path=path + '/draws')

    # 单个模型测试加载，测试前重新加载最优模型
    cfgs.train = False
    model = init_model(cfgs, cfgs.params["network"],path)
    model.to(device)

    # 单个模型测试
    test_loss, test_acc, f1,test_conf, test_conf_snr, test_acc_snr = Tester(model=model, device=device,
                                                                         criterion=criterion,
                                                                         classes=len(cfgs.mod_type),
                                                                         snrs=snrs)(test_loader)
    # 测试下总体混淆矩阵
    draw_conf(test_conf, save_path=path + '/draws',
              labels=mods, order="total")

    # 测试下不同snr的混淆矩阵
    for i in range(len(snrs)):
        logger.info(f'test_snr : {snrs[i]:.0f} | '
                    f'test_acc : {test_acc_snr[i]:.4f}')
        draw_conf(test_conf_snr[i],
                  save_path=path + '/draws',
                  labels=mods,
                  order=str(snrs[i]))

    # 测试下总体准确率曲线
    draw_acc(snrs, test_acc_snr,
             save_path=path + '/draws')

    train_info(train_loss, train_acc, valid_loss, valid_acc, path)

    highest_acc_snr = test_acc_snr.max().item()  # 获取单个 SNR 下的最高精度
    highest_acc_snr_idx = test_acc_snr.argmax().item()  # 获取对应的 SNR 索引

    global best_acc
    if test_acc > best_acc:
        best_acc = test_acc

    path_test = './results/' + cfgs.method + '/' + cfgs.dataset + '/' + cfgs.params["loss"] + '/' + str(t) + 't'
    logger.info(f'test_loss : {test_loss:.4e} | '
                f'test_acc : {test_acc:.4f} |'
                f'best_f1 :{f1:.4f}'
                f'Highest Acc SNR: {highest_acc_snr_idx} | '
                f'Acc: {highest_acc_snr:.4f}'
                f'best_acc :{best_acc:.4f}'
                f'args_t :{t}', file=path_test)

if __name__ == '__main__':

    t=[7]
    best_acc=0
    for t1 in t:
        cfgs = get_cfgs_2016aCen()
        cfgs.seed=t1+cfgs.seed
        main(cfgs,t1)