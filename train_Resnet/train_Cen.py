import torch
from amr.dataloaders.dataloader import *
from amr.utils import *
from amr.utils.solver import *

from Resnet_config import *

from amr.utils.log_train_info import train_info

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

    # load model
    model = init_model(cfgs,cfgs.params["network"],path)
    model.to(device)

    criterion = init_loss(cfgs.params["loss"])

    # train model
    if cfgs.train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfgs.params["lr"]), weight_decay=cfgs.params["weight_decay"])
        trainer = Trainer(model=model, device=device, optimizer=optimizer, lr_decay=cfgs.params["lr_decay"], criterion=criterion,
                          save_path=path,
                          early_stop=cfgs.params["early_stop"])
        train_loss, train_acc, valid_loss, valid_acc = trainer.loop(cfgs.params["epochs"], train_loader, valid_loader)

        draw_train(train_loss, train_acc, valid_loss, valid_acc,
                   save_path=path + '/draws')

    cfgs.train = False
    model = init_model(cfgs, cfgs.params["network"],path)
    model.to(device)

    test_loss, test_acc, f1,test_conf, test_conf_snr, test_acc_snr = Tester(model=model, device=device,
                                                                         criterion=criterion,
                                                                         classes=len(cfgs.mod_type),
                                                                         snrs=snrs)(test_loader)

    draw_conf(test_conf, save_path=path + '/draws',
              labels=mods, order="total")

    for i in range(len(snrs)):
        logger.info(f'test_snr : {snrs[i]:.0f} | '
                    f'test_acc : {test_acc_snr[i]:.4f}')
        draw_conf(test_conf_snr[i],
                  save_path=path + '/draws',
                  labels=mods,
                  order=str(snrs[i]))

    draw_acc(snrs, test_acc_snr,
             save_path=path + '/draws')

    train_info(train_loss, train_acc, valid_loss, valid_acc, path)

    highest_acc_snr = test_acc_snr.max().item()
    highest_acc_snr_idx = test_acc_snr.argmax().item()

    global best_acc
    if test_acc > best_acc:
        best_acc = test_acc

    path_test = './results/' + cfgs.method + '/' + cfgs.dataset + '/' + cfgs.params["loss"]
    logger.info(f'test_loss : {test_loss:.4e} | '
                f'test_acc : {test_acc:.4f} |'
                f'best_f1 :{f1:.4f}'
                f'Highest Acc SNR: {highest_acc_snr_idx} | '
                f'Acc: {highest_acc_snr:.4f}'
                f'best_acc :{best_acc:.4f}'
                f'args_t :{t}', file=path_test)

if __name__ == '__main__':

    t = [0]
    best_acc=0
    for t1 in t:
        cfgs = get_cfgs_Cen()
        cfgs.seed=t1+cfgs.seed
        main(cfgs,t1)