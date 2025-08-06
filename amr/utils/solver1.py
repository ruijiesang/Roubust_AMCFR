from amr.utils import logger
import torch
import time

from sklearn.metrics import recall_score, f1_score
from amr.utils.logger import save_output_to_hdf5, save_data_noshink, save_output_to_hdf5_TSNE
from amr.utils.static import *
from amr.dataloaders.preprocess import *
import os
import numpy as np
__all__ = ["Trainer1", "Tester1","Tester_1"]

#保存HDF5信息
class Trainer1:
    def __init__(self, model, device, optimizer, lr_decay, criterion, save_path,valid_freq=1, early_stop=True):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_decay,
                                                                    patience=5, verbose=True, threshold=0.0001,
                                                                    threshold_mode='rel', cooldown=0, min_lr=1e-6,
                                                                    eps=1e-08)
        self.criterion = criterion
        self.all_epoch = None
        self.cur_epoch = 1
        self.train_loss = None
        self.train_acc = None
        self.valid_loss = None
        self.valid_acc = None
        self.train_loss_all = []
        self.train_acc_all = []
        self.valid_loss_all = []
        self.valid_acc_all = []
        self.train_times = []  # 用于记录每轮训练的用时
        self.valid_freq = valid_freq
        self.save_path = save_path
        self.best_acc = None
        # early stopping
        self.early_stop = early_stop
        self.patience = 10
        self.delta = 0
        self.counter = 0
        self.stop_flag = False

    def loop(self, epochs, train_loader, valid_loader, eps=1e-5):
        self.all_epoch = epochs
        for ep in range(self.cur_epoch, epochs+1):
            self.cur_epoch = ep

            shrink_save=False
            '''
            if ep>180:
                shrink_save=True
            '''
            start_time=time.time()
            self.train_loss, self.train_acc ,Y_soft_list,Y_list= self.train(train_loader,shrink_save)
            end_time=time.time()
            elapsed_time = end_time - start_time  # 计算每轮训练的耗时
            self.train_times.append(elapsed_time)  # 将耗时添加到列表中
            #save_output_to_hdf5([ep],Y_soft_list,Y_list,self.save_path)

            self.train_loss_all.append(self.train_loss)
            self.train_acc_all.append(self.train_acc)

            #保存验证信息
            self.valid_loss, self.valid_acc,test_pre,test_label = self.val(valid_loader)

            self.valid_loss_all.append(self.valid_loss)
            self.valid_acc_all.append(self.valid_acc)

            self._loop_postprocessing(self.valid_acc)
            if not self.lr_decay < eps:
                self.scheduler.step(self.valid_loss)

            if self.early_stop and self.stop_flag:
                logger.info(f'early stopping at Epoch: [{self.cur_epoch}]')
                break
        # 输出所有轮次的最小用时、最长用时和平均用时
        min_time = min(self.train_times)
        max_time = max(self.train_times)
        avg_time = sum(self.train_times) / len(self.train_times)

        logger.info(f"所有轮次的最小用时：{min_time:.2f} 秒")
        logger.info(f"所有轮次的最长用时：{max_time:.2f} 秒")
        logger.info(f"所有轮次的平均用时：{avg_time:.2f} 秒")
        return self.train_loss_all, self.train_acc_all, self.valid_loss_all, self.valid_acc_all

    def train(self, train_loader,shrink_save):
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader,shrink_save)

    def val(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader,False)


    def _iteration(self, data_loader,shrink_save):
        iter_loss = AverageMeter('Iter loss')
        iter_acc = AverageMeter('Iter acc')
        stime = time.time()
        #记录训练中的信息
        Y_soft_list=[]
        Y_list=[]

        for batch_idx, (X, Y, _) in enumerate(data_loader):
            X, Y = X.to(self.device).float(), Y.to(self.device)
            Y_soft = self.model(X)
            loss, Y_pred = self.criterion(Y_soft, Y,X)

            #保存信息
            Y_args=torch.softmax(Y_soft,dim=1)
            Y_soft_list.extend(Y_args)
            Y_list.extend(Y)

            if shrink_save:
                save_data_noshink(X,Y,Y_args,self.save_path)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            acc_pred = (Y_pred == Y).sum()
            acc_total = Y.numel()
            iter_acc.update(acc_pred/acc_total, acc_total)
            iter_loss.update(loss)

        ftime = time.time()
        if self.model.training:
            logger.info(f'Train | '
                        f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                        f'loss: {iter_loss.avg:.3e} | '
                        f'Acc: {iter_acc.avg:.3f} | '
                        f'time: {ftime-stime:.3f}')
        else:
            logger.info(f'Valid | '
                        f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                        f'loss: {iter_loss.avg:.3e} | '
                        f'Acc: {iter_acc.avg:.3f} | '
                        f'time: {ftime-stime:.3f}')

        return iter_loss.avg.item(), iter_acc.avg.item(),Y_soft_list,Y_list

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return
        path=self.save_path + '/checkpoints'
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, name))

    def _loop_postprocessing(self, acc):
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': self.best_acc
        }
        if self.best_acc is None:
            self.best_acc = acc
            state['best_acc'] = self.best_acc
            self._save(state, name=f"best_acc.pth")
        elif acc < self.best_acc + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.best_acc = acc
            state['best_acc'] = self.best_acc
            self._save(state, name=f"best_acc.pth")
            self.counter = 0


class Tester1:
    def __init__(self, model, device, criterion, classes, snrs):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.classes = classes
        self.snrs = snrs
        self.conf = torch.zeros(classes, classes)
        self.conf_snr = torch.zeros(len(snrs), classes, classes)
        self.acc_pred_snr = torch.zeros(len(snrs))
        self.acc_total_snr = torch.zeros(len(snrs))
        self.acc_snr = torch.zeros(len(snrs))

    def __call__(self, test_loader, verbose=True):
        self.model.eval()
        with torch.no_grad():
            loss, acc,f1,mean_final = self._iteration(test_loader)

        # 输出结果，包括Recall和F1-score
        if verbose:
            logger.info(f'Test | '
                        f'loss: {loss:.3e} | '
                        f'Acc: {acc:.3f} | '
                        f'mean_final: {mean_final:.3f} | '
                        f'F1-score: {f1:.3f}')
        return loss, acc,  f1, self.conf, self.conf_snr, self.acc_snr

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_acc = AverageMeter('Iter acc')
        stime = time.time()

        all_true_labels = []  # 保存所有真实标签
        all_pred_labels = []  # 保存所有预测标签
        all_Yfinal=[]
        all_result=[] #如果正确记录为1，如果错误记录为0

        for batch_idx, (X, Y, Z) in enumerate(data_loader):
            X, Y, Z = X.to(self.device).float(), Y.to(self.device), Z.to(self.device)
            Y_soft = self.model(X)
            loss, Y_pred,Y_final = self.criterion(Y_soft, Y, X)
            Y_pred1=Y_soft
            acc_pred = (Y_pred == Y).sum()
            result = (Y_pred == Y).to(torch.int64)
            acc_total = Y.numel()

            # 将真实标签和预测标签保存到列表中
            all_true_labels.extend(Y.cpu().numpy())
            all_pred_labels.extend(Y_pred1.cpu().numpy())
            all_Yfinal.append(Y_final.item())
            all_result.extend(result.cpu().numpy())

            for i in range(Y.shape[0]):
                self.conf[Y[i], Y_pred[i]] += 1
                idx = self.snrs.index(Z[i])
                self.conf_snr[idx, Y[i], Y_pred[i]] += 1
                self.acc_pred_snr[idx] += (Y[i] == Y_pred[i]).cpu()
                self.acc_total_snr[idx] += 1

            iter_acc.update(acc_pred / acc_total, acc_total)
            iter_loss.update(loss)

        # 计算每类的混淆矩阵
        for i in range(self.classes):
            self.conf[i, :] /= torch.sum(self.conf[i, :])
        for j in range(len(self.snrs)):
            self.acc_snr[j] = self.acc_pred_snr[j] / self.acc_total_snr[j]
            for i in range(self.classes):
                self.conf_snr[j, i, :] /= torch.sum(self.conf_snr[j, i, :])
        path='/home/sangruijie_qyh/Code/TransGroupNet-master/TransGroupNet-master/compar'
        save_output_to_hdf5(0,all_pred_labels,all_true_labels,path)
        # 计算总体 Recall 和 F1-score
        #f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
        f1=0
        mean_final=np.mean(all_Yfinal)
        ftime = time.time()
        logger.info(f'Test | '
                    f'loss: {iter_loss.avg:.3e} | '
                    f'Acc: {iter_acc.avg:.3f} | '
                    f'time: {ftime-stime:.3f}')
        return iter_loss.avg.item(), iter_acc.avg.item(),f1,mean_final


class Tester_1:
    def __init__(self, model, device, criterion, classes, snrs,save_path):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.classes = classes
        self.snrs = snrs
        self.conf = torch.zeros(classes, classes)
        self.conf_snr = torch.zeros(len(snrs), classes, classes)
        self.acc_pred_snr = torch.zeros(len(snrs))
        self.acc_total_snr = torch.zeros(len(snrs))
        self.acc_snr = torch.zeros(len(snrs))
        self.save_path=save_path

    def __call__(self, test_loader, verbose=True):
        self.model.eval()
        with torch.no_grad():
            loss, acc,mean_final = self._iteration(test_loader)



        # 混淆矩阵
        if verbose:
            logger.info(f'Test | '
                        f'loss: {loss:.3e} | '
                        f'Mean_final: {mean_final:.3e} | '
                        f'Acc: {acc:.3f}')
        return loss, acc, self.conf, self.conf_snr, self.acc_snr

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_acc = AverageMeter('Iter acc')
        stime = time.time()
        Y_allsoft=[]
        Y_label=[]
        Y_allfinal=[]

        for batch_idx, (X, Y, Z) in enumerate(data_loader):
            X, Y, Z = X.to(self.device).float(), Y.to(self.device), Z.to(self.device)
            Y_soft ,p1= self.model(X)
            loss, Y_pred,Y_final = self.criterion(Y_soft, Y, X)

            Y_soft1=torch.softmax(Y_soft,dim=1)
            Y_allsoft.extend(p1)
            Y_label.extend(Y)
            Y_allfinal.extend(Y_final)


            acc_pred = (Y_pred == Y).sum()
            acc_total = Y.numel()
            for i in range(Y.shape[0]):
                self.conf[Y[i], Y_pred[i]] += 1
                idx = self.snrs.index(Z[i])
                self.conf_snr[idx, Y[i], Y_pred[i]] += 1
                self.acc_pred_snr[idx] += (Y[i] == Y_pred[i]).cpu()
                self.acc_total_snr[idx] += 1

            iter_acc.update(acc_pred / acc_total, acc_total)
            iter_loss.update(loss)


        #save_data_to_pkl(X, Y, Z, Y_soft,self.save_path)
        save_output_to_hdf5_TSNE([0],Y_allsoft,Y_label,self.save_path)

        for i in range(self.classes):
            self.conf[i, :] /= torch.sum(self.conf[i, :])
        for j in range(len(self.snrs)):
            self.acc_snr[j] = self.acc_pred_snr[j] / self.acc_total_snr[j]
            for i in range(self.classes):
                self.conf_snr[j, i, :] /= torch.sum(self.conf_snr[j, i, :])

        mean_final=np.mean(Y_allfinal)
        ftime = time.time()
        logger.info(f'Test | '
                    f'loss: {iter_loss.avg:.3e} | '
                    f'Acc: {iter_acc.avg:.3f} | '
                    f'time: {ftime-stime:.3f}')
        return iter_loss.avg.item(), iter_acc.avg.item(),mean_final