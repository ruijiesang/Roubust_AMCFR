import torch
import numpy as np
import random
import os
from amr.utils import logger
from amr.models import *
import importlib


__all__ = ["init_device", "init_model", "init_loss","init_loss_FR"]


def init_device(seed=None, cpu=None, gpu=None):
    if seed is not None:
        random.seed(1)
        torch.manual_seed(seed)
        np.random.seed(1)
        torch.backends.cudnn.deterministic = True

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(1)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args, network,path):
    model = getattr(importlib.import_module("amr.models.networks." + args.method + '.' + network), network)(
        len(args.mod_type))
    print(model)
    getattr(importlib.import_module("amr.models.networks." + args.method + '.' + network), "test")()
    if not args.train:
        pretrained = path + '/checkpoints/best_acc.pth'
        print(pretrained)
        assert os.path.isfile(pretrained)
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(pretrained))

    return model

def init_loss(loss_func):
    loss = getattr(importlib.import_module("amr.models.losses." + loss_func), loss_func)()
    return loss



def init_loss_FR(loss_func,c,t,alp):
    loss = getattr(importlib.import_module("amr.models.losses." + loss_func), loss_func)(c,t,alp)
    return loss


if __name__ == '__main__':
    getattr(importlib.import_module("amr.models.networks.ResNet.ResNet"), "test")()
