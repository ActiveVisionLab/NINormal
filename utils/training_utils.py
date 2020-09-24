import datetime

import numpy as np
import random
import torch


def set_randomness(args):
    if args.truerand is False:
        random.seed(args.randseed)
        np.random.seed(args.randseed)
        torch.manual_seed(args.randseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(epoch, train_loss, test_loss, model, optimizer, path, modelnet='checkpoint'):
    savepath = path + '/%s.pth' % modelnet
    state = {
        'epoch': epoch,
        'train_loss': train_loss,
        'eval_loss': test_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)


def load_ckpt_to_net(ckpt_path, net):
    # ckpt = torch.load(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    weights = ckpt['model_state_dict']
    net.load_state_dict(weights)
    return net
