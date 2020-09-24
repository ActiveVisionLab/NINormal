import argparse
import os
import sys
import logging
from pathlib import Path
import datetime
import shutil

import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from PcpKnnPatchesDataset import PcpKnnPatchesDataset
from utils.training_utils import set_randomness, save_checkpoint
import metrics
from model import NINormalNet


def gen_detail_name(args):
    num_nb = int(os.path.splitext(args.train_dataset_name)[0].split('_')[-1])
    outstr = 'lr_' + str(args.learning_rate) + \
             '_train_batch_' + str(args.batchsize_train) + \
             '_train_noise_' + args.train_noise_level + \
             '_eval_noise_' + args.eval_noise_level + \
             '_L2_reg_' + str(args.L2_reg) + \
             '_gpu' + str(args.gpu_id) + \
             '_randseed_' + str(args.randseed) + \
             '_num_nb_' + str(num_nb) + \
             '_' + str(args.alias) + \
             '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    return outstr


def comp_loss_terms(pred, gt):
    """
    :param pred:    (B, N, 3)
    :param gt:      (B, N, 3)
    :return:        (1, )
    """
    sin_loss = torch.mean(torch.norm(torch.cross(pred, gt, dim=2), dim=2) / (torch.norm(pred, dim=2) * torch.norm(gt, dim=2)))
    return sin_loss


def parse_args():
    parser = argparse.ArgumentParser('NINormalNetTrain')
    parser.add_argument('--batchsize_train', type=int, default=6,
                        help='set to 6 (if you have 3x1080Ti) to reproduce paper results, reduce if OOM')
    parser.add_argument('--batchsize_eval', type=int, default=6, help='default to 6, reduce if OOM')
    parser.add_argument('--epoch',  default=900, type=int)
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--milestones', default=[400, 800], type=int, nargs='+',
                        help='lr schedule milestones in training')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help="lr milestones gamma")
    parser.add_argument('--fastdebug', default=False, action='store_true', help="debug with very small portion data")
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu', default=True, type=bool)

    parser.add_argument('--datafolder', type=str, default='./dataset_dir/pcp_knn_patch_h5_files',
                        help='folder contains h5 dataset')
    parser.add_argument('--train_dataset_name', type=str, default='train_patchsize_2000_k_20.h5')
    parser.add_argument('--eval_dataset_name', type=str, default='eval_patchsize_2000_k_20.h5')
    parser.add_argument('--train_noise_level', type=str, default='none', choices=['none', '0.01', '0.05', '0.1', 'all'])
    parser.add_argument('--eval_noise_level', type=str, default='none', choices=['none', '0.01', '0.05', '0.1', 'all'])

    parser.add_argument('--L2_reg', type=float, default=1e-4, help='L2 regularisation for network weights')
    parser.add_argument('--model_name', default='ni_normal_net', help='for checkpoint saving')

    parser.add_argument('--truerand', default=False, type=bool, help="whether we want true randomness")
    parser.add_argument('--randseed', default=20, help="set random seed for np, python, and torch")

    parser.add_argument('--alias', type=str, default='', help="specify experiments")
    return parser.parse_args()


def train_one_epoch(train_dataloader, optimizer, model, args):
    model.train()

    train_loss_epoch = 0
    train_pgp003_epoch, train_pgp005_epoch, train_pgp010_epoch, train_pgp030_epoch, train_pgp060_epoch, train_pgp080_epoch, train_pgp090_epoch = 0, 0, 0, 0, 0, 0, 0

    for batch_id, data in enumerate(tqdm(train_dataloader, desc='train_batch')):
        knn_pts = data['knn_pt_list']  # (B, N, K, 3)
        knn_pts = knn_pts.transpose(1, 2).contiguous()  # (B, K, N, 3)
        knn_pts = knn_pts.to(device='cuda:' + str(args.gpu_id), non_blocking=True)  # (B, K, N, 3)

        gt_normals = data['gt_normals']  # (B, N, 3)
        gt_normals = gt_normals.to(device='cuda:'+str(args.gpu_id), non_blocking=True)  # (B, N, 3)

        pred_normals, weights = model(knn_pts)

        loss = comp_loss_terms(pred_normals, gt_normals)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        '''Error analysis'''
        pgp_ang_dic = metrics.comp_pgp_batch_unori(pred_normals, gt_normals)
        train_loss_epoch += loss.item()

        train_pgp003_epoch += pgp_ang_dic['pgp003'].item()
        train_pgp005_epoch += pgp_ang_dic['pgp005'].item()
        train_pgp010_epoch += pgp_ang_dic['pgp010'].item()
        train_pgp030_epoch += pgp_ang_dic['pgp030'].item()
        train_pgp060_epoch += pgp_ang_dic['pgp060'].item()
        train_pgp080_epoch += pgp_ang_dic['pgp080'].item()
        train_pgp090_epoch += pgp_ang_dic['pgp090'].item()

    train_loss_epoch /= len(train_dataloader)

    train_pgp003_epoch /= float(len(train_dataloader.dataset) * 2000)  # 2k is the number of points in each patch
    train_pgp005_epoch /= float(len(train_dataloader.dataset) * 2000)  # 2k is the number of points in each patch
    train_pgp010_epoch /= float(len(train_dataloader.dataset) * 2000)  # 2k is the number of points in each patch
    train_pgp030_epoch /= float(len(train_dataloader.dataset) * 2000)  # 2k is the number of points in each patch
    train_pgp060_epoch /= float(len(train_dataloader.dataset) * 2000)  # 2k is the number of points in each patch
    train_pgp080_epoch /= float(len(train_dataloader.dataset) * 2000)  # 2k is the number of points in each patch
    train_pgp090_epoch /= float(len(train_dataloader.dataset) * 2000)  # 2k is the number of points in each patch

    return {
        'loss': train_loss_epoch,
        'pgp003': train_pgp003_epoch,
        'pgp005': train_pgp005_epoch,
        'pgp010': train_pgp010_epoch,
        'pgp030': train_pgp030_epoch,
        'pgp060': train_pgp060_epoch,
        'pgp080': train_pgp080_epoch,
        'pgp090': train_pgp090_epoch,
    }


def eval_one_epoch(eval_dataloader, model, args):
    model.eval()
    eval_loss_epoch = 0
    eval_pgp003_epoch, eval_pgp005_epoch, eval_pgp010_epoch, eval_pgp030_epoch, eval_pgp060_epoch, eval_pgp080_epoch, eval_pgp090_epoch = 0, 0, 0, 0, 0, 0, 0

    for batch_id, data in enumerate(eval_dataloader):
        knn_pts = data['knn_pt_list']  # (B, N, K, 3)
        knn_pts = knn_pts.transpose(1, 2).contiguous()  # (B, K, N, 3)
        knn_pts = knn_pts.to(device='cuda:' + str(args.gpu_id), non_blocking=True)  # (B, K, N, 3)

        gt_normals = data['gt_normals']  # (B, N, 3)
        gt_normals = gt_normals.to(device='cuda:' + str(args.gpu_id), non_blocking=True)

        pred_normals, weights = model(knn_pts)

        loss = comp_loss_terms(pred_normals, gt_normals)

        '''Error analysis'''
        pgp_ang_dic = metrics.comp_pgp_batch_unori(pred_normals, gt_normals)
        eval_loss_epoch += loss.item()

        eval_pgp003_epoch += pgp_ang_dic['pgp003'].item()
        eval_pgp005_epoch += pgp_ang_dic['pgp005'].item()
        eval_pgp010_epoch += pgp_ang_dic['pgp010'].item()
        eval_pgp030_epoch += pgp_ang_dic['pgp030'].item()
        eval_pgp060_epoch += pgp_ang_dic['pgp060'].item()
        eval_pgp080_epoch += pgp_ang_dic['pgp080'].item()
        eval_pgp090_epoch += pgp_ang_dic['pgp090'].item()

    eval_loss_epoch /= len(eval_dataloader)

    eval_pgp003_epoch /= float(len(eval_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    eval_pgp005_epoch /= float(len(eval_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    eval_pgp010_epoch /= float(len(eval_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    eval_pgp030_epoch /= float(len(eval_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    eval_pgp060_epoch /= float(len(eval_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    eval_pgp080_epoch /= float(len(eval_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    eval_pgp090_epoch /= float(len(eval_dataloader.dataset) * 2000)  # 100k is the number of points in each object

    return {
        'loss': eval_loss_epoch,
        'pgp003': eval_pgp003_epoch,
        'pgp005': eval_pgp005_epoch,
        'pgp010': eval_pgp010_epoch,
        'pgp030': eval_pgp030_epoch,
        'pgp060': eval_pgp060_epoch,
        'pgp080': eval_pgp080_epoch,
        'pgp090': eval_pgp090_epoch,
    }


def main(args):
    '''Create the log dir for this run'''
    exp_root_dir = Path('./logs/')
    exp_root_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # copy train and model file to ensure reproducibility.
    shutil.copy('./model.py', experiment_dir)
    shutil.copy('./train.py', experiment_dir)

    '''Logger'''
    logger = logging.getLogger("NINormalNetTrain")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info(args)

    '''Summary Writer'''
    writer = SummaryWriter(log_dir=experiment_dir)

    '''Data Loading'''
    logger.info('Load dataset ...')
    train_dataset = PcpKnnPatchesDataset(datafolder=args.datafolder,
                                         dataset_type='train',
                                         dataset_name=args.train_dataset_name,
                                         fastdebug=args.fastdebug,
                                         noise_level=args.train_noise_level)

    eval_dataset = PcpKnnPatchesDataset(datafolder=args.datafolder,
                                        dataset_type='eval',
                                        dataset_name=args.eval_dataset_name,
                                        fastdebug=args.fastdebug,
                                        noise_level=args.eval_noise_level)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize_train,
                                  shuffle=True, num_workers=10, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batchsize_eval,
                                 shuffle=False, num_workers=10, pin_memory=True)

    '''Model Loading'''
    model = NINormalNet()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device='cuda:' + str(args.gpu_id))
    else:
        model = model.to(device='cuda:'+str(args.gpu_id))

    '''Set Optimiser'''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.L2_reg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)

    '''Training'''
    best_temp = 0.0
    best_eval_pgp010 = 0.0
    sys.stdout.flush()
    logger.info('Start training...')
    for epoch in tqdm(range(args.epoch), desc='epochs'):
        logger.info('Epoch (%d/%s):', epoch + 1, args.epoch)

        train_epoch_metric = train_one_epoch(train_dataloader, optimizer, model, args)
        with torch.no_grad():
            eval_epoch_metric = eval_one_epoch(eval_dataloader, model, args)
        scheduler.step()

        tqdm.write('pgp010: train: {0:.4f}, eval: {1:.4f}'.format(train_epoch_metric['pgp010'], eval_epoch_metric['pgp010']))
        logger.info('pgp010: train: {0:.4f}, eval: {1:.4f}'.format(train_epoch_metric['pgp010'], eval_epoch_metric['pgp010']))

        writer.add_scalar('train/loss', train_epoch_metric['loss'], epoch)
        writer.add_scalar('train/pgp003', train_epoch_metric['pgp003'], epoch)
        writer.add_scalar('train/pgp005', train_epoch_metric['pgp005'], epoch)
        writer.add_scalar('train/pgp010', train_epoch_metric['pgp010'], epoch)
        writer.add_scalar('train/pgp030', train_epoch_metric['pgp030'], epoch)
        writer.add_scalar('train/pgp060', train_epoch_metric['pgp060'], epoch)
        writer.add_scalar('train/pgp080', train_epoch_metric['pgp080'], epoch)
        writer.add_scalar('train/pgp090', train_epoch_metric['pgp090'], epoch)  # sanity check, should be always 1.0 in un-oriented normal estimation
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
        writer.add_scalar('temp', model.module.temp if args.multi_gpu else model.temp, epoch)

        writer.add_scalar('eval/loss', eval_epoch_metric['loss'], epoch)
        writer.add_scalar('eval/pgp003', eval_epoch_metric['pgp003'], epoch)
        writer.add_scalar('eval/pgp005', eval_epoch_metric['pgp005'], epoch)
        writer.add_scalar('eval/pgp010', eval_epoch_metric['pgp010'], epoch)
        writer.add_scalar('eval/pgp030', eval_epoch_metric['pgp030'], epoch)
        writer.add_scalar('eval/pgp060', eval_epoch_metric['pgp060'], epoch)
        writer.add_scalar('eval/pgp080', eval_epoch_metric['pgp080'], epoch)
        writer.add_scalar('eval/pgp090', eval_epoch_metric['pgp090'], epoch)  # sanity check, should be always 1.0 in un-oriented normal estimation

        if eval_epoch_metric['pgp010'] >= best_eval_pgp010:
            best_eval_pgp010 = eval_epoch_metric['pgp010']
            best_temp = model.module.temp if args.multi_gpu else model.temp

            logger.info('Saving model with the best pgp010: {0:.4%} at temp {1:.4f}'.format(best_eval_pgp010, best_temp))
            tqdm.write('Saving model with the best pgp010: {0:.4%} at temp {1:.4f}'.format(best_eval_pgp010, best_temp))
            save_checkpoint(epoch, train_epoch_metric['pgp010'], eval_epoch_metric['pgp010'], model,
                            optimizer, str(experiment_dir), args.model_name)

    print('Best eval pgp010: {0:.4%} at temp {1:.4f}'.format(best_eval_pgp010, best_temp))
    logger.info('Best eval pgp010: {0:.4%} at temp {1:.4f}'.format(best_eval_pgp010, best_temp))

    print('Final temp: {0:.4f}'.format(model.module.temp if args.multi_gpu else model.temp))
    logger.info('Final temp: {0:.4f}'.format(model.module.temp if args.multi_gpu else model.temp))

    return


if __name__ == '__main__':
    torch.backends.cudnn.enabled=False
    args = parse_args()
    set_randomness(args)
    main(args)

