import argparse
import os
import sys
import math
from timeit import default_timer as timer
from pathlib import Path
import shutil
import logging

import numpy as np
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader

from PcpKnnPatchesDataset import PcpKnnPatchesDataset
from utils.training_utils import set_randomness, load_ckpt_to_net
import metrics
from model import NINormalNet


def parse_args():
    parser = argparse.ArgumentParser('NINormalNetTest')
    parser.add_argument('--batchsize_test', type=int, default=10,
                        help='batch size in testing, this number should be dividable by 50, e.g, 1, 2, 5, 10, 25')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu', default=True, type=bool)

    parser.add_argument('--datafolder', type=str, default='./dataset_dir/pcp_knn_patch_h5_files',
                        help='folder contains h5 dataset')
    parser.add_argument('--datatype', type=str, default='test')
    parser.add_argument('--test_dataset_name', type=str, default='test_patchsize_2000_k_20.h5')
    parser.add_argument('--test_noise_level', type=str, default='none', choices=['none', '0.01', '0.05', '0.1', 'all'])

    parser.add_argument('--truerand', type=bool, default=False, help="whether we want true randomness")
    parser.add_argument('--randseed', default=20, help="set random seed for np, python, and torch")
    parser.add_argument('--fastdebug', default=False, action='store_true', help="debug with very small portion data")
    parser.add_argument('--ckpt_path', type=str, help="checkpoint folder",
                        default='./paper_ckpts/nb20')

    parser.add_argument('--normal_out', type=bool, default=False, help="whether output estimated normals to files")
    return parser.parse_args()


def test_one_epoch(test_dataloader, model, normal_out_path):
    model.eval()
    test_pgp003_epoch, test_pgp005_epoch, test_pgp010_epoch, test_pgp030_epoch, test_pgp060_epoch, test_pgp080_epoch, test_pgp090_epoch = 0, 0, 0, 0, 0, 0, 0
    test_pgp003_obj, test_pgp005_obj, test_pgp010_obj, test_pgp030_obj, test_pgp060_obj, test_pgp080_obj, test_pgp090_obj = 0, 0, 0, 0, 0, 0, 0
    obj_names = test_dataloader.dataset.obj_names
    obj_count = 0
    rms_angle_list_obj = []
    rms_angle_list_patch = []

    patch_count = 0

    # accumulate normal predictions for one object and write it to a txt file.
    pred_normals_one_obj = []

    # profile inference time for each object, and compute an average time.
    object_time = 0
    time_list = []

    print('Obj, pgp003, pgp005, pgp010, pgp030, pgp060, pgp080, pgp090, rms_angle, time')
    for batch_id, data in enumerate(test_dataloader):
        gt_normals = data['gt_normals']  # (B, N, 3)
        knn_pts = data['knn_pt_list']  # (B, N, K, 3)

        knn_pts = knn_pts.transpose(1, 2).to(device='cuda:' + str(args.gpu_id))  # (B, K, N, 3)
        gt_normals = gt_normals.to(device='cuda:' + str(args.gpu_id))

        start = timer()
        pred_normals, weights = model(knn_pts)  # weights: (N, K)
        end = timer()
        duration = end - start
        object_time += duration

        # for txt normal file writing
        pred_normals_one_obj.append(pred_normals.reshape(-1, 3))

        pgp_ang_dic = metrics.comp_pgp_batch_unori(pred_normals, gt_normals)
        rms_angle = metrics.comp_rms_angle_batch(pred_normals, gt_normals)
        rms_angle_list_patch.append(rms_angle)

        test_pgp003_obj += pgp_ang_dic['pgp003'].item()
        test_pgp005_obj += pgp_ang_dic['pgp005'].item()
        test_pgp010_obj += pgp_ang_dic['pgp010'].item()
        test_pgp030_obj += pgp_ang_dic['pgp030'].item()
        test_pgp060_obj += pgp_ang_dic['pgp060'].item()
        test_pgp080_obj += pgp_ang_dic['pgp080'].item()
        test_pgp090_obj += pgp_ang_dic['pgp090'].item()

        patch_count += test_dataloader.batch_size
        if patch_count == 50:

            if args.normal_out:
                pred_normals_one_obj = torch.cat(pred_normals_one_obj, axis=0).cpu().numpy()
                np.savetxt(os.path.join(normal_out_path, obj_names[obj_count]+'.normals'), pred_normals_one_obj)
                pred_normals_one_obj = []  # clean it and prepare for the next object.

            rms_angle_obj = np.mean(rms_angle_list_patch)
            rms_angle_list_obj.append(rms_angle_obj)
            print('{0:25s}, {1:.2%}, {2:.2%}, {3:.2%}, {4:.2%}, {5:.2%}, {6:.2%}, {7:.2%}, {8:.2f}, {9:.2f}sec'.format(obj_names[obj_count],
                                                                                                                       test_pgp003_obj / float(100000),
                                                                                                                       test_pgp005_obj / float(100000),
                                                                                                                       test_pgp010_obj / float(100000),
                                                                                                                       test_pgp030_obj / float(100000),
                                                                                                                       test_pgp060_obj / float(100000),
                                                                                                                       test_pgp080_obj / float(100000),
                                                                                                                       test_pgp090_obj / float(100000),
                                                                                                                       rms_angle_obj,
                                                                                                                       object_time))
            test_pgp003_epoch += test_pgp003_obj
            test_pgp005_epoch += test_pgp005_obj
            test_pgp010_epoch += test_pgp010_obj
            test_pgp030_epoch += test_pgp030_obj
            test_pgp060_epoch += test_pgp060_obj
            test_pgp080_epoch += test_pgp080_obj
            test_pgp090_epoch += test_pgp090_obj

            test_pgp003_obj = 0
            test_pgp005_obj = 0
            test_pgp010_obj = 0
            test_pgp030_obj = 0
            test_pgp060_obj = 0
            test_pgp080_obj = 0
            test_pgp090_obj = 0

            obj_count += 1
            patch_count = 0
            rms_angle_list_patch = []

            time_list.append(object_time)
            object_time = 0.0

    test_pgp003_epoch /= float(len(test_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    test_pgp005_epoch /= float(len(test_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    test_pgp010_epoch /= float(len(test_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    test_pgp030_epoch /= float(len(test_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    test_pgp060_epoch /= float(len(test_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    test_pgp080_epoch /= float(len(test_dataloader.dataset) * 2000)  # 100k is the number of points in each object
    test_pgp090_epoch /= float(len(test_dataloader.dataset) * 2000)  # 100k is the number of points in each object

    # print("------------------------")
    print('Total, {0:.2%}, {1:.2%}, {2:.2%}, {3:.2%}, {4:.2%}, {5:.2%}, {6:.2%}'.format(test_pgp003_epoch,
                                                                                        test_pgp005_epoch,
                                                                                        test_pgp010_epoch,
                                                                                        test_pgp030_epoch,
                                                                                        test_pgp060_epoch,
                                                                                        test_pgp080_epoch,
                                                                                        test_pgp090_epoch))

    print('Mean shape RMS angle error: {0:.2f}'.format(np.mean(rms_angle_list_obj)))
    print('Mean time for all objects: {0:.2f}'.format(np.mean(time_list)))

    if obj_count != len(obj_names) or len(obj_names) != len(rms_angle_list_obj):
        print('Warning: number of object is not correct. Need to double check. Exit now.')
        exit()

def main(args):
    normal_out_path = None
    if args.normal_out:
        # Prepare the path to write normal predictions to txt files.
        # If the normal txt folder exists, we remove all '.normal' files and 'test.py' and 'testlog.txt' file inside it,
        # otherwise we create the folder straightway.
        # Finally we copy this test file to the folder for record.
        normal_out_path = Path(os.path.join(args.ckpt_path, 'normal_txts'))
        if os.path.exists(normal_out_path):
            ans = input('Already have normal estimation txt folder, OVERWRITE? (yes/n) ')
            if ans == 'yes':
                filelist = [f for f in os.listdir(normal_out_path) if f.endswith('.normals') or f =='test.py' or f == 'testlog.txt']
                for f in filelist:
                    print('removed file: ', os.path.join(normal_out_path, f))
                    os.remove(os.path.join(normal_out_path, f))
            else:
                print("have normal estimation txt folder, exit to prevent overwrite.")
                exit()
        else:
            normal_out_path.mkdir(parents=False, exist_ok=False)
        shutil.copy('./test.py', normal_out_path)

        '''LOG'''
        logger = logging.getLogger("NINormalNetTest")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(normal_out_path, 'testlog.txt'))
        logger.addHandler(file_handler)
        logger.info(args)

    '''Data Loading'''
    test_dataset = PcpKnnPatchesDataset(datafolder=args.datafolder,
                                        dataset_type=args.datatype,
                                        dataset_name=args.test_dataset_name,
                                        fastdebug=args.fastdebug,
                                        noise_level=args.test_noise_level)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize_test, shuffle=False, num_workers=8)

    '''Model Loading'''
    ckpt_file = os.path.join(args.ckpt_path, 'ni_normal_net.pth')
    model = NINormalNet()

    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device='cuda:' + str(args.gpu_id))
    else:
        model = model.to(device='cuda:'+str(args.gpu_id))

    load_ckpt_to_net(ckpt_file, model)

    '''Testing'''
    test_one_epoch(test_dataloader, model, normal_out_path)


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)
