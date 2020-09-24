import argparse
import os
import sys

import open3d as o3d
import numpy as np
import sklearn.neighbors
import torch
import torch.nn.parallel
from tqdm import tqdm

from utils.pcp_name_filter import get_pt_clouds_path
from utils.training_utils import set_randomness, load_ckpt_to_net
from metrics import comp_rms_angle_batch
from model import NINormalNet


def norm_pts_to_unit_sphere(pts):
    """
    :param pts:     N, 3
    :return:
           pts:     N, 3
           radius:  scalar
    """
    pts_range = np.max(pts, axis=0) - np.min(pts, axis=0)
    max_range = np.max(pts_range)  # this is the max diameter
    radius = max_range / 2.0
    pts /= radius
    return pts, radius


def parse_args():
    parser = argparse.ArgumentParser('NINormalNetTestVis3DPatch')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu', default=True, type=bool)

    parser.add_argument('--datafolder', type=str, default='./dataset_dir/PCPNet_official_dataset',
                        help='folder contains h5 dataset')
    parser.add_argument('--num_neighbours', type=int, default=50)
    parser.add_argument('--datatype', type=str, default='test')

    parser.add_argument('--truerand', type=bool, default=False, help="whether we want true randomness")
    parser.add_argument('--randseed', default=20, help="set random seed for np, python, and torch")
    parser.add_argument('--fastdebug', default=False, action='store_true', help="debug with very small portion data")
    parser.add_argument('--ckpt_path', type=str, help="checkpoint_folder",
                        default='./paper_ckpts/nb50')
    return parser.parse_args()


def test_one_epoch(model):
    model.eval()

    K = args.num_neighbours
    noise_type = 'none'
    noise_intensity = '0.0'

    obj_names = None
    if args.datatype is 'train':
        obj_names = np.genfromtxt(os.path.join(args.datafolder, 'trainingset_no_noise.txt'), dtype='str')
    elif args.datatype is 'test':
        obj_names = np.genfromtxt(os.path.join(args.datafolder, 'testset_no_noise.txt'), dtype='str')
    elif args.datatype is 'eval':
        obj_names = np.genfromtxt(os.path.join(args.datafolder, 'validationset_no_noise.txt'), dtype='str')

    obj_paths = get_pt_clouds_path(args.datafolder, obj_names, noise_type=noise_type, noise_intensity=noise_intensity)
    obj_pts_files = [p + '.xyz' for p in obj_paths]
    obj_normals_files = [p + '.normals' for p in obj_paths]

    for i in range(len(obj_pts_files)):
        obj_name = obj_names[i]

        # Uncomment this to visualise a specific object.
        # if obj_name != 'netsuke100k':
        #     continue

        print(noise_type, obj_name, obj_pts_files[i])

        # shift the centre of the point cloud to origin, and normalise the entire point cloud to unit sphere
        pts = np.genfromtxt(obj_pts_files[i])
        pts = pts - np.mean(pts, axis=0)
        pts, _ = norm_pts_to_unit_sphere(pts)
        normals = np.genfromtxt(obj_normals_files[i])

        N = pts.shape[0]
        tree = sklearn.neighbors.KDTree(pts[:N], leaf_size=50)

        # this controls how many patches we would like to vis for each object.
        counter = 0
        max_count = 7

        for r in tqdm(range(N)):
            '''Estimate normal and attn weights for a patch'''
            # get neighbours and shift to the origin
            _, idx = tree.query(pts[r].reshape(1, 3), k=K)  # the first one is itself
            knn = pts[idx.squeeze()]
            normal_gt = normals[idx.squeeze()][0]  # (3, )
            centroid = np.mean(knn, axis=0)
            knn_centred = knn - centroid

            knn_model_input = torch.from_numpy(knn_centred)  # (K, 3)
            knn_model_input = knn_model_input.view(1, K, 1, 3)  # (B, K, N, 3)
            knn_model_input = knn_model_input.cuda().float()

            # pred_normals: (B, N, 3), weights: (N, K)
            pred_normals, weights = model(knn_model_input)  # (1, 1, 3), (1, 50)
            normal_gt = torch.from_numpy(normal_gt).float().view(1, 1, 3).cuda()
            rms_angle = comp_rms_angle_batch(pred_normals, normal_gt)  # (1, 1)

            tqdm.write('Angle err {0:.4f}'.format(rms_angle))

            '''This is vis the patch in the entire point cloud'''
            pcd_entire = o3d.geometry.PointCloud()
            color = np.zeros_like(pts)
            color[:, 1] = 0.3  # all points are green
            color[idx.squeeze(), 1] = 0
            color[idx.squeeze(), 0] = 1  # selected points are red
            color[r] = 0
            color[r, 2] = 1  # the point is blue

            pcd_entire.points = o3d.utility.Vector3dVector(pts)
            pcd_entire.colors = o3d.utility.Vector3dVector(color)
            pcd_entire.normals = o3d.utility.Vector3dVector(normals)

            o3d.visualization.draw_geometries([pcd_entire])

            '''This is just vis the patch'''
            pcd_knn = o3d.geometry.PointCloud()
            color = np.zeros_like(knn_centred)

            # we need to amplify all attention weights, otherwise the colour-coding looks dull.
            max_weight = weights.max().item()
            enlarge_ratio = 1.0 / max_weight
            color[:, 0] = weights.squeeze().cpu().numpy() * enlarge_ratio * 7  # selected points are red
            color[0] = 0
            color[0, 2] = 1  # the point is blue

            pcd_knn.points = o3d.utility.Vector3dVector(knn_centred)
            pcd_knn.colors = o3d.utility.Vector3dVector(color)

            o3d.visualization.draw_geometries([pcd_knn])

            if counter >= max_count:
                break
            counter += 1


def main(args):
    '''Model Loading'''
    ckpt_file = os.path.join(args.ckpt_path, 'ni_normal_net.pth')
    model = NINormalNet()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device='cuda:' + str(args.gpu_id))
    else:
        model = model.to(device='cuda:'+str(args.gpu_id))
    load_ckpt_to_net(ckpt_file, model)

    '''Testing'''
    test_one_epoch(model)


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)
