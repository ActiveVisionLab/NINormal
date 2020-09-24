import os
import sys
import argparse

import numpy as np
import sklearn.neighbors
import h5py

sys.path.append(os.path.join(sys.path[0], '..'))

from utils.pcp_name_filter import get_pt_clouds_path


def parse_args():
    parser = argparse.ArgumentParser('PCPNet Dataset KNN Preprocessing For Normal')
    parser.add_argument('--datafolder_read', type=str, default='./dataset_dir/PCPNet_official_dataset')
    parser.add_argument('--datafolder_write', type=str, default='./')
    parser.add_argument('--K', default=25, type=int)

    return parser.parse_args()


def norm_pts_to_unit_sphere(pts):
    """
    :param pts:     (N, 3)
    :return:
           pts:     (N, 3)
           radius:  scalar
    """
    pts_range = np.max(pts, axis=0) - np.min(pts, axis=0)
    max_range = np.max(pts_range)  # this is the max diameter
    radius = max_range / 2.0
    pts /= radius
    return pts, radius


def comp_knn_pts(pts, K):
    """
    :param pts:     (N, 3) np array
    :param K:       python int
    :return:        (N, K, 3) np array
    """
    N = pts.shape[0]

    tree = sklearn.neighbors.KDTree(pts[:N], leaf_size=50)

    knn_pts_list = np.zeros((N, K, 3), dtype=np.float32)
    _, idx = tree.query(pts, k=K) # idx is (N, K)

    for r in range(N):
        knn_pts_list[r] = pts[idx[r]]

    centroids = np.mean(knn_pts_list, axis=1, keepdims=True)  # (N, 1, 3)
    knn_pts_list = knn_pts_list - centroids  # (N, K, 3)

    return knn_pts_list  # (N, K, 3)


def store_obj_pt_cloud(h5f, pts, normals, knn_pt_list, global_patch_counter, patch_size):
    N = pts.shape[0]  # 100K points

    num_patches = int(N/patch_size)
    for i in range(num_patches):
        grp = h5f.create_group(str(global_patch_counter).zfill(10))
        grp.create_dataset('pts', data=pts[i*patch_size:(i+1)*patch_size])
        grp.create_dataset('normals', data=normals[i*patch_size:(i+1)*patch_size])
        grp.create_dataset('knn_pt_list', data=knn_pt_list[i*patch_size:(i+1)*patch_size])
        global_patch_counter += 1
    return global_patch_counter


if __name__ == '__main__':
    args = parse_args()

    dataset_types = ['train', 'test', 'eval']
    # noise_types = ['none', 'white', 'gradient', 'striped']
    noise_types = ['none', 'white']
    # noise_types = ['gradient', 'striped']
    noise_levels = [0.01, 0.05, 0.1]

    # # Use this setting to debug the preprocessing code faster.
    # dataset_types = ['train']
    # noise_types = ['none']

    # We have 2000 points and their knn neighbours in a h5 sub-group.
    patch_size = 2000

    for dataset_type in dataset_types:
        if dataset_type is 'train':
            obj_names = np.genfromtxt(os.path.join(args.datafolder_read, 'trainingset_no_noise.txt'), dtype='str')
        elif dataset_type is 'test':
            obj_names = np.genfromtxt(os.path.join(args.datafolder_read, 'testset_no_noise.txt'), dtype='str')
        elif dataset_type is 'eval':
            obj_names = np.genfromtxt(os.path.join(args.datafolder_read, 'validationset_no_noise.txt'), dtype='str')

        dataset_name = dataset_type + '_patchsize_' + str(patch_size) + '_k_' + str(args.K) + '.h5'
        if os.path.exists(os.path.join(args.datafolder_write, dataset_name)):
            print("have dataset file, exit to avoid overwriting.")
            exit()
        else:
            h5f = h5py.File(os.path.join(args.datafolder_write, dataset_name), 'w', libver='latest')

        global_patch_counter = 0

        for noise_type in noise_types:
            if noise_type == 'none':
                obj_paths = get_pt_clouds_path(args.datafolder_read, obj_names, noise_type=noise_type)

                obj_pts_files = [p + '.xyz' for p in obj_paths]
                obj_normals_files = [p + '.normals' for p in obj_paths]

                for i in range(len(obj_pts_files)):
                    obj_name = obj_names[i]
                    print(noise_type, obj_names[i], obj_pts_files[i])

                    # shift the centre of the point cloud to origin
                    pts = np.genfromtxt(obj_pts_files[i])
                    pts = pts - np.mean(pts, axis=0)
                    pts, _ = norm_pts_to_unit_sphere(pts)
                    knn_pts_list = comp_knn_pts(pts, args.K)
                    normals = np.genfromtxt(obj_normals_files[i])

                    global_patch_counter = store_obj_pt_cloud(h5f, pts, normals, knn_pts_list, global_patch_counter, patch_size)
                    print("global counter: ", global_patch_counter)

            if noise_type == 'white' or noise_type == 'brown':
                for noise_level in noise_levels:
                    obj_paths = get_pt_clouds_path(args.datafolder_read, obj_names, noise_type=noise_type, noise_intensity=noise_level)

                    obj_pts_files = [p + '.xyz' for p in obj_paths]
                    obj_normals_files = [p + '.normals' for p in obj_paths]

                    for i in range(len(obj_pts_files)):
                        obj_name = obj_names[i]
                        print(noise_type, noise_level, obj_names[i], obj_pts_files[i])

                        # shift the centre of the point cloud to origin
                        pts = np.genfromtxt(obj_pts_files[i])
                        pts = pts - np.mean(pts, axis=0)
                        pts, _ = norm_pts_to_unit_sphere(pts)
                        knn_pts_list = comp_knn_pts(pts, args.K)
                        normals = np.genfromtxt(obj_normals_files[i])

                        global_patch_counter = store_obj_pt_cloud(h5f, pts, normals, knn_pts_list, global_patch_counter, patch_size)
                        print("global counter: ", global_patch_counter)

            if noise_type == 'gradient' or noise_type == 'striped':
                obj_paths = get_pt_clouds_path(args.datafolder_read, obj_names, noise_type=noise_type)

                obj_pts_files = [p + '.xyz' for p in obj_paths]
                obj_normals_files = [p + '.normals' for p in obj_paths]

                # the analytic ones does not have gradient and striped version
                obj_pts_files = [f for f in obj_pts_files if 'analytic' not in f]
                obj_normals_files = [f for f in obj_normals_files if 'analytic' not in f]

                for i in range(len(obj_pts_files)):
                    obj_name = obj_names[i]
                    print(noise_type, obj_names[i], obj_pts_files[i])

                    # shift the centre of the point cloud to origin
                    pts = np.genfromtxt(obj_pts_files[i])
                    pts = pts - np.mean(pts, axis=0)
                    pts, _ = norm_pts_to_unit_sphere(pts)
                    knn_pts_list = comp_knn_pts(pts, args.K)
                    normals = np.genfromtxt(obj_normals_files[i])

                    global_patch_counter = store_obj_pt_cloud(h5f, pts, normals, knn_pts_list, global_patch_counter, patch_size)
