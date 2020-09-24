import os
import glob
import sys

sys.path.append(os.path.join(sys.path[0], '..'))
# from utils import o3d_draw
import torch
import torch.utils.data
import numpy as np
import h5py
from tqdm import tqdm


class PcpKnnPatchesDataset(torch.utils.data.Dataset):
    def __init__(self, datafolder, dataset_name, dataset_type, fastdebug, noise_level):
        """
        :param datafolder:      path contains h5 file
        :param dataset_name:    xxx_patch_k_xx.h5
        :param dataset_type:    'train', 'test', 'eval'
        :param fastdebug:       True/False to load only a small portion for fast debugging
        :param noise_level:     'none', '0.01', '0.05', '0.1', 'all'

        The h5 file has many groups index with a 10 digit patch_id, each group (patch) has:
            - 'pts':            (N, 3)          np.float64
            - 'normals':        (N, 3)          np.float64
            - 'knn_pt_list':    (N, K, 3)       np.float32

        Group name should be object names like in '*_set.txt' in the official PCPNet dataset files.
        """
        self.datafolder = datafolder
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.fastdebug = fastdebug
        self.noise_level = noise_level

        if dataset_type == 'train':
            self.obj_names = np.genfromtxt(os.path.join(datafolder, 'trainingset_no_noise.txt'), dtype='str')
        elif dataset_type == 'test':
            self.obj_names = np.genfromtxt(os.path.join(datafolder, 'testset_no_noise.txt'), dtype='str')
        elif dataset_type == 'eval':
            self.obj_names = np.genfromtxt(os.path.join(datafolder, 'validationset_no_noise.txt'), dtype='str')

        # We are slight abusing the 'patch' word here, this patch only denotes 2k points and their neighbours in
        # pre-processing context, this is not the local knn patch that we use to estimate a normal.

        # 100k each, 2k pts per patch, we have 100k/2k = 50 patches per objects and we have 4 noise levels.
        if dataset_type == 'train':
            # 8 obj * 50 patches (* 4 levels)
            self.patches_per_noise_level = 400
        elif dataset_type == 'eval':
            # 3 obj * 50 patches (* 4 levels)
            self.patches_per_noise_level = 150
        elif dataset_type == 'test':
            # 19 obj * 50 patches (* 4 levels)
            self.patches_per_noise_level = 950

        self.h5f_path = os.path.join(self.datafolder, self.dataset_name)

        if dataset_type not in self.dataset_name:
            print('dataset name does not match dataset type. exit.')
            exit()

    def __len__(self):
        if self.fastdebug:
            return 16  # 16 patches
        else:
            return self.patches_per_noise_level if self.noise_level != 'all' else self.patches_per_noise_level*4

    def remap_index_for_noise_level(self, index, noise_level):
        if noise_level == 'none' or noise_level == 'all':
            return index
        elif noise_level == '0.01':
            return index + self.patches_per_noise_level
        elif noise_level == '0.05':
            return index + self.patches_per_noise_level*2
        elif noise_level == '0.1':
            return index + self.patches_per_noise_level*3

    def __getitem__(self, item):
        item = self.remap_index_for_noise_level(item, noise_level=self.noise_level)  # picking noise level in h5py file.
        with h5py.File(self.h5f_path, 'r', libver='latest') as h5f:
            patch = h5f[str(item).zfill(10)]
            pts = np.array(patch['pts'], dtype=np.float32)
            gt_normals = np.array(patch['normals'], dtype=np.float32)
            knn_pt_list = np.array(patch['knn_pt_list'], dtype=np.float32)

            # Slicing first few neighbours and shift them to the origin.
            # This is useful when use a dataset has more neighbours.
            # knn_pt_list = knn_pt_list[:, :25]  # (N, 25, 3)
            # knn_pt_list = knn_pt_list - np.mean(knn_pt_list, axis=1, keepdims=True)

            data_input = {
                'pts': pts,
                'knn_pt_list': knn_pt_list,
                'gt_normals': gt_normals
            }

        return data_input


if __name__ == '__main__':
    dataset = PcpKnnPatchesDataset(datafolder='./dataset_dir/pcp_knn_patch_h5',
                                   dataset_type='train',
                                   dataset_name='train_patchsize_2000_k_25.h5',
                                   fastdebug=False,
                                   noise_level='none')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


    for batch_id, data in enumerate(dataloader):
        print(batch_id)
        # if batch_id % 50 == 0:
        #     pts = data['pts']
        #     gt_normals = data['gt_normals']
        #     o3d_draw.draw_object_and_normal(pts.squeeze(), gt_normals.squeeze())
        knn_pt_list = data['knn_pt_list']
        print(knn_pt_list)
        exit()