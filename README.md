# Neighbourhood-Insensitive Point Cloud Normal Estimation Network

**[Project Page](http://ninormal.active.vision/) | 
[Paper](https://arxiv.org/abs/2008.09965) | 
[Video](https://youtu.be/gxBeR2LBB0k) |
[Supp](http://www.robots.ox.ac.uk/~ryan/bmvc2020/0028_supp.pdf) | 
[Data](https://huggingface.co/datasets/active-vision-lab/NINormal/tree/main) | 
[Pretrained Models](https://huggingface.co/datasets/active-vision-lab/NINormal/tree/main)**

Zirui Wang and [Victor Adrian Prisacariu](http://www.robots.ox.ac.uk/~victor/). Active Vision Lab, University of Oxford.
BMVC 2020 (Oral Presentation).

**Update 30 June 2025**: data and pretrained checkpoints are now available at HuggingFace ([link](https://huggingface.co/datasets/active-vision-lab/NINormal/tree/main)). 

~**Update**: We use our university's OneDrive to store our pretrained models and the preprocessed dataset. The university just changed the access policy and this stops us sharing our data through a public link so the data and pretrained links above are broken. The easiest fix for now is **you can send your email address to ryan[AT]robots.ox.ac.uk and I'll share it through email**. We will try to find out a way to share it with a link properly later.~

## Environment:
```
Python == 3.7
PyTorch >= 1.1.0
CUDA >= 9.0
h5py == 2.10
```

We tried PyTorch 1.1/1.3/1.4/1.5 and CUDA 9.0/9.2/10.0/10.1/10.2. So the code should be able to run as long as you have a modern PyTorch and CUDA installed.

Setup env:
```
conda create -n ninormal python=3.7
conda activate ninormal
conda install pytorch=1.1.0 torchvision cudatoolkit=9.0 -c pytorch  # you can change the pytorch and cuda version here.
conda install tqdm future
conda install -c anaconda scikit-learn
pip install h5py==2.10
pip install tensorflow  # this is the cpu version, we just need the tensorboard.
```

(Optional) Install `open3d` for visualisation. You might need a physical monitor to install this lib.
```
conda install -c open3d-admin open3d
```

Clone this repo:
```
git clone https://github.com/ActiveVisionLab/NINormal.git
```

## Dataset:
We use the PCPNet dataset in our paper. The official PCPNet dataset is available at [here](https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/). We pre-processed the official PCPNet dataset using scikit-learn's KDTree and wrapped the processed knn point patches in h5 files. For the best training efficiency, we produce an h5 file for each K and each train/test/eval split.

To simply reproduce our paper results with k=20 or k=50, we provide a subset of our full pre-processed dataset [here](https://unioxfordnexus-my.sharepoint.com/:u:/g/personal/lina3315_ox_ac_uk/Ech7GImZcnhLvawKcARkwCoBbMEa5_I6qLZRsQvkRYCztQ?e=T1sJP1).


To fully reproduce our results from k=3 to k=50 (paper Fig. 2), the full pre-processed dataset is available [here](https://unioxfordnexus-my.sharepoint.com/:u:/g/personal/lina3315_ox_ac_uk/EQzIvFRy1PNOnB_aFo6qLQYBT7cr7hygZsom2a87wfukuQ?e=kMj1lK).

## Training
Untar the dataset.
```
tar -xvf path/to/the/tar.gz
```

We train all our models (except k=40 and k=50) using 3 Nvidia 1080Ti GPUs. For k=40 and k=50, we use 3 Nvidia Titan-RTX GPUs. All models are trained with batch size 6. To reproduce our paper results, set the `--batchsize_train=6` and `--batchsize_eval=6`. Reduce the batch size when out of memory.

Train with 20 neighbours:
```
python train.py \
--datafolder='path/to/the/folder/contains/h5/files' \
--batchsize_train=6 \
--batchsize_eval=6
```

Train with 50 neighbours:
```
python train.py \
--datafolder='path/to/the/folder/contains/h5/files' \
--train_dataset_name='train_patchsize_2000_k_50.h5' \
--eval_dataset_name='eval_patchsize_2000_k_50.h5' \
--batchsize_train=6 \
--batchsize_eval=6
```

#### Optional: use a symlink
Alternatively, you can create a symlink that points to the downloaded dataset:
```
cd NINormal  # our repo
mkdir dataset_dir
cd dataset_dir
ln -s path/to/the/folder/contains/h5/files ./pcp_knn_patch_h5_files
```

and train with:
```
python train.py
```

## Note on Batch Size

The batch size 6 is the batch size that the `Conv2D()` function processes. Our network can be implemented using the `Conv1D()` or `Linear()` but we use the 1x1 `Conv2D()` along with our pre-processed dataset to achieve the best balance between data loading and training. When setting the batch size to 6, the actual batch size our network processes is 6 x 2000 = 12000, as mentioned at the end of Sec. 3 in our paper. The number 2000 is the number of knn patches we packed in a subgroup in an h5 file. See the pre-processing script in `./utils` and the `PcpKnnPatchesDataset` for more details.


## Pretrained Models
Similar to the dataset, we provide a tar file that contains models trained with k=20 and k=50 [here](https://unioxfordnexus-my.sharepoint.com/:u:/g/personal/lina3315_ox_ac_uk/ETepIC914XVPnAbUm1BESTABZb3pOOeOU2JYLlnAWjxeeg?e=1hsGOe).

To evaluate all models that we present in Fig.2 (k=3 to k=50), download all models [here](https://unioxfordnexus-my.sharepoint.com/:u:/g/personal/lina3315_ox_ac_uk/EZCD7wK19bVMvXK6NGxbPMoBlvaJ_GzOq1szOF4ay7PcDg?e=5Q7ngq).

## Testing
Untar the downloaded checkpoints file.
```
tar -xvf path/to/the/ckpts/tar.gz
```

**IMPORTANT NOTE**:
The k for trained checkpoints and the k for a dataset must match. E.g. Use the nb20 ckpt with the nb20 dataset:
```
python test.py \
--ckpt_path='/path/to/the/ckpts/nb_20' \
--test_dataset_name='test_patchsize_2000_k_20.h5'
```

#### Optional: use a symlink
Like the dataset, you can also do a symlink that points to the downloaded checkpoint folder:
```
cd NINormal  # our repo
ln -s path/to/the/folder/just/extracted ./paper_ckpts
```

and run test with just:
```
python test.py
```

## Attention Weights Visualisation in 3D (paper Fig. 5)
We recommend visualising attention weights using k=50 (with the model trained with k=50 of course...) to see how our network pays extra attention to the boundary of a patch.  

Install the opend3d lib. You might need a PC with a physical monitor to install this library...
```
conda install -c open3d-admin open3d
```

Similar to the testing procedure, after got datasets, run:
```
python test_vis_attn_map_3d.py --ckpt_path='/path/to/the/ckpts/nb_50'
```


## ICP Iteration Experiment (paper Sec. 4.5)
We aim to release it soon.

## Acknowledgement
The authors would like to thank 
[Min Chen](https://sites.google.com/site/drminchen/home), 
[Tengda Han](https://tengdahan.github.io/),
[Shuda Li](https://lishuda.wordpress.com/),
[Tim Yuqing Tang](https://scholar.google.co.uk/citations?user=kQB_dOoAAAAJ&hl=en) and 
[Shangzhe Wu](https://elliottwu.com/)
for insightful discussions and proofreading.

## Citation
```
@inproceedings{wang2020ninormal,
    title={Neighbourhood-Insensitive Point Cloud Normal Estimation Network},
    author={Wang, Zirui and Prisacariu, Victor Adrian},
    booktitle={BMVC},
    year={2020}
}
```
