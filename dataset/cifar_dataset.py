#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import os
import ipdb
from tqdm import tqdm
import copy
import random

from PIL import Image
import numpy as np
import torch
from torchvision.datasets import CIFAR100


class CIFAR100WithIdx(CIFAR100):
    """
    Extends CIFAR100 dataset to yield index of element in addition to image and target label.
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 rand_fraction=0.0,
                 dataset_subset_ckpt=None,
                 subset_frac=-1
                 ):
        super(CIFAR100WithIdx, self).__init__(root=root,
                                              train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download)

        assert (rand_fraction <= 1.0) and (rand_fraction >= 0.0)
        self.rand_fraction = rand_fraction
        # We will make a copy of the labels since, we might augment them in corruption
        self.gt_targets = copy.deepcopy(self.targets) 

        if self.rand_fraction > 0.0:
            self.corrupt_fraction_of_data()
            assert (np.array(self.gt_targets) != np.array(self.targets)).mean() == self.rand_fraction, \
                'We did not corrupt correct fraction of data points'

        # Generate dataset subset
        if dataset_subset_ckpt:
            self.subset_dict = {'dataset_subset_ckpt': dataset_subset_ckpt, 'subset_frac': subset_frac}
            self.filtered_indices = self.get_data_subset_mask()

            # Compute accuracy for detection of noisy samples
            idx_corrupt_samples = len(self.data)*self.rand_fraction # Samples beyond this were clean
            accuracy_filtering = 100*(self.filtered_indices > idx_corrupt_samples).mean()
            print(f'The subset post filtering has {accuracy_filtering:0.1f}% clean data.')

            # Replace data and targets by filtered set
            self.data = self.data[self.filtered_indices, :]
            self.targets = np.array(self.targets)[self.filtered_indices].tolist() 

            print(f'Created a subset of dataset by selecting {subset_frac} fraction of images per class')

        split = 'train' if train else 'test'
        print(f'Loaded {self.data.shape[0]} files for {split} split of CIFAR100')

        #self.generate_data_dump_for_cleanlab()

    def generate_data_dump_for_cleanlab(self):
        ''' This function will the entire dataset as a dump for cleanlab upload '''

        nr_instances = len(self.data)
        save_dir = '/workspace/copy_data_scp/cleanlab_upload/CIFAR100_40pct_noisy_labels'

        for idx in tqdm(range(nr_instances), total=nr_instances, desc='saving data to disk'):
            img = Image.fromarray(self.data[idx])
            gt_target = self.gt_targets[idx]
            target = self.targets[idx]
            if not os.path.exists(f'{save_dir}/label_{target}/'):
                os.mkdir(f'{save_dir}/label_{target}/')

            save_path = f'{save_dir}/label_{target}/idx_{idx}_gt_label_{gt_target}.png'
            img.save(save_path)

    def get_data_subset_mask(self):
        '''Generate data subset mask using instance parameters.'''

        assert 0.0 < self.subset_dict['subset_frac'] < 1.0, 'condition not satisfied 0 < subset_frac < 1'
        print('We are going to load a subset of the dataset for training')
        print(f"Going to generate the subset mask from checkpoint present at: {self.subset_dict['dataset_subset_ckpt']}")
        _data = torch.load(self.subset_dict['dataset_subset_ckpt'], map_location='cpu') 
        inst_parameters = np.exp(_data['inst_parameters'])

        print(f'Mean temperature of instances: {inst_parameters.mean() :0.2f}')

        # Original indices
        indices = np.arange(len(inst_parameters))

        print('Filtering samples across the entire dataset in one go')
        drop_index = int(len(inst_parameters) * (1-self.subset_dict['subset_frac']))
        # Sorting by inst_parameters in descending order
        filtered_indices = np.argsort(inst_parameters)[::-1][drop_index:] 

        return filtered_indices

    def corrupt_fraction_of_data(self):
        """Corrupts fraction of train data by permuting image-label pairs."""

        # Check if we are not corrupting test data
        assert self.train is True, 'We should not corrupt test data.'

        nr_points = len(self.data)
        nr_corrupt_instances = int(np.floor(nr_points * self.rand_fraction))
        print('Randomizing {} fraction of data == {} / {}'.format(self.rand_fraction,
                                                                  nr_corrupt_instances,
                                                                  nr_points))
        # We will augment the labels for the top-frac of dataset.
        # We will add a random offset to labels
        ##########################################################3
        max_val_target = max(self.targets)
        np.random.seed(0) # Make this determinstic
        random.seed(0)
        for idx in range(nr_corrupt_instances):
            offset = random.randint(1, max_val_target-1) 
            new_target = (self.targets[idx]+ offset)%max_val_target
            assert new_target != self.targets[idx], 'New assigned target can not be the same as original'
            self.targets[idx] = new_target

    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (sample, target, index) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return img, target, index



