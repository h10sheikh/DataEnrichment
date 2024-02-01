import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import ipdb


class HFImageNet(Dataset):
    def __init__(self,
                 split,
                 transform=None,
                 target_transform=None,                    
                 dataset_subset_ckpt=None,
                 subset_level=None,
                 subset_frac=-1
                ):

        # Load ImageNet Dataset
        self.dataset = load_dataset('imagenet-1k', 
                                cache_dir='/workspace/hf_cache/', 
                                num_proc=32,
                                split=split) 

        self.transform = transform
        self.target_transform = target_transform

        # Generate dataset subset
        if dataset_subset_ckpt:
            self.subset_dict = {'dataset_subset_ckpt': dataset_subset_ckpt, 'subset_level': subset_level, 'subset_frac': subset_frac}
            self.filtered_indices = self.get_data_subset_mask()
            self.dataset = self.dataset.select(self.filtered_indices)
            print(f'Created a subset of dataset by selecting {subset_frac} fraction of images per class')

        print(f'Loaded {self.dataset.num_rows} files for {split} split of ImageNet')


    def get_data_subset_mask(self):
        '''Generate data subset mask using instance parameters.'''

        assert 0.0 < self.subset_dict['subset_frac'] < 1.0, 'condition not satisfied 0 < subset_frac < 1'
        print('We are going to load a subset of the dataset for training')
        print(f"Going to generate the subset mask from checkpoint present at: {self.subset_dict['dataset_subset_ckpt']}")
        _data = torch.load(self.subset_dict['dataset_subset_ckpt'], map_location='cpu') 
        inst_parameters = np.exp(_data['inst_parameters'])
        labels = self.dataset['label']

        print(f'Mean temperature of instances: {inst_parameters.mean() :0.2f}')

        # Filter dataset
        ##################
        # Original indices
        indices = np.arange(len(inst_parameters))

        if self.subset_dict['subset_level'] == 'dataset':
            print('Filtering samples across the entire dataset in one go')

            drop_index = int(len(inst_parameters) * (1-self.subset_dict['subset_frac']))
            # Sorting by inst_parameters in descending order
            filtered_indices = np.argsort(inst_parameters)[::-1][drop_index:] 

        elif self.subset_dict['subset_level'] == 'class':
            print(f'Filtering samples of each class independently')

            # Pairing the inst_parameters, labels, and original indices
            data = np.column_stack((inst_parameters, labels, indices))

            unique_classes = np.unique(data[:, 1])
            filtered_indices = []

            for cls_label in tqdm(unique_classes, total=len(unique_classes), desc='Filtering dataset using instance-parameters'):
                class_data = data[data[:, 1] == cls_label]
                # Sorting by inst_parameters in descending order
                class_data_sorted = class_data[class_data[:, 0].argsort()[::-1]]
                # Calculating the index to slice off the top (1-subset_frac)%
                drop_index = int(len(class_data) * (1-self.subset_dict['subset_frac']))

                ## How many instances in the dropped data had temperature below 1 (valid training instances)? 
                #percentage_valid = 100*np.mean(class_data_sorted[:drop_index][:, 0] < 1.0)
                #num_valid = np.sum(class_data_sorted[:drop_index][:, 0] < 1.0)
                #print(f'{percentage_valid} percentage data points were valid data points')

                # Dropping the top corrput data points
                class_data_filtered = class_data_sorted[drop_index:]
                # Extracting indices
                filtered_indices.append(class_data_filtered[:, 2])

            filtered_indices = np.concatenate(filtered_indices).astype(np.int64)

        return filtered_indices


    def __len__(self):
        return len(self.dataset['label'])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, index) where index is the index of this sample in dataset.
        """
        image, label = self.dataset[index]['image'], self.dataset[index]['label']
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, index
