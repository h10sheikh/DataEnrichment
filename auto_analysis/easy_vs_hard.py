import os
import torch
import numpy as np
import ipdb
from tqdm import tqdm
import sys
import json
sys.path.append('/workspace/projects/ml-data-parameters/')

from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from dataset.hf_imagenet_dataset import HFImageNet

def plot_images(images, row_label, start_subplot, ax_line=False):
    for idx, (img, temp) in enumerate(images):
        idx = idx + 1 # subplot starts the idx from 1
        ax = plt.subplot(2, len(images)+1, idx + 1 + start_subplot)
        #img = img.resize((256, 256), Image.Resampling.LANCZOS)
        ax.imshow(img)
        ax.set_title(f'{temp:.2f}', fontsize=16, color='white')
        ax.axis('off')  # Hide axis
            
    plt.figtext(0.01, 0.75 if start_subplot == 0 else 0.25, row_label, va='center', ha='left', fontsize=12, weight='bold', color='white')

def load_instance_level_params(total_epochs):
    # Initialize a dictionary to hold the loaded checkpoints
    epoch_instance_level_temp = [] # (nr_epochs, nr_instances)
    
    # Loop through the checkpoint files and load each one
    for epoch in tqdm(range(total_epochs), 
                        desc='Loading instance level parameters from all checkpoints'):
        checkpoint_path = f'{analysis_dir}/epoch_{epoch}.pth.tar'

        # Check if the file exists
        if os.path.isfile(checkpoint_path):
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)
            epoch_instance_level_temp.append(checkpoint['inst_parameters'])
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")

    # Convert the entries to temperature and vertically stack across epochs
    epoch_instance_level_temp = np.exp(np.vstack(epoch_instance_level_temp))

    return epoch_instance_level_temp


if __name__ == '__main__':
    #analysis_dir = './weights_CL/imagenet/with_inst_params_lr_0.8_wd_1e-8'
    analysis_dir = './weights_CL/imagenet/with_inst_params_lr_0.8_wd_0_no_LR_decay_no_data_aug'
    folder_name = analysis_dir.split('/')[-1]
    save_dir = f'./auto_analysis/output/easy_vs_hard/{folder_name}'

    if not os.path.exists(save_dir):
        print('Creating directory to save files..')
        os.mkdir(save_dir)
        
    total_epochs = 120
    topk = 10 # Number of top-k easy and hard images we want to plot

    ## Load  instance-level parameters from all epochs
    epoch_instance_level_temp = load_instance_level_params(total_epochs)

    # Print some stats
    temp_convergence = epoch_instance_level_temp[-1,:]
    print(f'Mean temperature at convergence: {np.mean(temp_convergence) :0.1f}')
    print(f'Max temperature at convergence: {np.max(temp_convergence) :0.1f}')
    print(f'Min temperature at convergence: {np.min(temp_convergence) :0.1f}')

    ## Load labels for entire dataset 
    imagenet_dataset = HFImageNet(split='train', transform=None)
    label_instances = np.array(imagenet_dataset.dataset['label'])

    # Mapping of label-ids to name of classes
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k):v for k,v in id2label.items()}


    ## Analyze samples of particular class
    for class_idx in tqdm(range(0, 1000, 10)): # Analayze 1 out of 10 classes. 

        # Find temperatuere of instances of this class at convergence
        instance_class_idx = np.where(label_instances == class_idx)[0]
        instance_class_temp_convergence = temp_convergence[instance_class_idx]

        # Print temperature stats
        print(f'Temperature stats for class: {id2label[class_idx]}')
        print(f'Mean temperature at convergence: {np.mean(instance_class_temp_convergence) :0.1f}')
        print(f'Max temperature at convergence: {np.max(instance_class_temp_convergence) :0.1f}')
        print(f'Min temperature at convergence: {np.min(instance_class_temp_convergence) :0.1f}')
        print('-------------------------')

        # Sort the data points based on temperature (low to high)
        sort_idx = np.argsort(instance_class_temp_convergence)
        instance_class_idx = instance_class_idx[sort_idx]
        instance_class_temp_convergence = instance_class_temp_convergence[sort_idx]

        easy_images, hard_images = [], []

        for print_idx, (instance_idx, instance_temp) in enumerate(zip(instance_class_idx, instance_class_temp_convergence)):
            
            # Top-k easiest samples
            if (print_idx < topk):
                easy_images.append((imagenet_dataset.dataset[int(instance_idx)]['image'], instance_temp))

            # Top-k hardest samples
            if (print_idx >= len(instance_class_idx) - topk):
                hard_images.append((imagenet_dataset.dataset[int(instance_idx)]['image'], instance_temp))        

        # Plot easy_images and hard_images
        fig= plt.figure(figsize=(20, 8), facecolor='black')  # Adjust the figure size as needed
        plot_images(easy_images, "Easy Images\n(score < 1.0)", 0, ax_line=True)
        plot_images(hard_images, "Hard Images\n(score > 1.0)", len(easy_images)+1)
        plt.suptitle(f"Analysis for images of visual category: {id2label[class_idx]}", fontsize=16, fontweight='bold', color='white')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/class_{class_idx}_{id2label[class_idx]}.pdf")
        plt.close(fig)