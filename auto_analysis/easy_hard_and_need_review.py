import os
import torch
import numpy as np
import ipdb
from tqdm import tqdm
import sys
import json
sys.path.append('/workspace/projects/ml-data-parameters/')

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from dataset.hf_imagenet_dataset import HFImageNet


def render_easy_hard_and_need_review(image_list, border_color_list, img_size=256, border_size=10):
    row_images = []

    # Process each row
    for images, border_color in zip(image_list, border_color_list):

        if len(images) == 0:
            continue

        images = [image.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    for image in images]

        # Add border to the image
        images= [ImageOps.expand(image, border=border_size, fill=border_color) for image in images]

        # Calculate total width and max height
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create new image for the row
        row_image = Image.new('RGB', (total_width, max_height))

        # Paste images into the row image
        x_offset = 0
        for img in images:
            y_offset = (max_height - img.height) // 2  # Center align vertically
            row_image.paste(img, (x_offset, y_offset))
            x_offset += img.width

        row_images.append(row_image)

    # Create final image
    final_width = max(img.width for img in row_images)
    final_height = sum(img.height for img in row_images)
    final_image = Image.new('RGB', (final_width, final_height))

    # Paste row images into final image
    y_offset = 0
    for img in row_images:
        x_offset = (final_width - img.width) // 2  # Center align horizontally
        final_image.paste(img, (x_offset, y_offset))
        y_offset += img.height

    return final_image

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
    analysis_dir = './weights_CL/imagenet/with_inst_params_lr_0.8_wd_1e-8'
    #analysis_dir = './weights_CL/imagenet/with_inst_params_lr_0.8_wd_0_no_LR_decay_no_data_aug'
    folder_name = analysis_dir.split('/')[-1]
    save_dir = f'./auto_analysis/output/easy_hard_need_review/{folder_name}'

    if not os.path.exists(save_dir):
        print('Creating directory to save files..')
        os.mkdir(save_dir)
        
    total_epochs = 120
    num_easy_samples = 5  # Number of easy images we are going to show
    noisy_threshold = 2.0 # Images with a temperature greater than this will be classified as hard

    # To classify images as interesting we care about two operating points:
    # max_val: if the instance's temperature value went over a certain threshold during learning.
    # conv_threshold: if the instance's temperature value was less than a particular threshold to assume this instance was learnt at convergence.
    max_val_threshold_need_review = 1.5 # This was set to 2 for the baseline.
    conv_threshold = 1.0 

    ## Load  instance-level parameters from all epochs
    epoch_instance_level_temp = load_instance_level_params(total_epochs) # (nr_epochs, N)

    # Print some stats
    temp_convergence = epoch_instance_level_temp[-1,:] # (1, N)
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
    for class_idx in tqdm(range(0, 1000, 100)): # Analayze 1 out of 10 classes. 
    #for class_idx in [145]: # Analayze penguin class

        # Find temperatuere of instances of this class at convergence
        instance_class_idx = np.where(label_instances == class_idx)[0]       # (1, n)
        instance_class_temp_convergence = temp_convergence[instance_class_idx] # (1, n)

        # Print temperature stats
        print(f'Temperature stats for class: {id2label[class_idx]}')
        print(f'Mean temperature at convergence: {np.mean(instance_class_temp_convergence) :0.1f}')
        print(f'Max temperature at convergence: {np.max(instance_class_temp_convergence) :0.1f}')
        print(f'Min temperature at convergence: {np.min(instance_class_temp_convergence) :0.1f}')
        print('-------------------------')

        # Find the easy, hard and need-review instances
        ###############################
        easy_instances_idx, hard_instances_idx, need_review_instances_idx = [], [], []

        # Sort the data points based on temperature (low to high)
        sort_idx = np.argsort(instance_class_temp_convergence)
        instance_class_idx = instance_class_idx[sort_idx]
        instance_class_temp_convergence = instance_class_temp_convergence[sort_idx]

        # Get the easy instances
        easy_instances_idx = instance_class_idx[:num_easy_samples]

        # Get the hard instances
        instance_class_idx = instance_class_idx[::-1]
        instance_class_temp_convergence = instance_class_temp_convergence[::-1]

        for inst_idx, inst_temp in zip(instance_class_idx, instance_class_temp_convergence):
            if inst_temp > noisy_threshold:
                hard_instances_idx.append(inst_idx)

        # Get the instances which need review (instances which were forgotten at the start, but eventually learnt)
        for inst_idx in instance_class_idx:
            inst_temperature_trajectory = epoch_instance_level_temp[:, inst_idx] # (nr_epochs, 1)
            #inst_delta_high_low = np.max(temperature_trajectory) - np.min(temperature_trajectory)
            inst_temp_convergence = inst_temperature_trajectory[-1]

            if np.max(inst_temperature_trajectory) > max_val_threshold_need_review and inst_temp_convergence <= conv_threshold:
                need_review_instances_idx.append(inst_idx)


        ### Convert indices to images
        easy_images = [imagenet_dataset.dataset[int(idx)]['image'] for idx in easy_instances_idx]
        hard_images = [imagenet_dataset.dataset[int(idx)]['image'] for idx in hard_instances_idx]
        need_review_images = [imagenet_dataset.dataset[int(idx)]['image'] for idx in need_review_instances_idx]

        #### Plotting
        final_image = render_easy_hard_and_need_review(image_list=[easy_images, hard_images, need_review_images], 
                                                        border_color_list=['green', 'red', 'orange'],
                                                        border_size=5)       
        name_class = id2label[class_idx].split(',')[0] # Selecting the first name in a list of names seperated by comma
        final_image.save(f'{save_dir}/class_{class_idx}_{name_class}.png')