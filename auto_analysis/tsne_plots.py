import os
import torch
import numpy as np
import ipdb
from tqdm import tqdm
import sys
import json

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from huggingface_hub import hf_hub_download
from sklearn.manifold import TSNE

sys.path.append('/workspace/projects/ml-data-parameters/')
from dataset.hf_imagenet_dataset import HFImageNet


def plot_image(x, y, image, ax, border_color='white'):
    # Add a white border to the image
    border_size = 10  # Adjust border size as needed
    image = ImageOps.expand(image, border=border_size, fill=border_color)
    
    im = OffsetImage(image, zoom=0.2)  # Adjust zoom as needed
    ab = AnnotationBbox(im, (x, y), frameon=False, pad=0)
    ax.add_artist(ab)

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


def classify_image_based_on_data_params(instance_class_temp_all_epochs):
    '''
    Classify images as: easy, outlier, or interesting for active learning based on trajectory of instance level data-params.

    Arguments:
        instance_class_temp_all_epochs (nr_epochs, N): numpy array containing instance level data-parameters.
    '''

    # Classify each instance as easy vs hard based on temperature value and use it to color the border
    all_image_label = []

    instance_class_temp_convergence = instance_class_temp_all_epochs[-1, :]

    for idx in range(len(instance_class_temp_convergence)):
        if instance_class_temp_convergence[idx] > 1.6:
            all_image_label.append('red')
        else:
            all_image_label.append('green')

        # Check if this is an image which was ignored at the start but then learnt towards the end
        temperature_trajectory = instance_class_temp_all_epochs[:, idx]
        delta_high_low = np.max(temperature_trajectory) - np.min(temperature_trajectory)

        if np.max(temperature_trajectory) > 1.6 and delta_high_low > 0.5 and temperature_trajectory[-1] < 1.0:
            all_image_label[-1] = 'orange'

    outliers = (np.array(all_image_label) == 'red').sum()
    active_learning = (np.array(all_image_label) == 'orange').sum()
    print(f'{outliers} elements classifed as outliers')
    print(f'{active_learning} elements classifed as active-learning candidates')

    return all_image_label

if __name__ == '__main__':
    analysis_dir = './weights_CL_imagenet/with_inst_params_lr_0.8_wd_1e-8'
    folder_name = analysis_dir.split('/')[-1]
    save_dir = f'./auto_analysis/output/scatter_plot_tsne/{folder_name}'

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
        print(f'Analyzing class {class_idx}: {id2label[class_idx]}')

        save_path = f"{save_dir}/class_{class_idx}_{id2label[class_idx]}.pdf"
        if os.path.exists(save_path):
            print('Result already computed. Skipping ...')
            continue

        # Find instances of this class and their temperature value
        instance_class_idx = np.where(label_instances == class_idx)[0] # (N, 1)
        instance_class_temp_convergence = temp_convergence[instance_class_idx] # (N, 1)
        instance_class_temp_all_epochs = epoch_instance_level_temp[:, instance_class_idx] # (nr_epochs, N)

        # Compute T-SNE embedding using the temporal fluctation in data-parameters as feature
        features_instances = instance_class_temp_all_epochs.T #(N, D=nr_epochs)
        coordinates = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=50).fit_transform(features_instances) # (N, 2)

        # Classify each instance as easy vs hard based on temperature value and use it to color the border
        all_border_label = classify_image_based_on_data_params(instance_class_temp_all_epochs)

        # Read the list of images for all instances in this class
        list_all_instances_images = [imagenet_dataset.dataset[int(idx)]['image'] for idx in instance_class_idx]

        fig, ax = plt.subplots(figsize=(25*8, 10*8))
        subset_num = -1
        img_size = 512
        ax.scatter(coordinates[:subset_num, 0], coordinates[:subset_num, 1])  # Plotting the points (optional)

        # Plot each image at its coordinates
        for coord, image, border_label in tqdm(zip(coordinates[:subset_num, :], list_all_instances_images[:subset_num], all_border_label[:subset_num])):
            plot_image(coord[0], coord[1], image.resize((img_size, img_size)), ax, border_label)

        fig.patch.set_facecolor('black')
        ax.axis('off')
        plt.suptitle(f"Analysis for images of visual category: {id2label[class_idx]}", fontsize=16, fontweight='bold', color='white')
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
        plt.close(fig)