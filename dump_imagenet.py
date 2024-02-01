import os
import ipdb
from multiprocessing import Pool
import json
from tqdm import tqdm

from PIL import Image
import torch
import torch.multiprocessing as mp
from torchvision import transforms
from huggingface_hub import hf_hub_download

from dataset.hf_imagenet_dataset import HFImageNet

def process_subset(indices, dataset, root_dir, id2label):
    for index in tqdm(indices, desc=f'Process {mp.current_process().pid}', position=mp.current_process()._identity[0]-1):
        image, label, _ = dataset[index]
        image = transforms.ToPILImage()(image)
        name_class = id2label[label].split(',')[0] # Selecting the first name in a list of names seperated by comma
        label_dir = os.path.join(root_dir, name_class)
        os.makedirs(label_dir, exist_ok=True)

        image_path = os.path.join(label_dir, f'image_{index}.jpg')
        image.save(image_path)

def split_indices(dataset_size, num_processes):
    indices = range(dataset_size)
    return [indices[i::num_processes] for i in range(num_processes)]

def process_images(dataset, root_dir, num_processes, id2label):
    dataset_size = len(dataset)
    indices_sets = split_indices(dataset_size, num_processes)

    processes = []
    for indices in indices_sets:
        p = mp.Process(target=process_subset, args=(indices, dataset, root_dir, id2label))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]) 
    dataset = HFImageNet(split='train', transform=train_transform)

    # Mapping of label-ids to name of classes
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k):v for k,v in id2label.items()}

    root_dir = 'imagenet_cleanlab_upload'
    num_processes = 1  # Adjust this based on your CPU

    process_images(dataset, root_dir, num_processes, id2label)
    print('All images have been saved successfully.')
