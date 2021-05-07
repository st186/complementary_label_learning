import sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import random_split
import os
import shutil

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

def parse_species(fname):
    parts = fname.split('_')
    return parts[0]

def open_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FlowersDataset(Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.files = []
        self.classes = [fname for fname in os.listdir(root_dir) if fname != 'flowers']
        for classes in self.classes:                         
            for file in os.listdir(root_dir + '/' + classes): 
                if file.endswith('jpg'):
                    self.files.append(file)
        self.transform = transform
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        species = parse_species(fname)
        fpath = os.path.join(self.root_dir, species, fname)
        img = self.transform(open_image(fpath))
        class_idx = self.classes.index(species)
        return img, class_idx

def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels

def class_prior(complementary_labels):
    return np.bincount(complementary_labels) / len(complementary_labels)

def rename_files(root_dir):
    classes = os.listdir(root_dir)
    for classes in classes:
        for file in os.listdir(root_dir + '/' + classes): 
            if file.endswith('jpg'):
                os.rename((root_dir + '/' + classes + '/' + file),(root_dir + '/' + classes + '/' + classes + "_" + file))

def prepare_flowers_data(batch_size):
    
    root_dir = 'data'
    data_dir = 'data/flowers'
    print(os.listdir(data_dir))

    # Delete the duplicate folder
    dup_dir = data_dir + '/flowers'
    if os.path.exists(dup_dir) and os.path.isdir(dup_dir):
        shutil.rmtree(dup_dir)

    # Look into the new data directory 
    print(os.listdir(data_dir))

    rename_files(data_dir)

    img_size = 64
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = T.Compose([T.Resize((img_size, img_size)),
                        T.RandomCrop(64, padding=4, padding_mode='reflect'),
                        T.RandomHorizontalFlip(),
                        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        T.ToTensor(),
                        T.Normalize(*stats,inplace=True)])
    dataset = FlowersDataset(data_dir, transform=transform)

    print(len(dataset))

    random_seed = 43
    torch.manual_seed(random_seed)

    val_pct = 0.1
    val_size = int(val_pct * len(dataset))
    train_size = len(dataset) - val_size

    train_ds, valid_ds= random_split(dataset, [train_size, val_size])
    print(len(train_ds))
    print(len(valid_ds))

    num_classes =  len(dataset.classes)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False)
    full_train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)

    return full_train_loader, train_loader, test_loader, train_ds, valid_ds, num_classes

def prepare_train_loaders(full_train_loader, batch_size, ordinary_train_dataset):

    for i, (data, labels) in enumerate(full_train_loader):
            K = torch.max(labels)+1 # K is number of classes, full_train_loader is full batch
    complementary_labels = generate_compl_labels(labels)
    ccp = class_prior(complementary_labels)
    complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).float())
    ordinary_train_loader = torch.utils.data.DataLoader(dataset=full_train_loader, batch_size=batch_size, shuffle=True)
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=batch_size, shuffle=True)
    return full_train_loader, complementary_train_loader, ccp

