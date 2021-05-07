import sys
import numpy as np
import pandas as pd
import torch
import random
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tt
from torch.utils.data import random_split
import os
import shutil
from torchvision.datasets import ImageFolder

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

def parse_disease(fname):
    parts = fname.split('_')
    return parts[0]

def open_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            return images
        
        self.images = {}
        self.class_names = ['normal', 'viral', 'covid']
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

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
            if file.endswith('png'):
                os.rename((root_dir + '/' + classes + '/' + file),(root_dir + '/' + classes + '/' + classes + "_" + file))

def prepare_covid_data(batch_size):

    # root_dir = 'data'
    data_dir = 'data/covid_dataset'
    # print(os.listdir(data_dir))

    # dup_dir = data_dir + '/covid_dataset'
    # if os.path.exists(dup_dir) and os.path.isdir(dup_dir):
    #     shutil.rmtree(dup_dir)

    rename_files(data_dir)

    train_dirs = {
    "normal" : "/media/subham/New Volume1/mywork/upwork/project3_260421/comp_final/data/covid_dataset/Normal",
    "viral" : "/media/subham/New Volume1/mywork/upwork/project3_260421/comp_final/data/covid_dataset/ViralPneumonia",
    "covid" : "/media/subham/New Volume1/mywork/upwork/project3_260421/comp_final/data/covid_dataset/COVID"
    }

    train_transform = tt.Compose([tt.Resize(size = (128,128)),
                                tt.RandomHorizontalFlip(),
                                tt.ToTensor(),
                                tt.Normalize(mean = [0.485,0.456,0.406], std = [0.229, 0.224, 0.225])
    ])


    test_transform = tt.Compose([tt.Resize(size = (128,128)),
                                tt.ToTensor(),
                                tt.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ChestXRayDataset(train_dirs, train_transform)

    # Delete the duplicate folder
    # stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5

    # mean_nums = [0.485, 0.456, 0.406]
    # std_nums = [0.229, 0.224, 0.225]

    # data_transforms = tt.Compose([tt.Resize((64,64)), #Resizes all images into same dimension
    #                                     tt.RandomRotation(10), # Rotates the images upto Max of 10 Degrees
    #                                     tt.RandomHorizontalFlip(p=0.4), #Performs Horizantal Flip over images 
    #                                     tt.ToTensor(), # Coverts into Tensors
    #                                     tt.Normalize(mean = mean_nums, std=std_nums)]) # Normalizes

    # dataset = CovidDataset(data_dir, transform=data_transforms)

    # print(len(dataset))

    random_seed = 43
    torch.manual_seed(random_seed)

    val_pct = 0.4
    val_size = int(val_pct * len(dataset))
    train_size = len(dataset) - val_size

    train_ds, valid_ds= random_split(dataset, [train_size, val_size])
    print(len(train_ds))
    print(len(valid_ds))

    num_classes =  len(dataset.class_names)
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

