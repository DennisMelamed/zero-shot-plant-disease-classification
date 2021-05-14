from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torchvision.transforms.functional as TF
import os.path as osp

class HealthyUnhealthyDataset(Dataset):
    
    
    dataset = None
    full_class_mapping = None
    healthy_images = None
    dims = None
    
    every_im_transform = transforms.Compose([
        transforms.Resize((224,224)),  # interpolation `BILINEAR` is applied by default
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomVerticalFlip(0.4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomPerspective(),
        transforms.RandomRotation(180),
        transforms.RandomCrop(224,224)
        
    ])
    def __init__(self, data_dir = "PlantVillage-Dataset", plant_types = 'all', transform = False, split = "train"):
        self.data_dir = data_dir
        self.split = split
        im_folders = glob.glob(data_dir + "/*/")
        self.im_folders = im_folders
        self.all_plants = sorted(list(set([a[:-1].split("/")[1].split("___")[0] for a in im_folders])))
        for plant in self.all_plants:
            if data_dir + "/" + plant + "___healthy/" not in im_folders:
                self.all_plants.remove(plant)
        if plant_types != 'all':
            self.all_plants = sorted(list(set(plant_types).intersection(set(self.all_plants))))
        print("plants in this dataset: {}".format(self.all_plants))
        self.weights = {a:{"healthy":0, "unhealthy":0} for a in self.all_plants}
        self.get_paths_and_full_classes()
        print("weights: {}".format(self.weights))
        print("\n")
        if not transform:
            self.transform = None
        
        
            
            
    def get_paths_and_full_classes(self):
        dataset = []
        healthy_images = {plant:[] for plant in self.all_plants}
        for plant in self.all_plants:
            
            diseases = [a.split("/")[-1] for a in glob.glob(osp.join(self.data_dir, "splits", plant, "*"))]
            for disease in diseases:
                with open(osp.join(self.data_dir, "splits", plant, disease, self.split + ".txt"), 'r') as f:
                    images = f.readlines()
                
                print("{}, {}: {}".format(plant, disease, len(images)) )
                self.weights[plant]["healthy" if disease=="healthy" else "unhealthy"] += len(images)
                
                for image in images:
                    dataset.append({"path":image.strip(), 
                                     "plant": plant,
                                     "disease": disease,
                                      "healthy": 1 if disease=="healthy" else 0
                                       })
                    if dataset[-1]["healthy"] == 1:
                        healthy_images[dataset[-1]["plant"]].append(dataset[-1]['path'])

        self.dataset = dataset
        self.healthy_images = healthy_images
        self.dims = self.pil_loader(dataset[-1]['path']).size
        
        weight_sum = np.sum([1/self.weights[a][b] for a in self.all_plants for b in ['healthy', 'unhealthy']])
        print("weight sum: {}".format(weight_sum))
        self.weights = {plant:{"healthy":1/(self.weights[plant]['healthy']*weight_sum), 
                               "unhealthy":1/(self.weights[plant]['unhealthy']*weight_sum) } for plant in self.all_plants}
    
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    @classmethod
    def pil_loader(cls, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    # TODO: specify the return type
    @classmethod
    def accimage_loader(cls, path: str) -> Any:
        import accimage
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return pil_loader(path)

    @classmethod
    def default_loader(cls, path: str) -> Any:
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return cls.accimage_loader(path)
        else:
            return cls.pil_loader(path)
    
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        plant_name = self.dataset[idx]['plant']
        
        healthy_image_fn = self.healthy_images[plant_name][np.random.randint(len(self.healthy_images[plant_name]))]
        unknown_image_fn = self.dataset[idx]['path']
        
        healthy_image = self.default_loader(healthy_image_fn)
        
        unknown_image = self.default_loader(unknown_image_fn)
        
    
        
        if self.transform is not None:
            healthy_image = self.transform(healthy_image)
            unknown_image = self.transform(unknown_image)
        
        healthy_tensor = self.every_im_transform(healthy_image)
        unknown_tensor = self.every_im_transform(unknown_image)
        
        label = self.dataset[idx]['healthy']
        
        #print(self.dataset[idx])
        
#        healthy_tensor = torch.from_numpy((np.array(healthy_image)/255)*2 - 1)
#        unknown_tensor = torch.from_numpy((np.array(unknown_image)/255)*2 - 1)
        
        weight = self.weights[plant_name]['healthy' if label==1 else 'unhealthy']
        
        return {"healthy image": healthy_tensor,
                "unknown image": unknown_tensor,
                "weight": weight,
                "label": float(label)}