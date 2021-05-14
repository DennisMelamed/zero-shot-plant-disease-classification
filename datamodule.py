import pytorch_lightning as pl
import torch
import random
import numpy as np
import glob
from torch.utils.data import random_split, DataLoader

from torchvision import transforms
from split_dataset import HealthyUnhealthyDataset

class ZeroShotDataModule(pl.LightningDataModule):

    def __init__(self, batch_size = 32, train_plants = 'all', test_plants = 'all', data_dir = "PlantVillage-Dataset"):
        super().__init__()
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self.batch_size = batch_size
        
        self.data_dir = data_dir
        im_folders = glob.glob(data_dir + "/*/")
        self.all_plants = sorted(list(set([a[:-1].split("/")[1].split("___")[0] for a in im_folders])))
        for plant in self.all_plants:
            if data_dir + "/" + plant + "___healthy/" not in im_folders:
                self.all_plants.remove(plant)
        if train_plants != 'all':
            self.train_plants = sorted(list(set(train_plants).intersection(set(self.all_plants))))
        else:
            self.train_plants = self.all_plants
        if test_plants != 'all':
            self.test_plants = sorted(list(set(test_plants).intersection(set(self.all_plants))))
        else:
            self.test_plants = self.all_plants


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = HealthyUnhealthyDataset(plant_types=self.train_plants, split='train', transform = True)
            self.val_data = HealthyUnhealthyDataset(plant_types=self.train_plants, split= 'val', transform = False)
            print("image dimensions: {}".format(self.val_data.dims))
            self.dims = self.train_data.dims

        if stage == 'test' or stage is None:
            ## TODO change to make a test dataset for each plant type, plus the first one contains all plant types
            self.test_data_full = HealthyUnhealthyDataset(plant_types=self.test_plants, split = "test", transform = False)
            self.test_data_per_plant = {plant_name:HealthyUnhealthyDataset(plant_types=[plant_name], split = "test", transform = False) for plant_name in self.test_plants}


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers = 24, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers = 24)

    def test_dataloader(self):
        # TODO return a dataloader for each test plant type
        return DataLoader(self.test_data_full, batch_size=self.batch_size, num_workers = 24)
    
    def test_indiv_plants_dataloaders(self):
        return {k:DataLoader(v, batch_size=self.batch_size, num_workers = 24) for k,v in self.test_data_per_plant.items()}