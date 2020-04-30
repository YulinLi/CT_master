#!/usr/bin/env python
# coding: utf-8

# In[149]:


from skimage import transform
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import glob
import torch
import pickle
import pydicom
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import csv
import pathlib
import torch
import torchvision
import torchvision.transforms as transforms


# In[160]:


class CBISDDSMDataset(Dataset):
    def __init__(self, csv_path, data_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(CBISDDSMDataset, self).__init__()
        with open(csv_path, newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            img_names = []
            pathology = []
            for row in rows:
                #ds = pydicom.filereader.dcmread(os.path.join(image_base_path,row['image file path']))
                dp = row['cropped image file path']
                dp = dp.split("/")
                bm = row['pathology']
                bm = bm.split("_")
                if(bm[0] == "BENIGN"):
                    annotations = 0
                elif(bm[0] == "MALIGNANT"):
                    annotations = 1
                pathology.append(annotations)

                # r=root, d=directories, f = files
                for r, d, f in os.walk(os.path.join(data_path, dp[0])):
                    for file in f:
                        if '01.dcm' in file:
                            img_names.append(os.path.join(r, file))

            self.img_names = img_names
            self.pathology = pathology
            self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        images = []
        for i in self.img_names:
            images.append(pydicom.dcmread(self.img_names[i]).pixel_array)
        sample = {'image': images, 'annotations': self.pathology}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def test(self, idx):
        images = []
        for i in self.img_names:
            images.append(pydicom.dcmread(self.img_names[i]).pixel_array)
        sample = {'image': images, 'annotations': self.pathology}

        if self.transform:
            sample = self.transform(sample)
        return sample


# In[ ]:
