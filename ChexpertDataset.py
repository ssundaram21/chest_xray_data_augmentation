from PIL import Image
from os.path import join
from skimage.io import imread, imsave
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os,sys,os.path
import pandas as pd
import pickle
import pydicom
import skimage
import glob
import collections
import pprint
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import skimage.transform
import warnings
import tarfile
import zipfile
import random

class CheX_Dataset(Dataset):
    """
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
    Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, 
    Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, 
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, 
    Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. https://arxiv.org/abs/1901.07031
    
    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """
    def __init__(self, imgpath, csvpath, views=["PA"], transform=None, data_aug=None,
                 flat_dir=True, seed=0, unique_patients=True):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255
        
        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]
        
        self.pathologies = sorted(self.pathologies)
        
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        
        # To list
        if type(self.views) is not list:
            views = [views]
            self.views = views
              
        self.csv["view"] = self.csv["Frontal/Lateral"] # Assign view column 
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"] # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"}) # Rename Lateral with L  
        self.csv = self.csv[self.csv["view"].isin(self.views)] # Select the view 
         
        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat = '(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()
                   
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
                
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        # make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan
        
        # rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        
        ########## add consistent csv values
        
        # offset_day_int
        
        # patientid
        if 'train' in csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif 'valid' in csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplemented

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient","")
        self.csv["patientid"] = patientid
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        imgid = self.csv['Path'].iloc[idx]
        imgid = imgid.replace("CheXpert-v1.0-small/","")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)      
        
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)
            
        if self.data_aug is not None:
            #indices which are augmented
            augmented_indices=[3,10,11]
            augment=False
            for i in augmented_indices:
                if self.labels[idx][i]==1:
                    augment=True
            if augment:
                img = self.data_aug(img)

        return {"img":img, "lab":self.labels[idx], "idx":idx}
    
class XRayResizerCustom(object):
    def __init__(self, size, engine="skimage"):
        self.size = size
        self.engine = engine
        if 'cv2' in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img):
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            import cv2 # pip install opencv-python
            return cv2.resize(img[0,:,:], 
                              (self.size, self.size), 
                              interpolation = cv2.INTER_AREA
                             ).reshape(1,self.size,self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")

class XRayCenterCropCustom(object):
    
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y,x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:,starty:starty + crop_size, startx:startx + crop_size]
    
    def __call__(self, img):
        return self.crop_center(img)

def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""
    
    if sample.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(sample.max(), maxval))
    
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    #sample = sample / np.std(sample)
    return sample

def relabel_dataset(pathologies, dataset, silent=False):
    """
    Reorder, remove, or add (nans) to a dataset's labels.
    Use this to align with the output of a network.
    """
    will_drop = set(dataset.pathologies).difference(pathologies)
    if will_drop != set():
        if not silent:
            print("{} will be dropped".format(will_drop))
    new_labels = []
    dataset.pathologies = list(dataset.pathologies)
    for pathology in pathologies:
        if pathology in dataset.pathologies:
            pathology_idx = dataset.pathologies.index(pathology)
            new_labels.append(dataset.labels[:,pathology_idx])
        else:
            if not silent:
                print("{} doesn't exist. Adding nans instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(np.nan)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T
    
    dataset.labels = new_labels
    dataset.pathologies = pathologies
   