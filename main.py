import os
import numpy as np
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torch.nn.functional as func
import torchxrayvision as xrv
from tqdm.notebook import tqdm

from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import random
import logging

import time
import os
import copy
import argparse
import pickle

import pandas as pd

from training import load_data, get_model, training, testing


use_gpu = torch.cuda.is_available()
print("Using GPU: {}".format(use_gpu))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")


parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, required=True)
parser.add_argument('--user', type=str, required=True)
FLAGS = parser.parse_args()

idx = FLAGS.idx
user = FLAGS.user

if user == "shobhita":
    data_path = "/om/user/shobhita/src/chexpert/data/CheXpert-v1.0-small/"
    output_path = "/om/user/shobhita/src/chexpert/output/"
elif user == "neha":
    data_path = "/local/nhulkund/UROP/Chexpert/data/CheXpert-v1.0-small/train.csv"
    output_path = "/local/nhulkund/UROP/6.819FinalProjectRAMP/outputs"
else:
    raise Exception("Invalid user")
train, datasetTest = load_data(data_path)

params = {}
model_id = 1
for batch_size in [16, 32, 64]:
    for lr in [1e-2, 0.005, 0.001]:
        for optimizer in ["momentum", "adam"]:
            params[model_id] = {
                "batch_size": batch_size,
                "lr": lr,
                "optimizer": optimizer
            }
            model_id += 1

if idx == 0:
    batch_size = 32
    lr = 0.01
    optimizer = "momentum"
else:
    model_params = params[idx]
    batch_size = model_params["batch_size"]
    learning_rate = model_params["lr"]
    optimizer = model_params["optimizer"]


split = 0.1
val_length = int(split * len(train))
datasetVal, datasetTrain = random_split(train, [val_length, len(train) - val_length])
dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=batch_size, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, num_workers=24, pin_memory=True)


model = get_model()

training(
    model=model,
    num_epochs=2,
    path_trained_model="models/densenet_model_{}".format(idx),
    train_loader=dataLoaderTrain,
    valid_loader=dataLoaderVal
)

model.load_state_dict(torch.load("models/densenet_model_{}".format(idx)))

class_names=['Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices']

testing(model, dataLoaderTest, len(class_names), class_names, output_path, idx)

print("Done :)")