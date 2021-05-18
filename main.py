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
parser.add_argument('--datasetsplit',type=int, required=True)
FLAGS = parser.parse_args()

idx = FLAGS.idx
user = FLAGS.user
datasetsplit=FLAGS.datasetsplit

if user == "shobhita":
    data_path = "/om/user/shobhita/src/chexpert/data/CheXpert-v1.0-small/"
    output_path = "/om/user/shobhita/src/chexpert/output/"
elif user == "neha":
    data_path = "/local/nhulkund/UROP/Chexpert/data/CheXpert-v1.0-small/"
    output_path = "/local/nhulkund/UROP/6.819FinalProjectRAMP/outputs/"
else:
    raise Exception("Invalid user")

if datasetsplit == 50:
    train_filename = 'train_preprocessed_subset_50.csv'
    n_epochs=15
elif datasetsplit == 10:
    train_filename = 'train_preprocessed_subset_10.csv'
    n_epochs=100
elif datasetsplit == 1:
    train_filename = 'train_preprocessed_subset_1.csv'
    n_epochs=40
elif datasetsplit == 5:
    train_filename = 'train_preprocessed_subset_5.csv'
    n_epochs=40
else:
    train_filename = 'train_preprocessed.csv'
    n_epochs=20
test_filename = 'test_train_preprocessed.csv'

dataset_full_train, dataset_test = load_data(data_path,train_filename,test_filename)

params = {}
model_id = 1
for batch_size in [8]:
    for lr in [0.01,0.001]:
        for optimizer in ["adam",'momentum']:
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


logging.basicConfig(filename='/local/nhulkund/UROP/6.819FinalProjectRAMP/log/idx_{}_{}_{}.log'.format(idx,datasetsplit,'all_data_aug'), filemode='w',level=logging.DEBUG)

logger1 = logging.getLogger('basic')
logger1.info('started logging')
if idx>0:
    logger1.info('model params: batch_size: %d, learning_rate:%.5f, optimizer: %s' % (batch_size, learning_rate, optimizer) )


split = 0.05
val_length = int(split * len(dataset_full_train))
dataset_val, dataset_train = random_split(dataset_full_train, [val_length, len(dataset_full_train) - val_length])
dataLoaderTrain = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,  num_workers=3, pin_memory=True)
dataLoaderVal = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
dataLoaderTest = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=3, pin_memory=True)


model = get_model()
model_path="models/densenet_model_{}_{}_{}".format(idx,datasetsplit,'all_data_aug')

training(
    model=model,
    num_epochs=n_epochs,
    path_trained_model=model_path,
    train_loader=dataLoaderTrain,
    valid_loader=dataLoaderVal,
    logger=logger1
)
model.to(device)
model.load_state_dict(torch.load(model_path))

class_names=['Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices']

testing(model, dataLoaderTest, len(class_names), class_names, output_path, str(idx)+"_"+str(datasetsplit))

print("Done :)")