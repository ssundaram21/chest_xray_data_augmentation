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
parser.add_argument('--with_gan', type=bool, required=True)
parser.add_argument('--dataset_size', type=int, required=True)
parser.add_argument('--skip_training', type=bool, required=False)
FLAGS = parser.parse_args()

idx = FLAGS.idx
user = FLAGS.user
with_gan = FLAGS.with_gan
skip_training = FLAGS.skip_training
dataset_size = FLAGS.dataset_size

if user == "shobhita":
    data_path = "/om/user/shobhita/src/chexpert/data/CheXpert-v1.0-small/"
    output_path = "/om/user/shobhita/src/chexpert/output/output_{}/results/".format(dataset_size)
    model_path = "/om/user/shobhita/src/chexpert/output/output_{}/models/".format(dataset_size)
elif user == "neha":
    data_path = "/local/nhulkund/UROP/Chexpert/data/CheXpert-v1.0-small/train.csv"
    output_path = "/local/nhulkund/UROP/6.819FinalProjectRAMP/outputs"
    model_path = output_path
else:
    raise Exception("Invalid user")

model_name = "densenet_{}_{}".format(idx, with_gan)

print("OUTPUT PATH: {}".format(output_path))
print("MODEL PATH: {}".format(model_path + model_name))
sys.stdout.flush()

dataset_full_train, dataset_test = load_data(data_path, dataset_size, with_gan)

params = {}
model_id = 1
# for batch_size in [16, 32, 64]:
#     for lr in [1e-2, 0.005, 0.001]:
#         for optimizer in ["momentum", "adam"]:
#             params[model_id] = {
#                 "batch_size": batch_size,
#                 "lr": lr,
#                 "optimizer": optimizer
#             }
#             model_id += 1

for batch_size in [16, 32]:
    for lr in [0.001]:
        for optimizer in ["adam"]:
            params[model_id] = {
                "batch_size": batch_size,
                "lr": lr,
                "optimizer": optimizer
            }
            model_id += 1

if idx == 0:
    model_params = {}
    batch_size = 32
    lr = 0.001
    optimizer = "momentum"
else:
    model_params = params[idx]
    batch_size = model_params["batch_size"]
    lr = model_params["lr"]
    optimizer = model_params["optimizer"]

split = 0.05
val_length = int(split * len(dataset_full_train))
dataset_val, dataset_train = random_split(dataset_full_train, [val_length, len(dataset_full_train) - val_length])
dataLoaderTrain = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,  num_workers=3, pin_memory=True)
dataLoaderVal = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
dataLoaderTest = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=3, pin_memory=True)

print("Batch size: {}".format(batch_size))
print("Learning rate: {}".format(lr))
print("Optimizer: {}".format(optimizer))
print("WITH GAN DATA: {}".format(with_gan))
print("Train dataset size: {}".format(len(dataset_train)))

model = get_model()

if not skip_training:
    print("TRAINING")
    best_valid_loss, best_epoch = training(
        model=model,
        num_epochs=10,
        model_path=model_path,
        model_name=model_name,
        train_loader=dataLoaderTrain,
        valid_loader=dataLoaderVal,
        lr=lr,
        optimizer=optimizer
    )

    # EPOCH = 10
    # PATH = model_path + "{}_checkpt.pt".format(model_name)
    #
    # torch.save({
    #             'epoch': EPOCH,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict()
    #             }, PATH)
else:
    model.to(device)
    print("SKIPPED TRAINING")
sys.stdout.flush()

model.load_state_dict(torch.load(model_path + model_name))

class_names=['Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices']

print("TESTING")
sys.stdout.flush()
auc_results = testing(model, dataLoaderTest, len(class_names), class_names)

output = {}
if not skip_training:
    output["best_epoch"] = best_epoch
    output["validation_loss"] = best_valid_loss
output["params"] = model_params
output["auc"] = auc_results

with open(output_path + "{}_{}_results2.pkl".format(idx, with_gan), "wb") as handle:
    pickle.dump(output, handle)

print("Done :)")