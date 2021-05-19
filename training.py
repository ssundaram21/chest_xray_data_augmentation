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
from ChexpertDataset import CheX_Dataset
from tqdm.notebook import tqdm

from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import random
import logging
import pandas as pd
import logging


use_gpu = torch.cuda.is_available()

def load_data(path,train_file,test_file):
    # add data augmentations transforms here
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224),
                                                transforms.ToTensor()])
    
    data_aug = torchvision.transforms.Compose([
                                               transforms.RandomVerticalFlip(),
                                               #transforms.RandomErasing(p=0.5,scale=(0.1,0.3)),
                                               transforms.RandomHorizontalFlip(),
                                              ])
    # replace the paths for the dataset here
#     d_chex_no_aug = CheX_Dataset(imgpath=path,
#                                        csvpath=path + train_file,
#                                        transform=transform, views=["PA", "AP"], unique_patients=False)
#     d_chex_train_aug = CheX_Dataset(imgpath=path,
#                                        csvpath=path + train_file,
#                                        transform=transform, views=["PA", "AP"], data_aug=data_aug, unique_patients=False)
#     d_chex_train = torch.utils.data.ConcatDataset([d_chex_no_aug,d_chex_train_aug])
    d_chex_train=CheX_Dataset(imgpath=path,
                                        csvpath=path + train_file,
                                        transform=transform, views=["PA", "AP"], data_aug=data_aug, unique_patients=False)
    
    d_chex_test = xrv.datasets.CheX_Dataset(imgpath=path,
                                       csvpath=path + test_file,
                                       transform=transform, views=["PA", "AP"], unique_patients=False)
    print("train_size:"+str(len(d_chex_train)))
    print("test_size:"+str(len(d_chex_test)))
    return d_chex_train, d_chex_test

def get_model():
    model = xrv.models.DenseNet(num_classes=13)
    print(model.classifier)
    return model


def preprocess_data(dataset):
    for idx, data in enumerate(dataset):
        data['lab']=np.nan_to_num(data['lab'],0)
        data['lab']=np.where(data['lab']==-1, 1, data['lab'])
    return dataset


def training(model, num_epochs, path_trained_model, train_loader, logger, valid_loader,lr=0.001, optimizer="momentum"):
    logger.info('training')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(device)
    # hyperparameters
    criterion = nn.BCEWithLogitsLoss()
    if optimizer == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise Exception("Invalid optimizer")

    best_valid_loss = 10000
    PATH = path_trained_model

    # going through epochs
    best_epoch = 0
    for epoch in range(num_epochs):
        logger.info('epoch'+str(epoch))
        # training loss
        print("epoch", epoch)
        model.train()
        model.to(device)
        train_loss = 0
        count = 0
        for data_all in train_loader:
            count += 1
            if count % 1000 == 0:
                logger.info('data:'+str(count))
                print(count)
            data = data_all['img']
            data=data.transpose(1,2)
            target = data_all['lab']
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation loss
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data_all in valid_loader:
                data = data_all['img']
                data=data.transpose(1,2)
                target = data_all['lab']
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        
        
        # saves best epoch
        logger.info(f'Epoch: {epoch + 1}/{num_epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')
        if valid_loss < best_valid_loss:
            best_epoch = epoch + 1
            torch.save(model.state_dict(), PATH)
            best_valid_loss = valid_loss
        logger.info("Best Valid Loss so far:" + str(best_valid_loss))
        logger.info("Best epoch so far: "+str(best_epoch))
        
    # checkpoints last epoch
    torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': criterion.state_dict(),
            }, PATH+"_checkpoint.pt")
    logger.info("done!")


def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            print(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
    return outAUROC


def testing(model, test_loader, nnClassCount, class_names, output_path, model_id):
    if use_gpu:
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

    model.eval()
    # print("class count")
    # print(nnClassCount)
    # print(class_names)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print()
    # print("targets")
    with torch.no_grad():
        for batch_idx, data_all in tqdm(enumerate(test_loader)):
            if batch_idx % 100 == 0:
                print(batch_idx)

            data = data_all['img']
            data=data.transpose(1,2)
            target = data_all['lab']
            # print(target.shape)
            # print(target)
            target = target.cuda()
            data = data.to(device)
            outGT = torch.cat((outGT, target), 0).cuda()

            # bs, c, h, w = data.size()
            # varInput = data.view(-1, c, h, w)

            out = model(data)
            outPRED = torch.cat((outPRED, out), 0)

    # print(outPRED.shape, "outpred")
    # print(outGT.shape, "outgt")
    aurocIndividual = computeAUROC(outGT, outPRED, nnClassCount)
    aurocMean = np.array(aurocIndividual).mean()

    # print(len(aurocIndividual))
    # print(aurocIndividual)

    print('AUROC mean ', aurocMean)
    sys.stdout.flush()

    results = {}
    for i in range(0, len(aurocIndividual)):
        results[class_names[i]] = [aurocIndividual[i]]
        logging.info(class_names[i]+" "+str(aurocIndividual[i]))
    sys.stdout.flush()
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path + "auc_results_all_data_aug_{}.csv".format(model_id), index=False)

    return outGT, outPRED

