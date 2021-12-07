import cv2
import numpy as np
from pathlib import Path
import requests
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import ImageFile
from PIL import Image
import requests
from io import BytesIO
import torch.nn.functional as nnf
import torch.nn as nn
import json
import ast
import gzip
import array
import torch.utils.data as data
import torch
import torch.optim as optim
import time
from copy import deepcopy
from fastai import *
from fastai.vision import *
from fastai.vision.all import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

model = torch.load('../datasets/model2.pt')
#model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)

tfms1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
tfms2 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

with open('../datasets/deepfashion/test.txt') as f :
    te = f.readlines()
with open('../datasets/deepfashion/test_cate.txt') as f :
    tec = f.readlines()

device = torch.device("cuda")
print(1)

class ImgDataset(data.Dataset):

    def __init__(self, im, lb):
        self.im = im
        self.lb = lb

    def __len__(self):
        return len(self.lb)

    def __getitem__(self, index):
        label = torch.tensor(int(self.lb[index]))
        path = "../datasets/deepfashion/"+self.im[index]
        try:
            img = Image.open(path)
        except:
            img = torch.zeros(1,3,224,224)
            return img, label
        try:
            img = tfms1(img).unsqueeze(0)
        except:
            img = tfms2(img).unsqueeze(0)
        return img, label

#teds = ImgDataset(te, tec)
#tedl =  data.DataLoader(teds, batch_size=32, shuffle=False, pin_memory=True, drop_last=False)

data = ImageDataLoaders.from_csv("../datasets/deepfashion", csv_fname="test_labels.csv",
                                 item_tfms=Resize(300),
                                 batch_tfms=aug_transforms(size=224, min_scale=0.9),
                                 valid_pct=0.1,
                                 splitter=RandomSplitter(seed=42),
                                 num_workers=0)

model.to(device)

def evaluateTop3(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    #print(total)
    for x, y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1,3))
            yr = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, yr).sum().float().item()
    return correct / total

def evaluateTop5(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1,5))
            y_resize = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total

def evaluateTop1(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
        #correct += torch.eq(pred, y).sum().item()
    return correct / total


print("top5 : "+str(float(str(evaluateTop5(model, data.train)))*100)+"%")
print("top3 : "+str(float(str(evaluateTop3(model, data.train)))*100)+"%")
print("top1 : "+str(float(str(evaluateTop1(model, data.train)))*100)+"%")
