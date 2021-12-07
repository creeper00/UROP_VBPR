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

print(0)
#model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=50)
model = torch.load('../datasets/model2.pt')
print(1)


tfms1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
tfms2 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
'''
with open('../datasets/deepfashion/train.txt') as f :
    tr = f.readlines()
#with open('../datasets/deepfashion/text.txt') as f :
 #   te = f.readlines()
with open('../datasets/deepfashion/val.txt') as f :
    va = f.readlines()

with open('../datasets/deepfashion/train_cate.txt') as f :
    trc = f.readlines()
#with open('../datasets/deepfashion/text_cate.txt') as f :
 #   tec = f.readlines()
with open('../datasets/deepfashion/val_cate.txt') as f :
    vac = f.readlines()
'''

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

def evaluateTop3(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    print(total)
    for x, y in loader:
        x,y = x.squeeze(1).to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1,3))
            yr = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, yr).sum().float().item()
    return correct / total

#pdataset = ImgDataset()
#print(2)
#pdataloader = data.DataLoader(pdataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=False)

#trds = ImgDataset(tr, trc)
#teds = ImgDataset(te, tec)
#vads = ImgDataset(va, vac)

#trdl =  data.DataLoader(trds, batch_size=96, shuffle=True, pin_memory=True, drop_last=False)
#tedl =  data.DataLoader(teds, batch_size=32, shuffle=False, pin_memory=True, drop_last=False)
#vadl =  data.DataLoader(vads, batch_size=96, shuffle=False, pin_memory=True, drop_last=False)


data = ImageDataLoaders.from_csv("../datasets/deepfashion", csv_fname="train_labels.csv",
                                 item_tfms=Resize(300),
                                 batch_tfms=aug_transforms(size=224, min_scale=0.9),
                                 valid_pct=0.1,
                                 splitter=RandomSplitter(seed=42),
                                 num_workers=0)

#data = { 'train' : trdl, 'val' : vadl }
#print(len(data.valid))
data = { 'train' : data.train, 'val' : data.valid }
print(data['train'].one_batch())
print(3)
model.to(device)
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("Params to learn:")
for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #inputs = inputs.squeeze(1).to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(labels) 

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

criterion = nn.CrossEntropyLoss()
model_ft, hist = train_model(model, data, criterion, optimizer_ft, num_epochs=2, is_inception=False)
torch.save(model_ft, "../datasets/model2.pt")
