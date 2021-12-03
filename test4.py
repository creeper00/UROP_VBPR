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

ImageFile.LOAD_TRUNCATED_IMAGES = True

print(0)
#model = EfficientNet.from_pretrained('efficientnet-b0')
model = torch.load("../datasets/model.pt")
print(1)
p = Path("../datasets/meta_Clothing_Shoes_and_Jewelry.json.gz")

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

tfms1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
tfms2 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

def loader() :
    predata = []
    for i in parse(p) :
        j = json.loads(i)
        pid = j['asin']
        predata.append(pid)
    return predata

class ImgDataset(data.Dataset):

    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        pid = self.img_list[index]
        path = "../datasets/Amazon_Clothing_Shoes_and_Jewelry_Img/"+pid+".jpg"
        try:
            img = Image.open(path)
        except:
            img = torch.zeros(1,3,224,224)
            return pid, img
        try:
            img = tfms1(img).unsqueeze(0)
        except:
            img = tfms2(img).unsqueeze(0)
        return pid, img

class ExtractEfficientNet(nn.Module) :
    def __init__(self):
         super(ExtractEfficientNet, self).__init__()
         self.features = model.extract_features
         self.gap = nn.AdaptiveAvgPool2d((4, 4)) 
    def forward(self, x):
        x = self.features(x)
        x = x.view([1,5,256,7,7])
        x = torch.mean(x,1)
        x = self.gap(x)
        x=x.squeeze(-1)
        return x

device = torch.device("cuda")
print(1)
pdataset = ImgDataset(img_list=loader())
print(2)
pdataloader = data.DataLoader(pdataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=False)

nmodel = ExtractEfficientNet()
print(3)
model.to(device)
nmodel.to(device)
file = open(Path("../datasets/deep_Clothing_Shoes_and_Jewelry.b"), "wb")
for batch in pdataloader :
    paths, images = batch
    inpt = images.to(device)
    for path, i in zip(paths, inpt) :
        x = nmodel(i).view(-1)
        f = x.detach().cpu().numpy().tolist()
        file.write(path.encode())
        file.write(bytes(array.array('f',f)))
    #for i in range(0, len(batch[0])):
        #inpt = batch[1][i].to(device)
        #feat = nmodel(inpt)
        #f = m(feat.view(-1))
        #fl = f.detach().cpu().numpy().tolist()
        #file.write(batch[0][i])
        #file.write(bytes(array.array('f', fl)))
#file.close()
