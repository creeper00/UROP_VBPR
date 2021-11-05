import cv2
import numpy as np
from pathlib import Path
import requests
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
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

print(0)
model = EfficientNet.from_pretrained('efficientnet-b0')
print(1)
p = Path("../datasets/meta_Cell_Phones_and_Accessories.json.gz")

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

tfms1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
tfms2 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

def loader() :
    predata = []
    x=0
    for i in parse(p) :
        x=x+1
        if x==100 : break
        j = json.loads(i)
        try:
            url = j['imUrl']
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except:
            print(x)
            continue
        try:
            img = tfms1(img).unsqueeze(0)
        except:
            img = tfms2(img).unsqueeze(0)
        pid = j['asin'].encode()
        tup = (pid, img)
        predata.append(tup)
    return predata

class ImgDataset(data.Dataset):

    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.img_list[index]

        return img

class ExtractEfficientNet(nn.Module) :
    def __init__(self):
         super(ExtractEfficientNet, self).__init__()
         self.features = model.extract_features
    def forward(self, x):
        x = self.features(x)
        return x

device = torch.device("cuda")
print(1)
pdataset = ImgDataset(img_list=loader())
print(2)
pdataloader = data.DataLoader(pdataset, batch_size=32, shuffle=False)

nmodel = ExtractEfficientNet()
print(3)

file = open("../datasets/efficient_Cell_Phones_and_Accessories.b", "wb")
for batch in pdataloader :
    nmodel.to(device)
    for i in range(0, len(batch[0])):
        feat = nmodel(batch[1][i])
        m = nn.Linear(62720, 4096)
        f = m(feat.view(-1))
        fl = f.detach().numpy().tolist()
        file.write(batch[0][i])
        file.write(bytes(array.array('f', fl)))
file.close()
