from fastai import *
from fastai.vision.all import *
import torch
from fastai.interpret import *
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import ImageFile
from PIL import Image
import requests
from io import BytesIO
import torch.nn.functional as nnf
import torch.nn as nn
import torch.utils.data as data

ImageFile.LOAD_TRUNCATED_IMAGES = True

model = torch.load("../datasets/model.pt")
p = Path("../datasets/train_labels.csv")
df = pd.read_csv(p, delimiter = ',')

tfms1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
tfms2 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

print(df.shape)
print(df.iloc[0].image_name)

class ImgDataset(data.Dataset):

    def __init__(self, ilist):
        self.ilist = ilist

    def __len__(self):
        return len(self.ilist)

    def __getitem__(self, index):
        label = self.ilist.iloc[index].category_name
        pid = self.ilist.iloc[index].image_name
        path = "../datasets/deepfashion/"+pid
        try:
            img = Image.open(path)
        except:
            img = torch.zeros(1,3,224,224)
            return pid, img
        try:
            img = tfms1(img).unsqueeze(0)
        except:
            img = tfms2(img).unsqueeze(0)
        return label, img

device = torch.device("cuda")
pdataset = ImgDataset(ilist=df)
pdataloader = data.DataLoader(pdataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=False)

model.to(device)
correct = 0
total = 0
with torch.no_grad():
    for data in pdataloader:
        labels, images = data
        output = []
        for img in images:
            oupt = model(img.to(device))
            output.append(oupt)
        output = torch.stack(output)
        _, predicted = torch.max(output, 1)
        print(output)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))





