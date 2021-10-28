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

print(0)
model = EfficientNet.from_pretrained('efficientnet-b0')
print(1)
p = Path("../datasets/meta_Cell_Phones_and_Accessories.json.gz")

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

print(3)
x = 0
file = open("../datasets/efficient_Cell_Phones_and_Accessories.b", "wb")
for i in parse(p) :
    x=x+1
    j = json.loads(i)
    try:
        url = j['imUrl']
    except:
        print(x)
        continue
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    tfms1 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    tfms2 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    try:
        img = tfms1(img).unsqueeze(0)
    except:
        img = tfms2(img).unsqueeze(0)
    features = model.extract_features(img)
    m = nn.Linear(81920, 4096)
    f = m(features.view(-1))
    pid = j['asin'].encode()
    data = (pid, f.detach().numpy().tolist())
    file.write(pid)
    fl = f.detach().numpy().tolist()
    file.write(bytes(array.array('f', fl)))


