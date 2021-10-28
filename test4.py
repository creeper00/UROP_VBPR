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
p = Path("../Downloads/meta_Cell_Phones_and_Accessories.json.gz")

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

print(3)
x = 0
file = open("../Downloads/efficient_Cell_Phones_and_Accessories.b", "wb")
for i in parse(p) :
    j = json.loads(i)
    url = j['imUrl']
    print(url)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    tfms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(img).unsqueeze(0)
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    if img.shape[1] == 1 :
        img = transform(img)
    features = model.extract_features(img)
    m = nn.Linear(81920, 4096)
    f = m(features.view(-1))
    pid = j['asin'].encode()
    data = (pid, f.detach().numpy().tolist())
    file.write(pid)
    fl = f.detach().numpy().tolist()
    file.write(bytes(array.array('f', fl)))



