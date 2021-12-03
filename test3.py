import numpy as np
from pathlib import Path
import array
import cornac
from cornac.datasets import amazon_clothing
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit
from vbpr import VBPR
import pandas as pd
import torch

p = Path('../datasets/ratings_Cell_Phones_and_Accessories.csv')
df = pd.read_csv(p, delimiter=',')
feedback = [tuple(row[0:3]) for row in df.values]
print(torch.cuda.device_count())
def readImageFeatures(path):
  f = open(path, 'rb')
  while True  :
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    try:
      a.fromfile(f, 4096)
      yield asin, a.tolist()
    except:
      yield b'-1', []
print(1)
p = Path('../datasets/efficient_Cell_Phones_and_Accessories.b')
ft = []
item_ids = []
num=0
div = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for idx, i in enumerate(readImageFeatures(p)):
    if i[0] != b'-1' :
      try:
        it = torch.tensor(i[1]).to(device)
        ft.append([it])
        item_ids.append(i[0])
      except:
          try:
              device = torch.device("cuda:1")
              it = torch.tensor(i[1]).to(device)
              ft.append([it])
              item_ids.append(i[0])
          except :
              print('second')
              print(idx)
              device = torch.device("cuda:2")
              it = torch.tensor(i[1]).to(device)
              ft.append([it])
              item_ids.append(i[0])
      if idx/div == 1 : 
          print(div)
          div = div*2
    else :
      break
print(2)
features = np.array(ft, dtype=object)
print(len(features.shape))
item_image_modality = ImageModality(features=features, ids=item_ids)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.01,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
    item_image=item_image_modality,
)

# VBPR
vbpr = VBPR(
    k=10,
    k2=20,
    n_epochs=7,
    batch_size=500,
    learning_rate=0.01,
    lambda_w=1,
    lambda_b=0.01,
    lambda_e=0.0,
    use_gpu=True,
)

auc = cornac.metrics.AUC()
rec_50 = cornac.metrics.Recall(k=50)

# Evaluation
cornac.Experiment(eval_method=ratio_split, models=[vbpr], metrics=[auc, rec_50]).run()
