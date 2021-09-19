import numpy as np
from pathlib import Path
import array
import cornac
from cornac.datasets import amazon_clothing
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit
from vbpr import VBPR
import pandas as pd

p = Path('../datasets/ratings_Cell_Phones_and_Accessories.csv')
df = pd.read_csv(p, delimiter=',')
feedback = [tuple(row[0:3]) for row in df.values]

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    try:
      asin = f.read(10)
      if asin == '': break
      a = array.array('f')
      a.fromfile(f, 4096)
      yield asin, a.tolist()
    except:
      yield b'-1', []
print(1)
p = Path('../datasets/image_features_Cell_Phones_and_Accessories.b')
ft = []
item_ids = []
for i in readImageFeatures(p):
    if i[0] != b'-1' :
      ft.append(i[1])
      item_ids.append(i[0])
print(2)
features = np.array(ft)

item_image_modality = ImageModality(features=features, ids=item_ids, normalized=True)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
    item_image=item_image_modality,
)

# VBPR
vbpr = VBPR(
    k=10,
    k2=20,
    n_epochs=20,
    batch_size=50,
    learning_rate=0.005,
    lambda_w=1,
    lambda_b=0.01,
    lambda_e=0.0,
    use_gpu=True,
)

auc = cornac.metrics.AUC()
rec_50 = cornac.metrics.Recall(k=50)

# Evaluation
cornac.Experiment(eval_method=ratio_split, models=[vbpr], metrics=[auc, rec_50]).run()
