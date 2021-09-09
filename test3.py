import numpy as np
from pathlib import Path
import array
import cornac
from cornac.datasets import amazon_clothing
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit
from vbpr import VBPR
import pandas as pd

p = Path('../Downloads/ratings_Cell_Phones_and_Accessories.csv')
df = pd.read_csv(p, delimiter=',')
feedback = [tuple(row[0:3]) for row in df.values]

def readImageFeatures(path):
  f = open(path, 'rb')
  for i in range(0, 303890):
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()
print(1)
p = Path('../Downloads/image_features_Cell_Phones_and_Accessories.b')
ft = []
item_ids = []
for i in readImageFeatures(p):
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
    use_gpu=False,
)

auc = cornac.metrics.AUC()
rec_50 = cornac.metrics.Recall(k=50)

# Evaluation
cornac.Experiment(eval_method=ratio_split, models=[vbpr], metrics=[auc, rec_50]).run()
