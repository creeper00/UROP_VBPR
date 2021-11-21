import concurrent.futures
import os
import requests
import json
from pathlib import Path
import gzip

p = Path("../../Downloads/meta_Clothing_Shoes_and_Jewelry.json.gz")

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

def loader() :    
    predata = []
    for i in parse(p) :
        j = json.loads(i)
        try:
            url = j['imUrl']
            pid = j['asin']
            predata.append((pid,url))
        except:
            continue
    return predata

def save_image_from_url(url):
    image = requests.get(url[1])
    with open(Path(url[0]+".jpg"), "wb") as f:
        f.write(image.content)

def load(dt):    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=5
    ) as executor:
        future_to_url = {
            executor.submit(save_image_from_url, url): url
            for url in dt
        }
        for future in concurrent.futures.as_completed(
            future_to_url
        ):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print(1)
load(loader())
