import urllib.request
import json
from pathlib import Path
import gzip

p = Path("../../Downloads/meta_Cell_Phones_and_Accessories.json.gz")

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

def loader() :    
    for i in parse(p) :
        j = json.loads(i)
        try:
            url = j['imUrl']
            pid = j['asin']
            name = pid+".jpg"
            urllib.request.urlretrieve(url, name)
        except:
            continue

loader()
