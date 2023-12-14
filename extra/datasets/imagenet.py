# for imagenet download prepare.sh and run it
import glob, random
import json
import numpy as np
from PIL import Image
import functools, pathlib
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

imgnet_dir = getenv("IMGNET_DIR", pathlib.Path(__file__).parent / "imagenet")
imgnet_dir = pathlib.Path(imgnet_dir)
print(f"imagenet_dir: {imgnet_dir}")

ci = json.load(open( imgnet_dir / "imagenet_class_index.json"))
cir = {v[0]: int(k) for k,v in ci.items()}

mean = Tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
std = Tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
img_size = None

@functools.lru_cache(None)
def get_train_files():
  train_files = open(imgnet_dir / "train_files").read().strip().split("\n")
  return [(imgnet_dir / "train" / x).as_posix() for x in train_files]

@functools.lru_cache(None)
def get_val_files():
  val_files = glob.glob(str(imgnet_dir / "val/*/*"))
  return [pathlib.Path(p).as_posix() for p in val_files]

#rrc = transforms.RandomResizedCrop(224)
#don't feel good about this but it already used torch functional..?
import torchvision.transforms as torch_t
# image loader does the resizing and cropping.. via torchvision transforms..?
# left out RANDOM FLIP and NORMALIZATION to tinygrad Tensor don't forget
def image_load(file):
  img = Image.open(file).convert('RGB')
  img = torch_t.RandomResizedCrop(size=[img_size,img_size],
                                  scale=(0.08, 1.),
                                  ratio=(3.0/4.0, 4.0/3.0),
                                  interpolation=torch_t.InterpolationMode.BILINEAR).forward(img)
  ret = np.array(img).transpose(2,0,1)
  return ret

def image_load_val(file):
  img = Image.open(file).convert('RGB')
  img = torch_t.Resize((img_size+32, img_size+32)).forward(img)
  img = torch_t.CenterCrop((img_size, img_size)).forward(img)
  ret = np.array(img).transpose(2,0,1)
  return ret

def iterate(bs=32, val=True, shuffle=True):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  from multiprocessing import Pool
  p = Pool(16)
  for i in range(0, len(files), bs):
    X = p.map(image_load, [files[i] for i in order[i:i+bs]])
    Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
    yield (np.array(X), np.array(Y))

def fetch_batch(bs, val=False):
  files = get_val_files() if val else get_train_files()
  samp = np.random.randint(0, len(files), size=(bs))
  files = [files[i] for i in samp]
  X = [image_load(x) for x in files]
  Y = [cir[x.split("/")[-2]] for x in files]
  return np.array(X), np.array(Y)

def my_fetch_batch(bs, val=False, inference=False, shuffle=-1):
  files = get_val_files() if val else get_train_files()

  if shuffle == -1:#shuffle true
    samp = np.random.randint(0, len(files), size=(bs))
  else:#shuffle is the last batch pos.
    samp = range(shuffle, min(shuffle+bs, len(files)))
  files = [files[i] for i in samp]

  #i use val set to train for now since train set is large mb change later
  if inference:
    X = [image_load_val(x) for x in files]
  else:
    X = [image_load(x) for x in files]

  Y = [cir[x.split("/")[-2]] for x in files]
  return np.array(X), np.array(Y)

if __name__ == "__main__":
  X,Y = fetch_batch(64)
  print(X.shape, Y)

