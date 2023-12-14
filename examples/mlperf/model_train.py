from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.device import Device
from tinygrad.nn.state import get_state_dict
from tinygrad.nn import optim
from tinygrad.jit import TinyJit
from tinygrad.helpers import GlobalCounters
from tinygrad.shape.symbolic import Node

import time
import numpy as np
import cProfile

from extra.lr_scheduler import CosineAnnealingLR
from extra.models.resnet import ResNet50
from extra.datasets import imagenet 
def train_resnet():
  BS = getenv("BS", 256)
  USE_VAL = getenv('USE_VAL', 0) #use val set as training set for testing 
  IMG_SIZE = getenv("IMG_SIZE", 224)
  EPOCHS = getenv("EPOCHS", 90)
  PROFILE = getenv("PROFILE", 0)
  imagenet_len = len(imagenet.get_val_files()) if USE_VAL else len(imagenet.get_train_files())
  e_step = imagenet_len/BS
  STEPS = EPOCHS*e_step
  lr = BS/1000 #0.256 for batch_size 256, linearly scaled for the rest -nvidia 
  imagenet.img_size = IMG_SIZE
  print(f"on Device: {Device.DEFAULT}")
  print(f"BS:{BS}, IMG_SIZE:{IMG_SIZE} LR:{lr}, EPOCHS:{EPOCHS}, S_per_EPOCH:{e_step}/ STEPS:{STEPS}, len:{imagenet_len}, USE_VAL:{USE_VAL}")

  rand = np.random.default_rng()
  
  # img is (n)CHW
  training_aug = [
    # move tensor to device to make ops
    lambda img: img.to(device=Device.DEFAULT).float(),
    # normalize to [0.,1.]
    lambda img: img/255.0,
    # normalize as signals
    lambda img: (img - imagenet.mean.repeat((1,1,IMG_SIZE,IMG_SIZE)) / imagenet.std.repeat((1,1,IMG_SIZE,IMG_SIZE))),
    # horizontal flip 50% axes:n,c,h,[w]
    lambda img: img.flip(-1) if rand.choice([0,1])==1 else img
  ]

  # img is (n)CHW
  eval_aug = [
    # move tensor to device to make ops
    lambda img: img.to(device=Device.DEFAULT).float(),
    # normalize to [0.,1.]
    lambda img: img/255.0,
    # normalize as signals
    lambda img: (img - imagenet.mean.repeat((1,1,IMG_SIZE,IMG_SIZE)) / imagenet.std.repeat((1,1,IMG_SIZE,IMG_SIZE))),
  ]
  model = ResNet50()
  params_dict = get_state_dict(model)
  params_bias = []
  params_non_bias = []
  for params in params_dict:
      if params_dict[params].requires_grad is not False:
          if 'bias' in params:
              params_bias.append(params_dict[params])
          else:
              params_non_bias.append(params_dict[params])

  opt  = optim.SGD(params=params_non_bias, lr=BS/1000, momentum=0.875, weight_decay=1/32768)
  opt_bias = optim.SGD(params=params_bias, lr=BS/1000, momentum=0.875, weight_decay=0)

  #no warmup??
  lr_sched = CosineAnnealingLR(optimizer=opt, T_max=STEPS, eta_min=0)
  lr_sched_bias = CosineAnnealingLR(optimizer=opt_bias, T_max=STEPS, eta_min=0)

  #tookied from timm
  def lscross_entropy(pred:Tensor, truth:Tensor, label_smoothing):
      print(f"   pred: {pred}")
      print(f"  truth: {truth}")
      print(f"truth.u: {truth.unsqueeze(1)}")
      print(f"pd:{pred.ndim}, td:{truth.ndim}")
      logprobs = pred.log_softmax(axis=-1)
      print(f"logd:{logprobs.ndim}, td:{truth.unsqueeze(1).ndim}")
      nll_loss = -logprobs.gather(idx=truth.unsqueeze(1), dim=-1)
      nll_loss = nll_loss.squeeze(1)
      smooth_loss = -logprobs.mean(axis=-1)
      loss = (1.0-label_smoothing) * nll_loss + label_smoothing * smooth_loss
      return loss.mean()
    
  def cross_entropy(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
    divisor = y.shape[1]
    assert not isinstance(divisor, Node), "sint not supported as divisor"
    y = (1 - label_smoothing)*y + label_smoothing / divisor
    if reduction=='none': return -x.log_softmax(axis=1).mul(y).sum(axis=1)
    if reduction=='sum': return -x.log_softmax(axis=1).mul(y).sum(axis=1).sum()
    return -x.log_softmax(axis=1).mul(y).sum(axis=1).mean()


  @TinyJit
  def train_step(model, optimizer, lr_scheduler, imgs, labels):
      preds = model(imgs)
      loss = cross_entropy(preds, labels, label_smoothing=0.1)

      optimizer[0].zero_grad()
      optimizer[1].zero_grad()
      loss.backward()

      optimizer[0].step()
      optimizer[1].step()
      lr_scheduler[0].step()
      lr_scheduler[1].step()

      return loss.realize()

  @TinyJit
  def eval_step(model, imgs, labels):
    preds = model(imgs)
    loss = cross_entropy(preds, labels, label_smoothing=0.1)
    correct = preds.argmax(axis=1) == labels.argmax(axis=1)
    return correct.realize(), loss.realize()
  
  def one_step(i):
    imgs, labels = imagenet.my_fetch_batch(bs=BS, val=USE_VAL, inference=0)
    imgs_test, labels_test = imagenet.my_fetch_batch(bs=BS, val=1, inference=1)

    imgs = Tensor(imgs).sequential(training_aug)
    imgs_test = Tensor(imgs_test).sequential(eval_aug)
    labels = Tensor(labels).to(device=Device.DEFAULT).float()
    labels_test = Tensor(labels_test).to(device=Device.DEFAULT).float()

    labels = Tensor.eye(1000)[labels]#onehot
    labels_test = Tensor.eye(1000)[labels_test]#onehot

    GlobalCounters.reset()
    st = time.monotonic()

    loss = train_step(model, [opt_bias, opt], [lr_sched_bias, lr_sched], imgs, labels)
    correct, eval_loss = eval_step(model, imgs_test, labels_test)

    et = time.monotonic()
    loss_cpu = loss.numpy()
    e_loss_cpu = eval_loss.numpy()
    corrects = correct.numpy().tolist()
    e_acc = (sum(corrects)/len(corrects))*100.
    cl = time.monotonic()
    print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {e_loss_cpu:7.2f} eval_loss, {e_acc:7.2f} eval_acc, {opt.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    return e_loss_cpu
  
  def x_steps(x):
    for i in range(x):
      one_step(i)
      
  if PROFILE:
    print(f" First 10:")
    cProfile.runctx("x_steps(10)", globals(), locals(), sort=1)
    print(f"Second 10:")
    cProfile.runctx("x_steps(10)", globals(), locals(), sort=1)
    return

  best_eval = None
  i = 0
  while i <= STEPS:
    if STEPS == 0 or i==STEPS: break

    if i%e_step==0:
      print(f"EPOCH {i//e_step}/{EPOCHS} (best_eval:{best_eval}):")

    eloss = one_step(i)
    if best_eval is None:
      best_eval = eloss
    elif eloss < best_eval:
      best_eval = eloss

    i += 1

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

import os
if __name__ == "__main__":
  os.environ["MODEL"] = "resnet"
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


