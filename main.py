import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import random

from utils.data_preprocess import get_train_val_set, Prepare_dataset
from torch.utils.data import DataLoader
from cfg import cfg
# from utils.model import Network
import torch.optim as optim 

from utils.train_tool import train
from utils.test_tool import load_parameters, test_result

# from efficientnet_pytorch import EfficientNet

# import torchvision.models as models
from utils.mobilenetv3 import mobilenetv3_small, mobilenetv3_large


# ------- load hyperparameters from cfg ---------
myseed = cfg['seed']
batch_size = cfg['batch_size']
num_epoch = cfg['num_epoch']
lr = cfg['lr']
milestones = cfg['milestones']
milestones_gamma = cfg['milestones_gamma']
save_path = cfg['save_path']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(myseed)
random.seed(myseed)
torch.manual_seed(myseed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    torch.cuda.manual_seed(myseed)
# --------------------------------------------


# step 1: load data and perform preprocess (transform and augementation)
trainset_path = "./dataset/train_data/synthetics_train/annot.pkl"
valset_path = "./dataset/train_data/aflw_val/annot.pkl"
train_set, val_set = get_train_val_set(trainset_path, valset_path)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

print('done DataLoader of training and validating set')


# step 2: define Network
# model = EfficientNet.from_pretrained('efficientnet-b0')
# model = models.mobilenet_v3_small(pretrained=True)
# model = mobilenetv3_small()
model = mobilenetv3_large()

# model = Network(output_class=136)
model = model.to(device)
print(model)


# step 3: training
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=milestones_gamma, last_epoch=-1)

train(model=model, train_loader=train_loader, val_loader=val_loader, num_epoch=num_epoch,
    	save_path=save_path, device=device, optimizer=optimizer, scheduler=scheduler)


# step 4: testing
# load model parameters
model_saved_path = "./bestmodel/best_model.pt"
load_parameters(model=model, path=model_saved_path)

# test model on result
testset_path = "./dataset/aflw_test"
test_result(testset_path=testset_path, model=model, device=device)
    
