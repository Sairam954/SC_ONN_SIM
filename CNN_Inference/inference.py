import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
from helperfunction import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset
import deeplake
import torchvision
from torchvision.models.quantization import resnet50,googlenet,mobilenet_v2,shufflenet_v2_x1_0,ShuffleNet_V2_X1_0_QuantizedWeights,ResNet50_QuantizedWeights,GoogLeNet_QuantizedWeights,MobileNet_V2_QuantizedWeights

best_acc1 = 0
# torch/ao/nn/quantized/modules/conv.py
def main():
    # Load dataset using hub.ai 
    ds = deeplake.load("hub://activeloop/imagenet-val")
    # Transform the images and normalize them 
    tform = transforms.Compose([
            transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
            transforms.Resize(256),
            transforms.CenterCrop(224), # Image augmentation
            transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)), # Some images are grayscale, so we need to add channels
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    # Load the dataset into torch.loader
    dataloader = ds.pytorch(num_workers=0, batch_size=1, transform = {'images': tform, 'labels': None}, shuffle = True)
    val_loader = dataloader
    # loss function
    criterion = nn.CrossEntropyLoss()
    # Initiating a trained model
    # model = torchvision.models.resnet50(pretrained=True)
    
    # Initiating a quantized model resnet50
    # model = torchvision.models.quantization.resnet50(quantize=True, pretrained=True)
    weights = ShuffleNet_V2_X1_0_QuantizedWeights.DEFAULT
    
    print("==========")
    model = shufflenet_v2_x1_0(weights=weights, quantize=True)
    # print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move model to device
    model.to(device)
    # change model to evaluation or inference mode
    model.eval()
    # validate the model 
    validate(val_loader, model, criterion)

if __name__ == '__main__':
    main()