# import statements for python, torch and companion libraries and your own modules
import torch
import os
from glob import glob
from pathlib import Path
from PIL import Image

# import Jupyter variables 
from dataset2 import COCOTrainImageDataset, COCOTestImageDataset
from loops2 import train_loop, validation_loop
from utils2 import update_graphs

# global variables defining training hyper-parameters among other things, data directories initialization
from config import CONFIG

# device initialization

## Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU detected. Using CUDA for training.")
else:
    print("No GPU detected. Training on CPU will be significantly slower. Please be sure of the model you are using before continuing.")
    choice = input("Do you want to continue using CPU? (y/n): ").strip().lower()
    if choice == "y":
        device = torch.device("cpu")
        print("Continuing on CPU...")
    else:
        print("Exiting program. Please use a machine with GPU.")
        exit()  # stops the program

print("Using device:", device)

# instantiation of transforms, datasets and data loaders
# TIP : use torch.utils.data.random_split to split the training set into train and validation subsets

# class definitions

# instantiation and preparation of network model

# instantiation of loss criterion
# instantiation of optimizer, registration of network parameters

# definition of current best model path
# initialization of model selection metric

# creation of tensorboard SummaryWriter (optional)

# epochs loop:
#   train
#   validate on train set
#   validate on validation set
#   update graphs (optional)
#   is new model better than current model ?
#       save it, update current best metric

# close tensorboard SummaryWriter if created (optional)