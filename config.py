import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
from skimage import io, transform
import numpy as np
import splitfolders
import math
import matplotlib.patches as patches
import itertools
from sklearn.metrics import auc
import pandas as pd
from IPython.display import display


TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'
NUM_WORKERS = 0
BATCH_SIZE = 4
MAX_EPOCHS = 15
IS_TRAIN = True
CLASSES_DICT = {
    'black-queen': 0,
    'black-knight': 1,
    'white-queen': 2,
    'white-bishop': 3,
    'white-king': 4,
    'black-king': 5,
    'white-knight': 6,
    'black-bishop': 7,
    'white-rook': 8,
    'black-rook': 9,
    'black-pawn': 10,
    'white-pawn': 11,
    'background': 12
}

