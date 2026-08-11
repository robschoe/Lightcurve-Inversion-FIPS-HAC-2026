import trimesh
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage import measure
import re
import os
import time
import torch.nn.functional as F

def load_lightcurve(csv_path):
    data = pd.read_csv(csv_path, header=None)

    values = data.values.astype(np.float32)

    # Falls erste Spalte Framezahl ist:
    values = values[:, 1:]

    values = values / (values.mean(axis=0, keepdims=True) + 1e-8)

    return values