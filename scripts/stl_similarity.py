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
import sys
import time
import meshio
import trimesh
import numpy as np

from lightcurve_fips.models.lightcurve_encoder import LightcurveSDFNet
from lightcurve_fips.data.lightcurves import load_lightcurve
from lightcurve_fips.training.utils import (reconstruct_sdf,sdf_to_stl)
from lightcurve_fips.evaluation.evaluate import (make_watertight_with_pymeshfix,compare_stl_files, relative_volume_difference_voxelized)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset2" / "test" / "test"
true_stl=DATASET_DIR / "asteroid731radius1.9181543359695385.stl"
recon_stl=DATASET_DIR / "asteroid731radius1.9181543359695385.stl_reconstruction.stl"
vox, side = compare_stl_files(
    true_stl,
    recon_stl,
    
    voxel_resolution=256,
    projection_resolution=128,
    per_axis_normalization=False
)
sample_folder=PROJECT_ROOT / "scripts"
true_stl=sorted(sample_folder.glob("asteroid*.stl"))[0]
recon_stl=sorted(sample_folder.glob("asteroid*.stl"))[1]

print(vox)

measure=relative_volume_difference_voxelized(true_stl, recon_stl, pitch = 0.05)

print(measure)