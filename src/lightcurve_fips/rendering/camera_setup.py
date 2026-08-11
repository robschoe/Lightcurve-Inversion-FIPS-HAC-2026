import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import imageio
import random
import math
import alphashape
import trimesh
import torch
from scipy.signal import savgol_filter
from pathlib import Path
import os
import time
import bpy

def angle_to_mode(angle):
    match angle:
        case 0:
            return 21 #24
        case 45 | 135:
            return 26
        case 90:
            return 26 #29
        case 225 | 270:
            return 24
        case 315:
            return 24 #25

def create_camera(name, angle, mode):
    r=20
    anglexy=np.deg2rad(angle+180)
    anglez=np.deg2rad(90-angle_to_mode(angle)*mode)
    x=r*np.cos(anglexy)*np.sin(anglez)
    y=r*np.sin(anglexy)*np.sin(anglez)
    z=r*np.cos(anglez)
    bpy.ops.object.camera_add(location=(x,y,z))
    cam = bpy.context.object
    cam.name = name
    
    direction = - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()
    
    return cam
