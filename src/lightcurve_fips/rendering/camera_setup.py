import numpy as np
from scipy.signal import savgol_filter

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

    pos = np.array([x, y, z], dtype=np.float32)
    
    return pos
