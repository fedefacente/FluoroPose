import streamlit as st
import torch
import diffdrr.data
from diffdrr.drr import DRR
import numpy as np
import time
from torchvision.transforms.functional import center_crop
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from diffdrr.pose import RigidTransform, matrix_to_axis_angle, convert, euler_angles_to_matrix, matrix_to_euler_angles,axis_angle_to_matrix
import cv2
import h5py
import matplotlib.pyplot as plt
from pydicom import dcmread
import torch.nn.functional as F
import pandas as pd
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import filedialog
import gc
from pathlib import Path
import os
from model import RegressorDecoderHRNetNew, map_grad_mag,map_grad_orient,map_raw
 
root = tk.Tk()
root.withdraw()
def get_actual_path(title):
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(parent=root, title = title)
    return path
 
def get_actual_folder(title):
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(parent=root, title=title)
    return path
 
 
if "subject_path" not in st.session_state:
    st.session_state.subject_path = get_actual_folder('Select subject CT folder') 
 
if st.button('New subject'):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    st.session_state.subject_path = get_actual_folder('Select subject CT folder')
 
    if "drr_engine" in st.session_state:
        keys_to_keep = {"primary_correction", "secondary_correction", "delx", "dely", "subject_path"}
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
 
if st.sidebar.button('New view'):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if "drr_engine" in st.session_state:
        keys_to_keep = {"primary_correction", "secondary_correction", "delx", "dely", "subject_path"}
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                print(key)
                del st.session_state[key]
 
 
if "drr_engine" not in st.session_state:
    torch.cuda.empty_cache()
    print("🚀 LOADING DATA AND INITIALIZING DRR")
    transform = Compose(
            [
                Resize((256, 256), antialias=True),
                Lambda(lambda x: (((x - x.min()) / (x.max() - x.min()+ 1e-6)))),
                #Normalize(mean=0.3080, std=0.1494),
            ]
        )
 
    
    subject = diffdrr.data.read(
    os.path.join(st.session_state.subject_path,'volume.nii.gz'), labelmap=os.path.join(st.session_state.subject_path,'segm.nii.gz'),
    orientation="PA", bone_attenuation_multiplier=2)
 
    st.session_state.device = torch.device("cuda")
    device = torch.device("cuda")
    path_dicom = get_actual_path('Select one view')
    ds = dcmread(path_dicom)
    st.session_state.name = Path(path_dicom).stem
    st.session_state.rx_dicom = ds.PositionerPrimaryAngle
    st.session_state.ry_dicom = ds.PositionerSecondaryAngle
 
    img = ds.pixel_array[5].astype(np.int32)
    img = torch.from_numpy(img).to(torch.float32)[None, None]
 
    # Get intrinsic parameters of the imaging system
    sdd = ds.DistanceSourceToDetector
    print(sdd)
    try:
        dely, delx = ds.PixelSpacing
        st.session_state.delx,st.session_state.dely = delx, dely 
 
    except AttributeError:
        try:
            dely, delx = ds.ImagerPixelSpacing
            st.session_state.delx,st.session_state.dely = delx, dely
        except AttributeError:
            delx, dely = st.session_state.delx,st.session_state.dely
    try:
        y0, x0 = ds.DetectorActiveOrigin
 
    except AttributeError:
        y0, x0 = 0.0, 0.0
    print(delx)
    height = ds.Rows  # Rows
    width = ds.Columns  # Columns
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
 
 
    target_size = max(height, width)
    p = delx
    print(delx)
    p = p * target_size/256
    height = int(256)
    width = int(256)
    transforms = Compose(
        [
            Resize((int(height), int(width)), antialias=True),
            Lambda(lambda x: (((x - x.min()) / (x.max() - x.min() + 1e-6)))),
        ]
    )
    
    st.session_state.xray_principal = transforms(img).to(device)
 
    st.session_state.drr_engine = DRR(
    subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
    sdd=sdd,  # Source-to-detector distance (i.e., the C-arm's focal length)
    height=height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    width=width,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    delx=p,  # Pixel spacing (in mm)
    x0=0,
    y0=0,
    renderer="trilinear",
    reverse_x_axis=True
    ).to(device)
 
 
    subject = diffdrr.data.read(
    os.path.join(st.session_state.subject_path,'volume.nii.gz'), labelmap=os.path.join(st.session_state.subject_path,'vertebrae.nii.gz'), labels =1,
    orientation="PA", bone_attenuation_multiplier=30)
 
    st.session_state.drr2 = DRR(
    subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
    sdd=sdd,  # Source-to-detector distance (i.e., the C-arm's focal length)
    height=height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    width=width,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    delx=p,  # Pixel spacing (in mm)
    x0=0,
    y0=0,
    renderer="trilinear",
    reverse_x_axis=True
    ).to(device)
    

 
 
 
    print("✅")
 
 
 
drr = st.session_state.drr_engine
device = st.session_state.device
rx_dicom = np.radians(st.session_state.rx_dicom)
print(rx_dicom)
ry_dicom = np.radians(st.session_state.ry_dicom)
print(ry_dicom)
xray_principal = st.session_state.xray_principal
drr2 = st.session_state.drr2
 
 
# ---  SIDEBAR Virtual C-arm---
 
st.sidebar.title("Virtual C-arm")
rx = st.sidebar.number_input("LAO/RAO", -2.0, 2.0, rx_dicom, step=0.01, format="%.4f")
ry = st.sidebar.number_input("CAU/CRA", -2.0, 2.0, ry_dicom, step=0.01, format="%.4f")
rz = st.sidebar.number_input("In plane rotation", -2.0, 2.0, 0.0, step=0.01, format="%.4f")
 
tx = st.sidebar.number_input("Horizontal translation (mm)", -100.0, 100.0, 0.0, step=1.0)
ty = st.sidebar.number_input("Depth translation (mm)", -1200.0, -400.0, -800.0, step=10.0)
tz = st.sidebar.number_input("Vertical translation (mm)", -200.0, 200.0, 0.0, step=1.0)
 
 
st.sidebar.title("Anatomical references")
 
bones = st.sidebar.number_input("Vertebrae", 0.0, 1.0, 0.3, step=0.3)
trachea = st.sidebar.number_input("Trachea", 0.0, 1.0, 0.0, step=0.1)
rib = st.sidebar.number_input("Ribs", 0.0, 1.0, 0.0, step=0.1)
aorta = st.sidebar.number_input("Aorta", 0.0, 1.0, 0.0, step=0.1)
vertebrae = st.sidebar.number_input("Vertebrae segmentations", 0.0, 1.0, 0.0, step=0.1)
 
 
rot = torch.tensor([[rx, ry, rz]], dtype=torch.float32, device=device)
xyz = torch.tensor([[tx, ty, tz]], dtype=torch.float32, device=device) 
# drr.density = transform_hu_to_density(st.session_state.drr_engine.subject.volume.data,bone_attenuation_multiplier=torch.tensor(1,dtype=torch.float32)).to(device).squeeze(0)
 
time_i = time.time()
with torch.no_grad():
    img = drr(rot, xyz, parameterization='euler_angles', convention='ZXY', mask_to_channels=True)
    bones_drr = drr2(rot, xyz, parameterization='euler_angles', convention='ZXY')
 
 
fluoro = xray_principal
segm = img[:, 1:]
 
def norm_img(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
 
drr_1= 1- norm_img(img.sum(dim=1, keepdim=True)).squeeze().detach().cpu().numpy()
drr_1 = np.stack([drr_1] * 3, axis=-1)
fluoro_np = norm_img(fluoro).squeeze().detach().cpu().numpy()
bones_drr= norm_img(bones_drr).squeeze().detach().cpu().numpy()
bones_drr2d = bones_drr
# rgb = np.zeros((*bones_drr.shape, 3), dtype=np.float32)
# rgb[:, :, 0] = bones_drr  
bones_drr = np.stack([bones_drr] * 3, axis=-1) 
 
rgb_fluoro = np.stack([fluoro_np] * 3, axis=-1)
 
segm_np2 = segm.squeeze().detach().cpu().numpy()
 
### extract anatomies
bones_segm = segm_np2[2:20]
ribs = segm_np2[20:].sum(0)
trachea_segm = segm_np2[0:1].sum(0)
aorta_segm = segm_np2[1:2].sum(0)
 
### choose colors
cmap = plt.get_cmap('gist_ncar') 
colors = cmap(np.linspace(0, 1, 45))[:, :3]
np.random.seed(42) # Seed ensures the colors stay the same every time you run the app
np.random.shuffle(colors)
segm_color = np.array([1.0, 0.0, 0.0])  
segm_color2 = np.array([0.0, 1.0, 0.0])  
segm_color3 = np.array([0.0, 0.0, 1.0])  
 
### anatomies segm rgb
colored_segm2 = np.expand_dims(trachea_segm > 0, axis=-1) * segm_color2
ribs_segm = np.expand_dims(ribs > 0, axis=-1) * segm_color
aorta_segm = np.expand_dims(aorta_segm > 0, axis=-1) * segm_color3
colored_segm = np.zeros((256, 256, 3))
 
 
for i in range(bones_segm.shape[0]):
    bone = np.expand_dims(bones_segm[i] > 0, axis=-1) * colors[i]
    colored_segm[np.any(bone > 0, axis=-1)] = bone[np.any(bone > 0, axis=-1)]
 
if trachea > 0.01:
    colored_segm[np.any(colored_segm2 > 0, axis=-1)] = [0.0, 0.0, 0.0]
 
if aorta > 0.01:
    colored_segm[np.any(aorta_segm > 0, axis=-1)] = [0.0, 0.0, 0.0]
 
 
overlay1 =  rgb_fluoro + trachea * colored_segm2 + rib *ribs_segm + aorta * aorta_segm + bones_drr * bones + vertebrae * colored_segm
 
 
col1, col2,col3 = st.columns(3, gap="large")
 

 
fluoro_inv = 1.0 - rgb_fluoro[:, :, 0]
 
t  = np.percentile(fluoro_inv, 10)
hi = np.percentile(fluoro_inv, 99.5)          
fluoro_inv = np.clip((fluoro_inv - t) / (hi - t + 1e-8), 0, 1)
 
rgb = np.zeros((*bones_drr[:,:,0].shape, 3), dtype=np.float32)
rgb[:, :, 0] = bones_drr[:,:,0]  *1.5
rgb[:, :, 1] = fluoro_inv
# rgb[:, :, 2] = 1-rgb_fluoro[:,:,0]
 
with col1:
    st.image(overlay1, caption="Target fluoroscopy", clamp=True, width=250)
 
with col2:
    st.image(drr_1, caption="DRR manual pose", clamp=True, width=250)
 
with col3:
    st.image(rgb, caption="Difference map", clamp=True, width=250)
 
 
with col2:
 
    if st.button('Save pose',  type="primary"):
        pose = convert(rot, xyz, parameterization='euler_angles', convention='ZXY')
        print(pose.rotation)
        print(pose.translation)
        file_path = get_actual_folder('Select folder to save pose') + '/' + st.session_state.name + '_pose.pt' 
        print(file_path)
        torch.save(pose, file_path)
        st.success(f"Pose saved!")
