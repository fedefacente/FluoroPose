import matplotlib.pyplot as plt
import torch
import diffdrr.data
from diffdrr.drr import DRR
import numpy as np
import pandas as pd
import h5py
from torchvision.transforms import Compose, Lambda, Resize
from torchvision.transforms.functional import center_crop
import torch.nn as nn
import os
import cv2
import kornia
from kornia.augmentation import AugmentationSequential
import kornia.augmentation as K
from scipy.stats import beta
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from kornia.filters import box_blur
from utils import sobel
from simulation_instruments_TAVI import simulate_catheter
import math
from hu_to_mu import AttenuationMapper, AttenuationMapperRatio

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="0f7f6475")
args = parser.parse_args()
id = args.id

import torch.nn.functional as F

def generate_pytorch_catheter_batch(batch_size=6, img_size=256, num_samples=200, device='cuda'):
    """
    Generate a batch of masks with random Bézier curves.

    Returns:
        masks: [batch_size, img_size, img_size], float32, 0/1
    """

    # 1. Randomize 4 control points for each batch item
    p = torch.randint(0, img_size - 1, (batch_size, 4, 2), device=device).float()

    # 2. Cubic Bézier parameter t
    t = torch.linspace(0, 1, num_samples, device=device).view(1, num_samples, 1)  # [1, num_samples, 1]

    # 3. Compute Bézier points: [batch, num_samples, 2]
    curve_pts = (1 - t) ** 3 * p[:, 0:1, :] + \
                3 * (1 - t) ** 2 * t * p[:, 1:2, :] + \
                3 * (1 - t) * t ** 2 * p[:, 2:3, :] + \
                t ** 3 * p[:, 3:4, :]

    # 4. Convert to integer pixel indices
    curve_px = curve_pts.round().long().clamp(0, img_size - 1)  # [B, N, 2]
    B, N, _ = curve_px.shape

    # 5. Flatten batch and points for scatter
    batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, N).flatten()  # [B*N]
    x_idx = curve_px[:, :, 0].flatten()  # [B*N]
    y_idx = curve_px[:, :, 1].flatten()  # [B*N]

    # 6. Create flat mask and scatter
    masks = torch.zeros((B, img_size, img_size), device=device)
    masks[batch_idx, y_idx, x_idx] = 1

    # 7. Apply thickness using 2D convolution
    thickness = torch.randint(2,8,(1,1)).item()

    if thickness > 1:
        kernel = torch.ones((1, 1, thickness, thickness), device=device)
        masks = F.conv2d(masks.unsqueeze(1), kernel, padding="same").squeeze(1)
        masks = (masks > 0).float().squeeze(1)
        intensities = torch.rand(B, device=device)
        masks *= intensities

    return masks.to(device)



def transform_hu_to_density(volume):
    volume = volume.float()
    density = volume.clamp(min=-800)
    shift = torch.empty(1).uniform_(0, 600).to(density.device)
    print(shift)
    soft = torch.empty(1).uniform_(-200, 0).to(density.device)
    print(soft)
    density += 0.2 * ((density > 350).to(density.dtype) )

    density -= density.min()
    density /= density.max()
    #density = torch.sigmoid(alpha * (density - beta))
    #xa = torch.pow(density, alpha)
    #xb = torch.pow(1 - density, beta)
    #density = xa / (xa + xb)
    return density

def transform_labels_to_density(volume):
    volume = volume.float()
    out = torch.zeros_like(volume)
    out = torch.where(volume == 1, 10.0,  out)   # Safari wire (stiff, radiopaque)
    out = torch.where(volume == 2, 1.0,  out)   # delivery sheath (polymer)
    out = torch.where(volume == 3, 6.0, out)   # valve frame
    out -= out.min()
    out /= out.max()
    return out

norm_drr = Compose(
    [
        # Lambda(lambda x: (x.max() + x.min() - x)),
        # lambda x: (x - x.amin(dim=(2, 3), keepdim=True)) / (x.amax(dim=(2, 3), keepdim=True) - x.amin(dim=(2, 3), keepdim=True) + 1e-6)   ,
        Lambda(lambda x: (((x - x.min()) / (x.max() - x.min() + 1e-6)))),
        # Lambda(lambda x: (
        #    lambda q_low, q_high: (torch.clamp(x, q_low, q_high) - q_low) / (q_high - q_low + 1e-6)
        # )(*torch.quantile(x, torch.tensor([0.02, 0.98], device=x.device))))

    ]
)



transform = Compose(
    [
        # lambda x: (x - x.amin(dim=(2, 3), keepdim=True)) / (x.amax(dim=(2, 3), keepdim=True) - x.amin(dim=(2, 3), keepdim=True) + 1e-6)   ,
        Lambda(lambda x: (((x - (x.min())) / ((x.max())- x.min() + 1e-6)))),
        Resize((256, 256), antialias=True),
    ]
)
aug = AugmentationSequential(
    kornia.augmentation.RandomEqualize(p=0, keepdim=True, same_on_batch=False),
    kornia.augmentation.RandomSaltAndPepperNoise(amount=(0.005, 0.01), salt_vs_pepper=(0.4, 0.6), p=0,
                                                    same_on_batch=False, keepdim=True),
    kornia.augmentation.RandomGamma(gamma=(0.6, 1.8), gain=(1.0, 1.0), same_on_batch=False, p=0, keepdim=True),
    kornia.augmentation.RandomGaussianNoise(mean=0.0, std=0.03, same_on_batch=False, p=0, keepdim=True),
    kornia.augmentation.RandomLinearIllumination(p=0, keepdim=True, same_on_batch=False),
    kornia.augmentation.RandomSharpness(p=0, keepdim=True, same_on_batch=False),
    kornia.augmentation.RandomAutoContrast(p=0, keepdim=True, same_on_batch=False),
    kornia.augmentation.RandomBoxBlur(p=0, keepdim=True, same_on_batch=False),
    kornia.augmentation.RandomPlasmaShadow(roughness=(0.3, 0.7),shade_quantity = (0.2,0.4),shade_intensity =(-0.3,0.3), p=1.0, keepdim=True, same_on_batch=False),
    #RandomCenterCrop(p=1.0, maxcrop=10),
    data_keys=["input"],
    same_on_batch=None,
)
csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/{id}_50_vertebrae.csv"
df = pd.read_csv(csv_file_path)
fiducials = np.array(df.values)  # Extract as NumPy array
fiducials = torch.tensor([fiducials]).to(torch.float)
csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/points_centerline.csv"
df = pd.read_csv(csv_file_path)
centerline = np.array(df.values)  # Extract as NumPy array
centerline = torch.tensor([centerline]).to(torch.float)
subject_catheter = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/ventricle.nii.gz",
                            orientation="PA", bone_attenuation_multiplier=2, fiducials=centerline)
volume_original = torch.where(subject_catheter.volume.data ==2,0, subject_catheter.volume.data)
ventricle = torch.where(subject_catheter.volume.data >0,1, 0)
# volume_anatomy2 = subject_catheter.volume.data.to('cuda') *300
subject = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/volume.nii.gz",orientation="PA", bone_attenuation_multiplier=2, fiducials=fiducials)
volume_anatomy = subject.volume.data.to('cuda')
iso = pd.read_csv(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/isocenter.csv")
iso = iso[["T8x", "T8y", "T8z"]].to_numpy()

t_iso = np.array([((iso[0][0] - subject.shape[1] / 2) * subject.spacing[0]),
                  ((iso[0][1] - subject.shape[2] / 2) * subject.spacing[1]),
                  ((iso[0][2] - subject.shape[3] / 2) * subject.spacing[2])], dtype=np.float32)

# Initialize the DRR module for generating synthetic X-rays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
drr = DRR(
    subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
    sdd=1125,  # Source-to-detector distance (i.e., the C-arm's focal length)
    height=256,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    delx=(0.4 * 512 / 256),  # Pixel spacing (in mm)
    renderer="trilinear",
    reverse_x_axis=True
).to(device)

drr_catheter = DRR(
    subject_catheter,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
    sdd=1125,  # Source-to-detector distance (i.e., the C-arm's focal length)
    height=256,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    delx=(0.4* 512 / 256),
    renderer="trilinear",
    reverse_x_axis=True
).to(device)

sid = -820.0

dest_path = os.path.join("/lustre/fswork/projects/rech/gfu/uur34ii//data/TAVI/domain_adp")
os.makedirs(dest_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from data_loader import InstrumentDataset
from torch.utils.data import Dataset, DataLoader

dataset = InstrumentDataset(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/instruments_volume")

loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    prefetch_factor=2,
    persistent_workers=True
)

instrument_iter = iter(loader)

mapper = AttenuationMapperRatio().to('cuda').eval()
dens = mapper.density(volume_anatomy.clone())                         # once per CT

subject_bones = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/volume.nii.gz",f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/vertebrae.nii.gz",labels =1,orientation="PA", bone_attenuation_multiplier=30, fiducials=fiducials)

drr_bones = DRR(
    subject_bones,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
    sdd=1125,  # Source-to-detector distance (i.e., the C-arm's focal length)
    height=256,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    delx=(0.4 * 512 / 256),  # Pixel spacing (in mm)
    renderer="trilinear",
    reverse_x_axis=True
).to(device)
for i in range(50):
    print(i)

    # ct_volume = drr.subject.volume.data
    #
    # #contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
    # #contrast = contrast_distribution.sample().item()
    #
    # alpha = torch.empty(1).uniform_(0.5, 0.5).item()
    # beta = torch.empty(1).uniform_(0.5, 0.5).item()
    #
    # # time_i = time.time()
    # #
    # #ct_volume = global_sigmoid_contrast(ct_volume, strength=strength, center = center)
    #drr.density = transform_hu_to_density(ct_volume).to(device).squeeze(0)
    # print(time.time() - time_i)
    rot = torch.tensor([[np.float32(0), np.float32(0), np.float32(0.2)]], dtype=torch.float32,
                             device=device)
    xyz = torch.tensor([[np.float32(t_iso[0]), np.float32(sid), np.float32(t_iso[2]-20)]], dtype=torch.float32,
                             device=device)

    # pose = torch.load('/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/1e7dd220/poses/1e7dd220_XA_SER001_IMG000018_pose.pt')
    # Homogeneous coords: (N, 4)
    # B, N, _ = subject_catheter.fiducials.shape  # 1, 10, 3
    # ones = torch.ones(B, N, 1, dtype=subject_catheter.fiducials.dtype, device=subject_catheter.fiducials.device)
    # pts_h = torch.cat([subject_catheter.fiducials, ones], dim=-1)
    # inv_affine = torch.linalg.inv(torch.tensor(subject_catheter.volume.affine)).float()
    #
    # vox_points = pts_h @ inv_affine.T  # (1, 10, 4)
    # vox_points = vox_points[...,:3].squeeze(0)
    # P0 = vox_points[0, :]
    # P1 = vox_points[1, :]

    with torch.no_grad():

        # volume = simulate_catheter_v1(subject.volume.data.clone(), subject.fiducials.to(device),
        #                            torch.tensor(subject.volume.affine).to(device),
        #                            subject.spacing, drr.subject.volume.data.clone())
        #
        # volume = torch.where(volume>0, volume,drr.subject.volume.data)
        # drr.density = volume.to(device)
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:


                instruments = next(instrument_iter)

            except StopIteration:
                # If we ran through all files, automatically reshuffle and restart
                instrument_iter = iter(loader)
                instruments = next(instrument_iter)

            instruments = instruments.to(device, non_blocking=True).float().squeeze(0)
            # # volume = torch.where(instruments > 0, instruments, volume_anatomy)
            # drr_catheter.density = transform_labels_to_density(instruments).squeeze(0)
            # catheter = drr_catheter(rot, xyz, parameterization='axis_angle', convention=None)
            # catheter = transform(catheter.sum(dim=1, keepdim=True))

            mu = mapper(volume_anatomy.clone(),instruments)
            drr.density = mu.to(drr.density)

            # drr.density = hu_to_mu(volume_anatomy.clone(),instruments.clone())
            torch.cuda.synchronize()
            print("hu_to_mu:", (time.perf_counter() - t) / 50 * 1000, "ms")



    #volume = simulate_catheter(volume_original.clone(),subject_catheter.fiducials.to(device),torch.tensor(subject_catheter.volume.affine).to(device),subject_catheter.spacing, subject.volume.data.clone(), ventricle.clone())
    #drr_catheter.density = transform_hu_to_density2(volume).to(device).squeeze(0)
    # catheter_3d = drr_catheter.subject.volume.data.squeeze()
    # coords = torch.nonzero(catheter_3d)
    # shifted_coords = coords.float().to(device) + torch.tensor([[0, 0, 15]], device=device)
    # indices = torch.round(shifted_coords).long()
    # indices[:, 0] = torch.clamp(indices[:, 0], 0, catheter_3d.shape[0] - 1)  # Z
    # indices[:, 1] = torch.clamp(indices[:, 1], 0, catheter_3d.shape[1] - 1)  # Y
    # indices[:, 2] = torch.clamp(indices[:, 2], 0, catheter_3d.shape[2] - 1)  # X
    # vol = torch.zeros(catheter_3d.shape, device=device)
    # z_idx = indices[:, 2]
    # y_idx = indices[:, 1]
    # x_idx = indices[:, 0]
    # vol = torch.zeros(catheter_3d.shape, device=device)
    # vol[x_idx, y_idx, z_idx] = 1
    #
    # drr_catheter.density = vol.squeeze(0)


    #drr.density = transform_hu_to_density(torch.where(vol>0, vol,drr.subject.volume.data.to(device)) ,bone_attenuation_multiplier=torch.tensor(2,dtype=torch.float32)).to(device).squeeze(0)
    # k_size = 2 * 2 + 1
    # drr.mask = F.max_pool3d(catheter_3d.float(), kernel_size=k_size, stride=1, padding=2).squeeze(0).to(device)
    bones_gt = drr_bones(rot, xyz, parameterization='axis_angle', convention=None)
    bones_gt = transform(bones_gt.sum(dim=1, keepdim=True))

    img = drr(rot, xyz, parameterization='axis_angle', convention=None)
    img = transform(img.sum(dim=1, keepdim=True))
    # img = aug(img)
    # img = drr(rot, xyz, parameterization='axis_angle', convention=None)
    # img = transform(img.sum(dim=1, keepdim=True))
    # img = sobel(img.to('cpu')).to('cuda')
    # img = (img - img.min()) / (img.max() - img.min())
    # max = (torch.rand(1, device=device) * 0.3) + 0.7
    #
    # min = (torch.rand(1, device=device) * 0.3)
    # img = (img * (max - min)) + min
    #img = torch.clamp(img, 0.0, 1.0)
    # catheter = transform(catheter.sum(dim=1, keepdim=True))
    # catheter = torch.where(catheter > 0, 1, 0).float()
    # thickness = torch.randint(0,5,(1,1)).item()
    # if thickness > 1:
    #     kernel = torch.ones((1, 1, thickness, thickness), device=device)
    #     catheter = F.conv2d(catheter, kernel, padding="same").squeeze(1)
    #     catheter = (catheter > 0).float().squeeze(1)
    #     #catheter = box_blur(catheter.unsqueeze(0), (5, 5))
    # intensities = (torch.rand(1, device=device) * 0.5) + 0.5
    # catheter *= intensities
    # catheter = generate_pytorch_catheter_batch(batch_size=1).unsqueeze(0)
    # alpha = torch.rand(1, device=device).item()* 0.5
    # img = torch.where(catheter > 0, catheter *alpha + img , img)

    '''plt.figure()
    plt.grid(False)
    plt.imshow(input.squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
    plt.axis('off')'''
    plt.figure(figsize=(6, 6))
    plt.imshow(1-bones_gt[0].squeeze(0).detach().cpu(), cmap='gray', vmax =1, vmin = 0)

    plt.axis('off')


    '''plt.subplot(131)
    plt.grid(False)
    plt.title("DRR gt")
    plt.imshow(input.squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.grid(False)
    plt.title("X-Ray")
    plt.imshow(img.squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.grid(False)
    plt.title("Histograms")
    plt.hist(xray_flat, bins=256, alpha=0.5, label='X-ray', color='blue', density=True)
    plt.hist(drr_flat, bins=256, alpha=0.5, label='DRR', color='red', density=True)
    plt.axis('off')

    plt.legend()'''
    plt.savefig(os.path.join(dest_path, f"{i}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
