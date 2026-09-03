import diffdrr.data
from diffdrr.drr import DRR
import numpy as np
import pandas as pd
import os
import torch
from simulation_instruments_TAVI import simulate_catheter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="0f7f6475")
args = parser.parse_args()
id = args.id
csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/points_centerline.csv"
df = pd.read_csv(csv_file_path)
centerline = np.array(df.values)  # Extract as NumPy array
centerline = torch.tensor([centerline]).to(torch.float)
subject_catheter = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/ventricle.nii.gz",
                            orientation="PA", bone_attenuation_multiplier=2, fiducials=centerline)
subject = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/aorta.nii.gz",orientation="PA", bone_attenuation_multiplier=2)

aorta = subject.volume.data
ventricle = torch.where(subject_catheter.volume.data >0,1, 0)
device = torch.device("cuda")
dest_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/instruments_volume"
os.makedirs(dest_path, exist_ok=True)
for i in range(1000):
    print(f"instrument_{i+1}_labels_tg.npy")
    path = os.path.join(dest_path,f"instrument_{i+1}_labels_tg")
    volume = simulate_catheter(aorta.clone(),subject_catheter.fiducials.to(device),torch.tensor(subject_catheter.volume.affine).to(device),subject_catheter.spacing, subject.volume.data.clone(), ventricle.clone())
    # volume = transform_hu_to_density(volume).squeeze(0).detach().cpu().float()
    # np.save(os.path.join(dest_path,f"instrument_{i+1}_labels_tg.npy"),volume.squeeze(0).detach().cpu().float().numpy())
    vol = volume.squeeze(0).detach().cpu()
    idx = torch.nonzero(vol)  # (K, 3) occupied voxels
    lab = vol[idx[:, 0], idx[:, 1], idx[:, 2]].to(torch.uint8)
    np.savez_compressed(path,
                        coords=idx.to(torch.int16).numpy(),  # int16 if dims < 32767
                        labels=lab.numpy(),
                        shape=np.array(vol.shape, np.int32))
    # volume_np = volume.squeeze(0).detach().cpu().numpy()
    # nii_img = nib.Nifti1Image(volume_np, affine=subject_catheter.volume.affine)
    # nib.save(nii_img, os.path.join(dest_path, f"nifti_instrument_{i + 1}.nii.gz"))