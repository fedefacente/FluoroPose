from pathlib import Path
from diffdrr.pose import matrix_to_euler_angles

import matplotlib.pyplot as plt
import torch
from diffdrr.pose import RigidTransform
from huggingface_hub import snapshot_download
from diffdrr.registration import PoseRegressor

from xvr.metrics import Evaluator
from opt_utils import RegistrarModelTAVI
# from xvr.registrar import RegistrarModel, RegistrarFixed
from diffdrr.drr import DRR

from xvr.utils import XrayTransforms
from xvr.visualization import plot_registration
import time
import tqdm
import glob
import numpy as np
import diffdrr
import torch.nn as nn
import pandas as pd
import os
from utils import sobel
from torchvision.transforms import Resize,Compose, Lambda
from pydicom import dcmread
from torchvision.transforms.functional import center_crop

subjects =['1e7dd220']
pred_poses = []
true_poses = []
alpha = []
beta = []
gamma = []
x = []
y = []
z = []
time_opt = []
mpd = []
mtre = []
for SUBJECT_ID in subjects:

    dest_path = os.path.join("/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/", f"results_{SUBJECT_ID}_opt")

    os.makedirs(dest_path, exist_ok=True)
    mean_absolute_error = nn.L1Loss(reduction='none')
    transform = Compose(
        [
            Resize((256, 256), antialias=True),
            Lambda(lambda x: (((x - x.min()) / (x.max() - x.min() + 1e-6)))),
            # Normalize(mean=0.3080, std=0.1494),
        ]
    )
    # Download the dataset

    datapath = Path(f'/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{SUBJECT_ID:02d}/ct')
    volume = datapath / "1e7dd220_volume.nii.gz"
    mask_path = datapath / "1e7dd220_vertebrae.nii.gz"

    # Download a pretrained model
    ckptpath = f"/lustre/fswork/projects/rech/gfu/uur34ii/model/RigidReg/subject{SUBJECT_ID:02d}_1000_shadow/best1.pth"

    model = f"deepfluoro{SUBJECT_ID:02d}.pth"
    download_path = '/lustre/fswork/projects/rech/gfu/uur34ii/xvr-data/pelvis/patient_specific'
    ckptpath = Path(download_path) / model
    print(ckptpath)


    model = RegistrarModelTAVI(
        volume=volume,
        mask=mask_path,
        ckptpath=ckptpath,
        linearize=True,
        reverse_x_axis=True,
        scales="24,12,6",
        crop=100,
        labels = "1,2,3,4,7",
        init_only=False,
    )
    # fiducials = torch.load(datapath / "fiducials.pt", weights_only=False).cuda()

    xrays = datapath / "xrays"

    #dest_path = os.path.join("/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente//6_DOF_estimation",f"specimen_{SUBJECT_ID}")
    #os.makedirs(dest_path, exist_ok=True)
    key = 0

    for img in tqdm.tqdm(list(zip(sorted([f for f in xrays.glob("*.dcm") if not f.name.endswith("_max.dcm")])))):
        print(key)

        time1 = time.time()
        gt, _, drr, init_pose, final_pose, _ = model.run(img, beta=0.5)

        #final_pose = init_pose
        time_opt.append(time.time() - time1)

        drr.density = diffdrr.data.transform_hu_to_density(drr.subject.volume.data,
                                                           bone_attenuation_multiplier=torch.tensor(5,
                                                                                                    dtype=torch.float32)).to(
            'cuda').squeeze(0)

        ds = dcmread(img)
        print(ds)
        img = ds.pixel_array.astype(np.int32)
        img = torch.from_numpy(img).to(torch.float32)[None, None]
        img = img.max().log() - img.log()
        gt = center_crop(img, (1436, 1436))
        gt = transform(gt)
        pred_poses.append(final_pose.matrix)
        true_poses.append(true_pose.matrix)
        drr = DRR(
            drr.subject,
            sdd=1175,
            height=int(256),
            width =int(256),
            delx=(0.1940000057220459*1436/256),
            dely=(0.1940000057220459*1436/256),
            x0 = 0,
            y0 = 0,
            renderer="trilinear",
            reverse_x_axis=False,
        ).to('cuda')
        angles_pred = diffdrr.pose.matrix_to_euler_angles(final_pose.rotation, convention="ZXY") * (180.0 / torch.pi)
        print(angles_pred)
        angles_GT = diffdrr.pose.matrix_to_euler_angles(true_pose.rotation, convention="ZXY") * (180.0 / torch.pi)
        print(angles_GT)
        R_T = true_pose.rotation.transpose(1, 2)  # Shape: (1, 3, 3)
        gt_t = torch.einsum("bij, bj -> bi", R_T, true_pose.translation)

        R_T = final_pose.rotation.transpose(1, 2)  # Shape: (1, 3, 3)
        pred_t = torch.einsum("bij, bj -> bi", R_T, final_pose.translation)
        alpha.append(mean_absolute_error((angles_pred).detach().cpu()[:, 0], angles_GT.detach().cpu()[:, 0])[0].numpy())
        beta.append(mean_absolute_error((angles_pred).detach().cpu()[:, 1], angles_GT.detach().cpu()[:, 1])[0].numpy())
        gamma.append(mean_absolute_error((angles_pred).detach().cpu()[:, 2], angles_GT.detach().cpu()[:, 2])[0].numpy())
        x.append(mean_absolute_error((pred_t).detach().cpu()[:, 0], gt_t.detach().cpu()[:, 0])[0].numpy())
        y.append(mean_absolute_error((pred_t).detach().cpu()[:, 1], gt_t.detach().cpu()[:, 1])[0].numpy())
        z.append(mean_absolute_error((pred_t).detach().cpu()[:, 2], gt_t.detach().cpu()[:, 2])[0].numpy())


        points2d_pred = drr.perspective_projection(final_pose.cuda(), fiducials).cpu()
        landmarks2d = drr.perspective_projection(true_pose.cuda(), fiducials).cpu()

        mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= 256) & (points2d_pred[:, :, 1] >= 0) & (
                points2d_pred[:, :, 1] <= 256)
        mask2 = (landmarks2d[:, :, 0] >= 0) & (landmarks2d[:, :, 0] <= 256) & (landmarks2d[:, :, 1] >= 0) & (
                landmarks2d[:, :, 1] <= 256)

        mask = mask1 & mask2
        mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

        points2d_pred[~mask2d] = float('nan')
        landmarks2d[~mask2d] = float('nan')

        points2d_pred[~mask2d] = float('nan')
        landmarks2d[~mask2d] = float('nan')
        mpd_batch = ((landmarks2d - points2d_pred).norm(dim=-1).nanmean()) *(0.1940000057220459*1436/256)
        print(mpd_batch)
        mpd.append(mpd_batch.mean().detach().cpu().numpy())

        points3d = true_pose(fiducials.cuda())
        points3d_pred = final_pose(fiducials.cuda())
        mtre_batch = (points3d - points3d_pred).norm(dim=-1).mean(dim=-1)
        mtre.append(mtre_batch.mean().detach().cpu().numpy())

        # pred_img = drr(final_pose).cpu()
        # pred_img = transform(pred_img)
        # true_img = drr(true_pose).cpu()
        # gradient = sobel(pred_img.to('cpu'))
        # gradient_pred = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        # gradient = sobel(gt.to('cpu'))
        # gradient_img = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        # overlay1 = np.zeros((256, 256, 3))  # Shape: (510, 510, 3)
        # overlay1[:, :, 0] = gradient_img.detach().cpu().numpy()  # Red channel for edges from the first image
        # overlay1[:, :, 1] = gradient_pred.detach().cpu().numpy()  # Blue channel for edges from the second image
        # plt.figure(figsize=(6, 6))
        # plt.imshow(pred_img[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
        # plt.scatter(landmarks2d.squeeze()[:, 0].detach().cpu().numpy(),
        #             landmarks2d.squeeze()[:, 1].detach().cpu().numpy(),
        #             c='fuchsia', s=70, alpha=1, marker='o')
        # plt.scatter(points2d_pred.squeeze()[:, 0].detach().cpu().numpy(),
        #             points2d_pred.squeeze()[:, 1].detach().cpu().numpy(),
        #             c='lime', s=70, alpha=1, marker='x')
        # plt.axis('off')
        # plt.savefig(os.path.join(dest_path,  f"{key}_final_pose.png"),
        #             bbox_inches='tight', pad_inches=0)
        # plt.close()
        '''plt.figure(figsize=(12, 4))
        plt.subplot(141)
        plt.grid(False)
        plt.title("Fluoroscopy")
        plt.imshow(gt.squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
        plt.scatter(landmarks2d[mask][:, 0].detach().cpu().numpy(), landmarks2d[mask][:, 1].detach().cpu().numpy(), c='g', s=20,
                        alpha=0.7, marker='o')
        plt.scatter(points2d_pred[mask][:, 0].detach().cpu().numpy(), points2d_pred[mask][:, 1].detach().cpu().numpy(), c='r', s=8,
                        alpha=0.7, marker='x')
        plt.axis('off')

        plt.subplot(142)'''
        '''plt.grid(False)
        plt.imshow((gt.squeeze(0).squeeze(0).detach().cpu() - pred_img.squeeze(0).squeeze(0).detach().cpu()),
                   origin='upper',
                   cmap='RdBu_r', vmin=-1, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(dest_path, f"{key}.png"), bbox_inches='tight', pad_inches=0)
        plt.axis('off')
        plt.close()'''
        '''plt.subplot(143)
        plt.grid(False)
        plt.title("Overlay")
        plt.imshow(overlay1)
        plt.axis('off')

        plt.subplot(144)
        plt.grid(False)
        plt.imshow((gt.squeeze(0).squeeze(0).detach().cpu()- pred_img.squeeze(0).squeeze(0).detach().cpu()), origin='upper',
                cmap='RdBu_r', vmin=-1, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(dest_path, f"{key}.png"))
        print(os.path.join(dest_path, f"{key}.png"))
        plt.close()'''
        key += 1

pred_poses = RigidTransform(torch.concat(pred_poses)).cuda()
true_poses = RigidTransform(torch.concat(true_poses)).cuda()
evaluator = Evaluator(model.drr, fiducials)

metrics = evaluator(true_poses, pred_poses)
print('OLD METRICS')

print(np.mean(np.array(metrics)[:, 0]))
print(np.array(metrics)[:, 0].std())
print(np.mean(np.array(metrics)[:, 2]))
print(np.array(metrics)[:, 2].std())
print('TIME')
print(np.mean(np.array(time_opt)))
print(np.std(np.array(time_opt)))
print('METRICS')

print(np.mean(np.array(mpd)))
print(np.array(mpd).std())
print(np.mean(np.array(mtre)))
print(np.array(mtre).std())
print(np.mean(np.array(alpha)))
print(np.array(alpha).std())
print(np.mean(np.array(beta)))
print(np.array(beta).std())
print(np.mean(np.array(gamma)))
print(np.array(gamma).std())
print(np.mean(np.array(x)))
print(np.array(x).std())
print(np.mean(np.array(y)))
print(np.array(y).std())
print(np.mean(np.array(z)))
print(np.array(z).std())
df = pd.DataFrame(
    {
        "mPD": np.array(mpd),
        "mTRE": np.array(mtre),
        "rx": np.array(alpha),
        "ry": np.array(beta),
        "rz": np.array(gamma),
        "tx": np.array(x),
        "ty": np.array(y),
        "tz": np.array(z)
    }
)
df.head()
df.to_csv(os.path.join(dest_path,'results.csv'), index=False)