import matplotlib.pyplot as plt
import torch
import diffdrr.data
from diffdrr.drr import DRR
import numpy as np
import pandas as pd
import h5py
from diffdrr.pose import convert, axis_angle_to_matrix, matrix_to_axis_angle
import time
from torchvision.transforms import Compose, Lambda, Normalize, Resize
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d, DoubleGeodesicSE3
import cv2
from scipy.ndimage import binary_fill_holes, binary_dilation
from skimage.segmentation import find_boundaries

from models import RegressorDecoderHRNetNew, RegistrationHRNetNew, RegressorDecoderHRNet
from models import encoder_decoder
from models import regressor_attention, registration_attention
import torch.nn as nn
import os
from utils import warp_to_canonical_space, resample, square, sobel
from diffdrr.pose import RigidTransform
from pydicom import dcmread
import tqdm
from pathlib import Path
from opt_utils import RegistrarModelTAVI
import argparse
import csv
from scipy.ndimage import distance_transform_edt

def edge_assd(mag_pred, mag_real, pct=90, spacing=(1.,1.)):
    ep = mag_pred > np.percentile(mag_pred[mag_pred>0], pct)
    er = mag_real > np.percentile(mag_real[mag_real>0], pct)
    dt_r = distance_transform_edt(~er, sampling=spacing)
    dt_p = distance_transform_edt(~ep, sampling=spacing)
    assd = (dt_r[ep].mean() + dt_p[er].mean())/2
    hd95 = np.percentile(np.concatenate([dt_r[ep], dt_p[er]]), 95)
    return assd, hd95   # in mm if spacing is mm/pixel

opt = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
mean_absolute_error = nn.L1Loss(reduction='none')
ncc = MultiscaleNormalizedCrossCorrelation2d([None, 16], [0.5, 0.5]).to(device)
geodesic = DoubleGeodesicSE3(sdd=1125)
patients = ["0f7f6475","2bc9e42f","579b5d2f","c2cc4c5b","d27c9be3", "12cc710", "3f9c36fa", "752399da"]
summary_dir = "/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/summary"
all_metrics = {}
per_patient_stats = []
iter_records = []
for id in patients:
    try:
        mtre_tot = []
        mpd = []
        alpha = []
        beta = []
        gamma = []
        x = []
        y = []
        z = []
        domain = []
        ncc_gt = []
        ncc_fin = []
        assd_tot = []
        hd95_tot = []

        geodesic_distances = []
        parameterization = "axis_angle"
        convention = None

        datapath = Path(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/fluoro_new")
        posepath = Path(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/poses")

        dest_path = os.path.join(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/results_{id}")
        os.makedirs(dest_path, exist_ok=True)

        model_dict = torch.load(
            f"/lustre/fswork/projects/rech/gfu/uur34ii/model/tavi_new_patients_instance_norm/{id}_hu_to_mu_ratio_edges_sobel/model_1499.pth")
        model1 = regressor_attention()

        model2 = registration_attention()

        model1.load_state_dict(model_dict["model_state_dict1"])
        model1.to("cuda")
        model1.eval()

        model2.load_state_dict(model_dict["model_state_dict2"])
        model2.to("cuda")
        model2.eval()

        model_dict = torch.load(
            f"/lustre/fswork/projects/rech/gfu/uur34ii/model/tavi_bones_new_patients_instance_norm/{id}_hu_to_mu_ratio_edges/model_final.pth")
        model_bones = RegressorDecoderHRNetNew()

        model_bones.load_state_dict(model_dict["model_state_dict1"])
        model_bones.to("cuda")
        model_bones.eval()

        csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/{id}_50_toracic.csv"
        df = pd.read_csv(csv_file_path)
        fiducials = np.array(df.values)
        fiducials = torch.tensor([fiducials]).to(torch.float)
        subject = diffdrr.data.read(
            f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/volume.nii.gz",
            labelmap=f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/aorta_leaflets.nii.gz",
            orientation="PA", bone_attenuation_multiplier=1, fiducials=fiducials)

        subject2 = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/volume.nii.gz",
                                     labelmap=f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/vertebrae.nii.gz",
                                     labels=1, orientation="PA", bone_attenuation_multiplier=1, fiducials=fiducials)
        iso = pd.read_csv(
            f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/isocenter.csv")
        iso = iso[["T8x", "T8y", "T8z"]].to_numpy()

        t_iso = np.array([((iso[0][0] - subject.shape[1] / 2) * subject.spacing[0]),
                          ((iso[0][1] - subject.shape[2] / 2) * subject.spacing[1]),
                          ((iso[0][2] - subject.shape[3] / 2) * subject.spacing[2])], dtype=np.float32)

        with (torch.no_grad()):
            for pose_file in tqdm.tqdm(posepath.glob("*_pose.pt")):
                img_name = pose_file.stem.replace("_pose", "") + ".dcm"
                img_path = datapath / img_name
                print(img_path)
                pose = torch.load(pose_file)
                ds = dcmread(img_path)
                width, height = 256, 256
                fps = 10
                for i, frame in enumerate(ds.pixel_array):
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                    print(i)
                    img = frame.astype(np.int32)
                    img = torch.from_numpy(img).to(torch.float32)[None, None]

                    # Get intrinsic parameters of the imaging system
                    sdd = ds.DistanceSourceToDetector
                    try:
                        dely, delx = ds.PixelSpacing

                    except AttributeError:
                        try:
                            dely, delx = ds.ImagerPixelSpacing
                            print(dely, delx)
                        except AttributeError:
                            dely, delx = 0.390625, 0.390625
                            print('default')

                    try:
                        y0, x0 = ds.DetectorActiveOrigin

                    except AttributeError:
                        y0, x0 = 0.0, 0.0
                    height = ds.Rows  # Rows
                    width = ds.Columns  # Columns
                    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

                    if height != width:
                        img = square(img, height, width)

                    target_size = max(height, width)
                    p = delx
                    p_target = 0.4 * 512 / 256
                    p = p * target_size / 256
                    height = int(256)
                    width = int(256)
                    sid = -820.0

                    transforms = Compose(
                        [
                            Resize((int(height), int(width)), antialias=True),
                            Lambda(lambda x: (((x - x.min()) / (x.max() - x.min() + 1e-6)))),
                        ]
                    )

                    img = transforms(img).to(device)
                    img += 1
                    img = img.max().log() - img.log()
                    img = transforms(img).to(device)
                    drr = DRR(
                        subject,
                        sdd=sdd,
                        height=height,
                        width=width,
                        delx=p,
                        x0=-x0,
                        y0=y0,
                        renderer="trilinear",
                        reverse_x_axis=True
                    ).to(device)
                    K_real = drr.detector.intrinsic

                    drr2 = DRR(
                        subject2,
                        sdd=1125,
                        height=256,
                        delx=(0.4 * 512 / 256),
                        renderer="trilinear",
                        reverse_x_axis=True
                    ).to(device)

                    drr = DRR(
                        subject,
                        sdd=1125,
                        height=256,
                        delx=(0.4 * 512 / 256),
                        renderer="trilinear",
                        reverse_x_axis=True
                    ).to(device)

                    opt_model = RegistrarModelTAVI(
                        drr,
                        init_only=False,
                        parameterization='euler_angles',
                        convention="ZXY",
                    )

                    K_can = drr.detector.intrinsic
                    img = warp_to_canonical_space(img, K_real.unsqueeze(0), K_can.unsqueeze(0))
                    gt_pose = drr(pose)
                    gt_pose = transforms(gt_pose)
                    img = transforms(img)

                    time_i = time.time()
                    _, _, bones = model_bones(img)
                    bones_sobel = sobel(bones)

                    _, _, bones_from_drr = model_bones(gt_pose)

                    r, t = model1(bones_sobel)

                    T = t + torch.tensor([np.float32(t_iso[0]), np.float32((sid)), np.float32(t_iso[2])],
                                         dtype=t.dtype, device=t.device)
                    pose_pred = convert(r, T, parameterization=parameterization, convention=convention)
                    init_pose = drr2(r, T, parameterization=parameterization, convention=convention)
                    init_pose = transforms(init_pose.sum(dim=1, keepdim=True))
                    init_pose_sobel = sobel(init_pose)

                    bones_gt = drr2(pose)
                    bones_gt = transforms(bones_gt.sum(dim=1, keepdim=True))

                    input2 = torch.cat((bones_sobel.detach(), init_pose_sobel), dim=1)

                    mpd_iters_tot, mtre_iters_tot = [], []

                    for iter in range(1):
                        print(iter)
                        delta_r, delta_t = model2(input2)
                        delta_pose = convert(delta_r, delta_t, parameterization=parameterization, convention=convention)
                        pose_pred = pose_pred.compose(delta_pose)

                        init_pose = drr2(pose_pred)
                        init_pose = transforms(init_pose.sum(dim=1, keepdim=True))
                        init_pose_sobel = sobel(init_pose)
                        input2 = torch.cat((bones_sobel.detach(), init_pose_sobel), dim=1)

                        # --- metrics per iteration ---
                        points2d = drr.perspective_projection(pose, subject.fiducials.cuda())
                        points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
                        mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & \
                                (points2d_pred[:, :, 1] >= 0) & (points2d_pred[:, :, 1] <= height)
                        mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= width) & \
                                (points2d[:, :, 1] >= 0) & (points2d[:, :, 1] <= height)
                        mask = mask1 & mask2
                        mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)
                        mask3d = mask.unsqueeze(-1).expand_as(subject.fiducials.cuda())

                        p2, p2p = points2d.clone(), points2d_pred.clone()
                        p2[~mask2d] = float('nan')
                        p2p[~mask2d] = float('nan')
                        mpd_i = ((p2 - p2p).norm(dim=-1).nanmean()) * p_target

                        p3 = pose(subject.fiducials.cuda())
                        p3p = pose_pred(subject.fiducials.cuda())
                        p3[~mask3d] = float('nan')
                        p3p[~mask3d] = float('nan')
                        mtre_i = (p3 - p3p).norm(dim=-1).nanmean()

                        mpd_iters_tot.append(float(mpd_i.detach().cpu()))
                        mtre_iters_tot.append(float(mtre_i.detach().cpu()))

                    for it, (m, t) in enumerate(zip(mpd_iters_tot, mtre_iters_tot)):
                        iter_records.append({"patient": id, "iter": it, "mpd": m, "mtre": t})

                    if opt == True:
                        with torch.enable_grad():
                            gt, _, _, _, pose_pred = opt_model.run(init_pose=pose_pred, gt=img, beta=0.5)
                    bones_pred = init_pose
                    R_p = pose_pred.rotation.transpose(1, 2)
                    T = torch.einsum("bij, bj -> bi", R_p, pose_pred.translation)

                    time_f = time.time()

                    points2d = drr.perspective_projection(pose, subject.fiducials.cuda())
                    points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
                    mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & (
                                points2d_pred[:, :, 1] >= 0) & (
                                    points2d_pred[:, :, 1] <= height)
                    mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= width) & (points2d[:, :, 1] >= 0) & (
                            points2d[:, :, 1] <= height)

                    mask = mask1 & mask2

                    mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)
                    mask3d = mask.unsqueeze(-1).expand_as(subject.fiducials.cuda())

                    points2d_pred[~mask2d] = float('nan')
                    points2d[~mask2d] = float('nan')
                    mpd2 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * p_target
                    print(mpd2.cpu().detach().numpy().round(2))
                    mpd.append(mpd2.nanmean().detach().cpu().numpy())
                    points3d = pose(subject.fiducials.cuda())
                    points3d_pred = pose_pred(subject.fiducials.cuda())
                    points3d_pred[~mask3d] = float('nan')
                    points3d[~mask3d] = float('nan')
                    mtre = (points3d - points3d_pred).norm(dim=-1).nanmean(dim=-1)
                    print(mtre.squeeze().detach().cpu().numpy().round(2))
                    mtre_tot.append(mtre.nanmean().detach().cpu().numpy())

                    _, _, double = geodesic(pose, pose_pred)

                    final_pose = drr(pose_pred, mask_to_channels=True)  # (1, C, H, W)
                    fp = final_pose[0, 1:].detach().cpu().numpy()  # (C, H, W)
                    CHANNEL_COLORS = {
                        0: np.array([0.0, 0.0, 1.0]),  # aorta - blue
                        1: np.array([0.0, 1.0, 0.0]),  # ncc   - green
                        2: np.array([0.0, 0.5, 1.0]),  # lcc   - light blue
                        3: np.array([1.0, 1.0, 0.0]),  # rcc   - yellow
                    }

                    rgb_fluoro = np.stack([1 - img[0].squeeze(0).squeeze(0).detach().cpu()] * 3, axis=-1)

                    cusps = np.zeros_like(rgb_fluoro)  # (H, W, 3)
                    cusp_mask = np.zeros(rgb_fluoro.shape[:2], dtype=bool)

                    for ch, color in CHANNEL_COLORS.items():
                        if ch >= fp.shape[0]:
                            continue
                        mask = fp[ch] > 0
                        if not mask.any():
                            continue

                        if ch == 0:  # aorta: fill + contour, drawn directly
                            alpha_aorta = 0.25
                            rgb_fluoro[mask] = (1 - alpha_aorta) * rgb_fluoro[mask] + alpha_aorta * color
                            mask = binary_fill_holes(mask)
                            edge = find_boundaries(mask, mode='inner')
                            rgb_fluoro[edge] = color
                        else:  # cusps: fill only, accumulate
                            cusps[mask] = color  # last cusp wins on overlap
                            cusp_mask |= mask

                    alpha_cusp = 0.45
                    rgb_fluoro[cusp_mask] = (1 - alpha_cusp) * rgb_fluoro[cusp_mask] + alpha_cusp * cusps[cusp_mask]

                    rgb_fluoro = np.clip(rgb_fluoro, 0.0, 1.0)
                    frame_ready_for_video = cv2.cvtColor(
                        (np.clip(rgb_fluoro, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR
                    )

                    rgb_bones = np.zeros((*init_pose.squeeze(0).squeeze(0).shape, 3), dtype=np.float32)
                    bones = transforms(bones)
                    rgb_bones[:, :, 0] = bones.squeeze(0).squeeze(0).detach().cpu() * 0.8
                    rgb_bones[:, :, 1] = init_pose.squeeze(0).squeeze(0).detach().cpu()
                    assd_i, hd95_i = edge_assd(bones_sobel.squeeze(0).squeeze(0).detach().cpu().numpy(),init_pose_sobel.squeeze(0).squeeze(0).detach().cpu().numpy(), spacing=(p_target,p_target))
                    assd_tot.append(assd_i)
                    hd95_tot.append(hd95_i)
                    print(img.shape)
                    plt.figure(figsize=(12, 4))
                    plt.subplot(151)
                    plt.grid(False)
                    plt.title("Fluoroscopy")
                    plt.imshow(1 - img[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
                    plt.axis('off')

                    plt.subplot(152)
                    plt.grid(False)
                    plt.title("bones pred")
                    plt.imshow(init_pose[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
                    plt.axis('off')

                    plt.subplot(153)
                    plt.grid(False)
                    plt.title("bones from fluoro")
                    plt.imshow(bones[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
                    plt.axis('off')

                    plt.subplot(154)
                    plt.grid(False)
                    plt.title("Bones DiffMap")
                    plt.imshow(rgb_bones)
                    plt.axis('off')

                    plt.subplot(155)
                    plt.grid(False)
                    plt.title("Aorta projection")
                    plt.imshow(rgb_fluoro)
                    plt.axis('off')

                    plt.savefig(os.path.join(dest_path, img_name + f"_{i}.png"))
                    plt.close()

                    del final_pose, init_pose, img, input2, bones, bones_gt
                    #### Parameters error
                    M = pose_pred.rotation
                    angles_pred = diffdrr.pose.matrix_to_euler_angles(M, convention="ZXY") * (180.0 / torch.pi)
                    angles_GT = diffdrr.pose.matrix_to_euler_angles(pose.rotation, convention="ZXY") * (180.0 / torch.pi)
                    R_T = pose.rotation.transpose(1, 2)
                    R_p = pose_pred.rotation.transpose(1, 2)
                    T = torch.einsum("bij, bj -> bi", R_p, pose_pred.translation)
                    print(diffdrr.pose.matrix_to_euler_angles(pose_pred.rotation, convention="ZXY"))
                    print(T)
                    gt_t = torch.einsum("bij, bj -> bi", R_T, pose.translation)
                    print(diffdrr.pose.matrix_to_euler_angles(pose.rotation, convention="ZXY"))
                    print(gt_t)
                    alpha.append(
                        mean_absolute_error((angles_pred).detach().cpu()[:, 0], angles_GT.detach().cpu()[:, 0]).tolist())
                    print(mean_absolute_error((angles_pred).detach().cpu()[:, 0], angles_GT.detach().cpu()[:, 0]).tolist())
                    beta.append(
                        mean_absolute_error((angles_pred).detach().cpu()[:, 1], angles_GT.detach().cpu()[:, 1]).tolist())
                    print(mean_absolute_error((angles_pred).detach().cpu()[:, 1], angles_GT.detach().cpu()[:, 1]).tolist())
                    gamma.append(
                        mean_absolute_error((angles_pred).detach().cpu()[:, 2], angles_GT.detach().cpu()[:, 2]).tolist())
                    x.append(mean_absolute_error(T.detach().cpu()[:, 0], gt_t.detach().cpu()[:, 0]).tolist())
                    y.append(mean_absolute_error(T.detach().cpu()[:, 1], gt_t.detach().cpu()[:, 1]).tolist())
                    z.append(mean_absolute_error(T.detach().cpu()[:, 2], gt_t.detach().cpu()[:, 2]).tolist())
                    geodesic_distances.append(double.detach().cpu())

        alpha = np.concatenate(alpha)
        beta = np.concatenate(beta)
        gamma = np.concatenate(gamma)
        x = np.concatenate(x)
        y = np.concatenate(y)
        z = np.concatenate(z)
        geodesic_distances = np.concatenate(geodesic_distances)
        assd_tot = np.concatenate(assd_tot)
        hd95_tot = np.concatenate(hd95_tot)
        metrics = {
            "mPD": np.array(mpd),
            "mTRE": np.array(mtre_tot),
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "x": x, "y": y, "z": z,
            "geodesic_distance": geodesic_distances,
            "assd": assd_tot,"hd95_tot": hd95_tot,
        }
        all_metrics[id] = metrics

        # per-patient stats
        row = {"patient": id}
        for name, arr in metrics.items():
            row[f"{name}_mean"] = np.nanmean(arr)
            row[f"{name}_median"] = np.nanmedian(arr)
            row[f"{name}_std"] = np.nanstd(arr)
            row[f"{name}_iqr"] = np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)
        per_patient_stats.append(row)

        # save raw arrays + stats for this patient
        np.savez(os.path.join(dest_path, f"{id}_metrics.npz"), **metrics)
        pd.DataFrame([row]).to_csv(os.path.join(dest_path, f"{id}_stats.csv"), index=False)

    except Exception as e:
        print(f"[skip] {id}: {e}")
        continue

# ---- after the patient loop: aggregate ----

os.makedirs(summary_dir, exist_ok=True)

per_patient_df = pd.DataFrame(per_patient_stats)
per_patient_df.to_csv(os.path.join(summary_dir, "per_patient_stats.csv"), index=False)

# pooled across all frames of all patients
if not all_metrics:
    raise RuntimeError("all_metrics is empty - every patient was skipped. Check the [skip] messages above.")

first_key = list(all_metrics.keys())[0]
metric_names = list(all_metrics[first_key].keys())
pooled = {name: np.concatenate([all_metrics[p][name] for p in all_metrics])
          for name in metric_names}

pooled_rows = []
for name, arr in pooled.items():
    pooled_rows.append({
        "metric": name,
        "mean": np.nanmean(arr),
        "median": np.nanmedian(arr),
        "std": np.nanstd(arr),
        "iqr": np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25),
        "n": np.sum(~np.isnan(arr)),
    })
pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv(os.path.join(summary_dir, "pooled_stats.csv"), index=False)

# mean-of-per-patient-medians (equal weight per patient, not per frame)
patient_median_cols = [c for c in per_patient_df.columns if c.endswith("_median")]
across_patients = per_patient_df[patient_median_cols].agg(["mean", "median", "std"])
across_patients.to_csv(os.path.join(summary_dir, "across_patient_stats.csv"))
iter_df = pd.DataFrame(iter_records)

# step 1: one value per (patient, iter) — median over that patient's frames
per_patient_iter = (iter_df.groupby(["patient", "iter"])[["mpd", "mtre"]]
                    .median().reset_index())

# step 2: median (and spread) across patients, per iter
iter_summary = per_patient_iter.groupby("iter").agg(
    mpd_median=("mpd", "median"),
    mpd_std=("mpd", "std"),
    mtre_median=("mtre", "median"),
    mtre_std=("mtre", "std"),
)
iter_summary.to_csv(os.path.join(summary_dir, "iter_summary.csv"))
print(pooled_df.to_string(index=False))
print(across_patients.to_string())