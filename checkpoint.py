import matplotlib.pyplot as plt
import torch
import diffdrr.data
from diffdrr.drr import DRR
import numpy as np
import pandas as pd
from diffdrr.pose import convert
from torchvision.transforms import Compose, Lambda, Resize
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d, DoubleGeodesicSE3

from models import RegressorDecoderHRNetNew
from models import regressor_attention, registration_attention
import torch.nn as nn
import os
from utils import warp_to_canonical_space, square, sobel
from pydicom import dcmread
import tqdm
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import gc
from hu_to_mu import AttenuationMapper, AttenuationMapperRatio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINTS = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999, 1099, 1199, 1299, 1399]
CKPT_TMPL = "/lustre/fswork/projects/rech/gfu/uur34ii/model/tavi_new_patients_instance_norm/{id}_hu_to_mu_ratio_edges_sobel/model_{ep}.pth"
BONES_TMPL = "/lustre/fswork/projects/rech/gfu/uur34ii/model/tavi_bones_1500/{id}_ratio_edges/model_{ep}.pth"
SUMMARY_DIR = "/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/summary"
os.makedirs(SUMMARY_DIR, exist_ok=True)

patients = ["1e7dd220", "f0d28531"]

parameterization = "axis_angle"
convention = None


# ---------------------------------------------------------------------------
# Per-patient evaluation
# ---------------------------------------------------------------------------
def run_patient(id, model1, model2, model_bones):
    """Run eval for one patient, return (mtre_list, mpd_list) per-frame."""
    datapath = Path(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/fluoro_new")
    posepath = Path(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/poses")

    csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/{id}_50_toracic.csv"
    df = pd.read_csv(csv_file_path)
    fiducials = torch.tensor([np.array(df.values)]).to(torch.float)

    subject = diffdrr.data.read(
        f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/volume.nii.gz",
        labelmap=f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/aorta_leaflets.nii.gz",
        orientation="PA", bone_attenuation_multiplier=1, fiducials=fiducials)
    subject2 = diffdrr.data.read(
        f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/volume.nii.gz",
        labelmap=f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/vertebrae.nii.gz",
        labels=1, orientation="PA", bone_attenuation_multiplier=1, fiducials=fiducials)

    iso = pd.read_csv(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/isocenter.csv")
    iso = iso[["T8x", "T8y", "T8z"]].to_numpy()
    t_iso = np.array([((iso[0][0] - subject.shape[1] / 2) * subject.spacing[0]),
                      ((iso[0][1] - subject.shape[2] / 2) * subject.spacing[1]),
                      ((iso[0][2] - subject.shape[3] / 2) * subject.spacing[2])], dtype=np.float32)

    # Canonical DRRs depend only on subject/subject2 -> build once per patient.
    drr2 = DRR(subject2, sdd=1125, height=256, delx=(0.4 * 512 / 256),
               renderer="trilinear", reverse_x_axis=True).to(device)
    drr_can = DRR(subject, sdd=1125, height=256, delx=(0.4 * 512 / 256),
                  renderer="trilinear", reverse_x_axis=True).to(device)

    volume_anatomy = drr_can.subject.volume.data.to(device)
    mapper = AttenuationMapperRatio().to('cuda').eval()
    dens = mapper.density(volume_anatomy.clone())

    K_can = drr_can.detector.intrinsic

    p_target = 0.4 * 512 / 256
    sid = -820.0
    height = width = 256

    transforms = Compose([
        Resize((height, width), antialias=True),
        Lambda(lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-6))),
    ])

    mtre_tot, mpd = [], []

    with torch.no_grad():
        for pose_file in tqdm.tqdm(posepath.glob("*_pose.pt"), desc=f"{id}"):
            batch_size = 1
            img_name = pose_file.stem.replace("_pose", "") + ".dcm"
            img_path = datapath / img_name
            pose = torch.load(pose_file)
            ds = dcmread(img_path)

            for i, frame in enumerate(ds.pixel_array):
                gc.collect()
                torch.cuda.empty_cache()

                img = torch.from_numpy(frame.astype(np.int32)).to(torch.float32)[None, None]

                sdd = ds.DistanceSourceToDetector
                try:
                    dely, delx = ds.PixelSpacing
                except AttributeError:
                    try:
                        dely, delx = ds.ImagerPixelSpacing
                    except AttributeError:
                        dely, delx = 0.390625, 0.390625
                try:
                    y0, x0 = ds.DetectorActiveOrigin
                except AttributeError:
                    y0, x0 = 0.0, 0.0

                h_raw = ds.Rows
                w_raw = ds.Columns
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                if h_raw != w_raw:
                    img = square(img, h_raw, w_raw)

                target_size = max(h_raw, w_raw)
                p = delx * target_size / 256

                img = transforms(img).to(device)
                img += 1
                img = img.max().log() - img.log()
                img = transforms(img).to(device)

                # Intrinsic-specific DRR (depends on sdd, delx, origin) -> per frame.
                drr = DRR(subject, sdd=sdd, height=height, width=width, delx=p,
                          x0=-x0, y0=y0, renderer="trilinear", reverse_x_axis=True).to(device)
                K_real = drr.detector.intrinsic

                img = warp_to_canonical_space(img, K_real.unsqueeze(0), K_can.unsqueeze(0))

                from data_loader import InstrumentDataset
                from torch.utils.data import Dataset, DataLoader

                dataset = InstrumentDataset(
                    f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/instruments_volume")

                loader = DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=4,
                    pin_memory=True,
                    shuffle=True,
                    prefetch_factor=2
                )

                instrument_iter = iter(loader)

                rx = torch.normal(mean=0.0, std=0.4, size=(batch_size, 1)).to(device)
                ry = torch.normal(mean=0.0, std=0.2, size=(batch_size, 1)).to(device)
                rz = torch.normal(mean=0.0, std=0.4, size=(batch_size, 1)).to(device)
                rot = torch.cat([rx, ry, rz], dim=1).to(device)

                tx = torch.normal(mean=0, std=30, size=(batch_size, 1)).to(device)
                ty = torch.normal(mean=0, std=75, size=(batch_size, 1)).to(device)
                tz = torch.normal(mean=0, std=30, size=(batch_size, 1)).to(device)

                xyz = torch.cat([tx, ty, tz], dim=1).to(device)
                xyz = xyz + torch.tensor([np.float32(t_iso[0]), np.float32(sid), np.float32(t_iso[2])], dtype=tx.dtype,
                                         device=tx.device)
                with torch.no_grad():
                    try:
                        instruments = next(instrument_iter)
                    except StopIteration:
                        # If we ran through all files, automatically reshuffle and restart
                        instrument_iter = iter(loader)
                        instruments = next(instrument_iter)

                    instruments = instruments.to(device, non_blocking=True).float().squeeze(0)
                    mu = mapper(volume_anatomy.clone(), instruments, dens=dens)
                    drr_can.density = mu.to(drr_can.density)
                img = drr_can(pose)
                img = transforms(img)


                _, _, bones = model_bones(img)
                bones_sobel = sobel(bones)

                r, t = model1(bones_sobel)
                T = t + torch.tensor([np.float32(t_iso[0]), np.float32(sid), np.float32(t_iso[2])],
                                     dtype=t.dtype, device=t.device)
                pose_pred = convert(r, T, parameterization=parameterization, convention=convention)
                init_pose = drr2(r, T, parameterization=parameterization, convention=convention)
                init_pose = transforms(init_pose.sum(dim=1, keepdim=True))
                init_pose_sobel = sobel(init_pose)

                input2 = torch.cat((bones_sobel.detach(), init_pose_sobel), dim=1)
                for _ in range(1):
                    delta_r, delta_t = model2(input2)
                    delta_pose = convert(delta_r, delta_t, parameterization=parameterization, convention=convention)
                    pose_pred = pose_pred.compose(delta_pose)
                    init_pose = drr2(pose_pred)
                    init_pose = transforms(init_pose.sum(dim=1, keepdim=True))
                    init_pose_sobel = sobel(init_pose)
                    input2 = torch.cat((bones_sobel.detach(), init_pose_sobel), dim=1)

                # ---- metrics ----
                fid = subject.fiducials.cuda()
                points2d = drr.perspective_projection(pose, fid)
                points2d_pred = drr.perspective_projection(pose_pred, fid)
                mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & \
                        (points2d_pred[:, :, 1] >= 0) & (points2d_pred[:, :, 1] <= height)
                mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= width) & \
                        (points2d[:, :, 1] >= 0) & (points2d[:, :, 1] <= height)
                mask = mask1 & mask2
                mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)
                mask3d = mask.unsqueeze(-1).expand_as(fid)

                points2d_pred[~mask2d] = float('nan')
                points2d[~mask2d] = float('nan')
                mpd2 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * p_target
                mpd.append(mpd2.nanmean().detach().cpu().numpy())

                points3d = pose(fid)
                points3d_pred = pose_pred(fid)
                points3d_pred[~mask3d] = float('nan')
                points3d[~mask3d] = float('nan')
                mtre = (points3d - points3d_pred).norm(dim=-1).nanmean(dim=-1)
                mtre_tot.append(mtre.nanmean().detach().cpu().numpy())

                del img, input2, init_pose, bones, drr

    return mtre_tot, mpd


# ---------------------------------------------------------------------------
# Checkpoint-level evaluation
# ---------------------------------------------------------------------------
def evaluate_checkpoint(epoch):
    """Return (pooled_mtre, pooled_mpd) across all frames, all patients."""
    pooled_mtre, pooled_mpd = [], []
    for id in patients:
        try:
            ckpt = CKPT_TMPL.format(id=id, ep=1499)
            if not os.path.exists(ckpt):
                print(f"[skip] {id} ep{epoch}: no checkpoint")
                continue
            model_dict = torch.load(ckpt)
            model1 = regressor_attention()
            model2 = registration_attention()
            model1.load_state_dict(model_dict["model_state_dict1"]); model1.to(device).eval()
            model2.load_state_dict(model_dict["model_state_dict2"]); model2.to(device).eval()

            bones_dict = torch.load(BONES_TMPL.format(id=id, ep=epoch))
            model_bones = RegressorDecoderHRNetNew()
            model_bones.load_state_dict(bones_dict["model_state_dict1"]); model_bones.to(device).eval()

            mtre_id, mpd_id = run_patient(id, model1, model2, model_bones)
            pooled_mtre.extend(mtre_id)
            pooled_mpd.extend(mpd_id)

            del model1, model2, model_bones
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"[skip] {id} ep{epoch}: {e}")
            continue
    return np.array(pooled_mtre), np.array(pooled_mpd)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
epochs_done = []
stats = {"mtre_mean": [], "mtre_median": [], "mpd_mean": [], "mpd_median": []}

for ep in CHECKPOINTS:
    mtre, mpd = evaluate_checkpoint(ep)
    if mtre.size == 0:
        print(f"[skip] ep{ep}: no data")
        continue
    epochs_done.append(ep)
    stats["mtre_mean"].append(np.nanmean(mtre))
    stats["mtre_median"].append(np.nanmedian(mtre))
    stats["mpd_mean"].append(np.nanmean(mpd))
    stats["mpd_median"].append(np.nanmedian(mpd))
    print(f"ep {ep}: mTRE {stats['mtre_mean'][-1]:.2f}/{stats['mtre_median'][-1]:.2f}  "
          f"mPD {stats['mpd_mean'][-1]:.2f}/{stats['mpd_median'][-1]:.2f} (mean/median)")

epochs_done = np.array(epochs_done, dtype=float)
stats = {k: np.array(v) for k, v in stats.items()}

# save raw sweep numbers
pd.DataFrame({"epoch": epochs_done, **stats}).to_csv(
    os.path.join(SUMMARY_DIR, "sweep_stats.csv"), index=False)

# ---------------------------------------------------------------------------
# Plot: one figure per metric
# ---------------------------------------------------------------------------
xs = np.linspace(epochs_done.min(), epochs_done.max(), 400)


def smooth(y):
    return PchipInterpolator(epochs_done, y)(xs)


def plot_metric(metric, unit, fname):
    plt.figure(figsize=(7, 5))
    plt.plot(xs, smooth(stats[f"{metric}_mean"]), '-', label="mean (interp)")
    plt.scatter(epochs_done, stats[f"{metric}_mean"], zorder=5, label="mean")
    plt.plot(xs, smooth(stats[f"{metric}_median"]), '--', label="median (interp)")
    plt.scatter(epochs_done, stats[f"{metric}_median"], marker='s', zorder=5, label="median")
    plt.xlabel("Training epoch")
    plt.ylabel(f"Pooled {unit}")
    plt.title(f"{unit} vs. checkpoint")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, fname), dpi=150)
    plt.close()


plot_metric("mtre", "mTRE (mm)", "mtre_vs_epoch.png")
plot_metric("mpd", "mPD (mm)", "mpd_vs_epoch.png")

print("done ->", SUMMARY_DIR)