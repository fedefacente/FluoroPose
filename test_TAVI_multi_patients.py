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
from scipy.ndimage import distance_transform_edt

opt = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
mean_absolute_error = nn.L1Loss(reduction='none')
ncc = MultiscaleNormalizedCrossCorrelation2d([None, 16], [0.5, 0.5]).to(device)
geodesic = DoubleGeodesicSE3(sdd=1125)
patients = ["752399da","0f7f6475","12cc710","2bc9e42f","3f9c36fa","579b5d2f","c2cc4c5b","d27c9be3"]
summary_dir = "/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/summary_ablations"
os.makedirs(summary_dir, exist_ok=True)

all_metrics = {}
per_patient_stats = []
iter_records = []
dcm_records = []

torch.backends.cudnn.benchmark = True
for id in patients:

        geodesic_distances = []
        parameterization = "axis_angle"
        convention = None

        datapath = Path(f"/CTAVIX/{id}/fluoro_new")
        posepath = Path(f"/CTAVIX/{id}/poses")

        dest_path = os.path.join(f"CTAVIX/results_{id}")
        os.makedirs(dest_path, exist_ok=True)

        model_dict = torch.load(
            f"/cascade/{id}/model_1499.pth")
        model1 = regressor_attention()

        model2 = registration_attention()

        model1.load_state_dict(model_dict["model_state_dict1"])
        model1.to("cuda")
        model1.eval()

        model2.load_state_dict(model_dict["model_state_dict2"])
        model2.to("cuda")
        model2.eval()

        model_dict = torch.load(
            f"/BonesHRNet/{id}/model_final.pth")
        model_bones = RegressorDecoderHRNetNew()

        model_bones.load_state_dict(model_dict["model_state_dict1"])
        model_bones.to("cuda")
        model_bones.eval()
        # ===== ONCE, at setup (after model_bones is loaded and .eval()) =====
        model_bones.eval()
        torch.backends.cudnn.benchmark = True

        static_input = torch.zeros(1, 1, 256, 256, device="cuda")

        # warmup on side stream (required before capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                with torch.no_grad():
                    model_bones(static_input)
        torch.cuda.current_stream().wait_stream(s)

        bones_graph = torch.cuda.CUDAGraph()  # was: g
        with torch.cuda.graph(bones_graph):  # was: g
            with torch.no_grad():
                static_out = model_bones(static_input)


        def run_bones(img):
            static_input.copy_(img)
            bones_graph.replay()  # was: g.replay()
            return static_out



        csv_file_path = f"/CTAVIX/{id}/{id}_50_toracic.csv"
        df = pd.read_csv(csv_file_path)
        fiducials = np.array(df.values)
        fiducials = torch.tensor([fiducials]).to(torch.float)
        subject = diffdrr.data.read(
            f"/CTAVIX/{id}/ct/volume.nii.gz",
            labelmap=f"/CTAVIX/{id}/ct/aorta_leaflets.nii.gz",
            orientation="PA", bone_attenuation_multiplier=2, fiducials=fiducials)

        subject2 = diffdrr.data.read(f"/CTAVIX/{id}/ct/volume.nii.gz",
                                     labelmap=f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/vertebrae.nii.gz",
                                     labels=1, orientation="PA", bone_attenuation_multiplier=1, fiducials=fiducials)
        iso = pd.read_csv(
            f"/CTAVIX/{id}/isocenter.csv")
        iso = iso[["T8x", "T8y", "T8z"]].to_numpy()

        t_iso = np.array([((iso[0][0] - subject.shape[1] / 2) * subject.spacing[0]),
                          ((iso[0][1] - subject.shape[2] / 2) * subject.spacing[1]),
                          ((iso[0][2] - subject.shape[3] / 2) * subject.spacing[2])], dtype=np.float32)

        with (torch.no_grad()):
            for pose_file in tqdm.tqdm(posepath.glob("*_pose.pt")):
                # if pose_file.stem.replace("_pose", "") != "752399da_XA_SER001_IMG000015":
                #     continue
                img_name = pose_file.stem.replace("_pose", "") + ".dcm"
                img_path = datapath / img_name
                print(img_path)
                pose = torch.load(pose_file)
                ds = dcmread(img_path)
                width, height = 256, 256
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
                    img_plot = img
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

                    torch.cuda.synchronize()

                    time_i = time.time()

                    _, _, bones = run_bones(img)
                    bones = bones.clone()

                    # bones= transforms(bones)
                    bones_sobel = sobel(bones)

                    r, t = model1(bones_sobel)

                    T = t + torch.tensor([np.float32(t_iso[0]), np.float32((sid)), np.float32(t_iso[2])],
                                         dtype=t.dtype, device=t.device)
                    pose_pred = convert(r, T, parameterization=parameterization, convention=convention)
                    init_pose = drr2(r, T, parameterization=parameterization, convention=convention)
                    init_pose = transforms(init_pose.sum(dim=1, keepdim=True))
                    init_pose_sobel = sobel(init_pose)

                    input2 = torch.cat((bones_sobel.detach(), init_pose_sobel), dim=1)

                    mpd_iters_tot, mtre_iters_tot = [], []
                    alpha = []
                    beta = []
                    gamma = []
                    x = []
                    y = []
                    z = []
                    time_tot = []
                    for iter in range(1):
                        print(iter)
                        torch.cuda.synchronize()
                        time_i_iter = time.time()
                        delta_r, delta_t = model2(input2)

                        if iter == 0:
                            torch.cuda.synchronize()
                            time_f = time.time() - time_i
                            time_tot.append(float(time_f))

                        delta_pose = convert(delta_r, delta_t, parameterization=parameterization, convention=convention)
                        pose_pred = pose_pred.compose(delta_pose)

                        init_pose = drr2(pose_pred)
                        init_pose = transforms(init_pose.sum(dim=1, keepdim=True))
                        init_pose_sobel = sobel(init_pose)
                        input2 = torch.cat((bones_sobel.detach(), init_pose_sobel), dim=1)
                        time_f_iter = time.time() - time_i_iter

                        if iter != 0:
                            torch.cuda.synchronize()
                            time_f = time_f +time_f_iter
                            time_tot.append(float(time_f))

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

                        p3 = pose.inverse()(subject.fiducials.cuda())
                        p3p = pose_pred.inverse()(subject.fiducials.cuda())
                        # p3[~mask3d] = float('nan')
                        # p3p[~mask3d] = float('nan')
                        mtre_i = (p3 - p3p).norm(dim=-1).nanmean()

                        mpd_iters_tot.append(float(mpd_i.detach().cpu()))
                        mtre_iters_tot.append(float(mtre_i.detach().cpu()))

                        #### Parameters error ####
                        M = pose_pred.rotation
                        angles_pred = diffdrr.pose.matrix_to_euler_angles(M, convention="ZXY") * (180.0 / torch.pi)
                        angles_GT = diffdrr.pose.matrix_to_euler_angles(pose.rotation, convention="ZXY") * (180.0 / torch.pi)
                        R_T = pose.rotation.transpose(1, 2)
                        R_p = pose_pred.rotation.transpose(1, 2)
                        T = torch.einsum("bij, bj -> bi", R_p, pose_pred.translation)
                        gt_t = torch.einsum("bij, bj -> bi", R_T, pose.translation)
                        alpha.append(float(mean_absolute_error(angles_pred.detach().cpu()[:, 0], angles_GT.detach().cpu()[:, 0]).mean()))
                        beta.append(float(mean_absolute_error(angles_pred.detach().cpu()[:, 1], angles_GT.detach().cpu()[:, 1]).mean()))
                        gamma.append(float(mean_absolute_error(angles_pred.detach().cpu()[:, 2], angles_GT.detach().cpu()[:, 2]).mean()))
                        x.append(float(mean_absolute_error(T.detach().cpu()[:, 0], gt_t.detach().cpu()[:, 0]).mean()))
                        y.append(float(mean_absolute_error(T.detach().cpu()[:, 1], gt_t.detach().cpu()[:, 1]).mean()))
                        z.append(float(mean_absolute_error(T.detach().cpu()[:, 2], gt_t.detach().cpu()[:, 2]).mean()))

                    for it, (m,t,a,b,g,tx,ty,tz,ti) in enumerate(zip(mpd_iters_tot, mtre_iters_tot, alpha, beta, gamma, x, y, z, time_tot)):
                        iter_records.append({"patient": id, "iter": it, "mpd": m, "mtre": t, "alpha": a, "beta": b, "gamma": g, "tx": tx, "ty": tz, "tz":ty,"time":ti})

                    if opt == True:
                        with torch.enable_grad():
                            gt, _, _, _, pose_pred = opt_model.run(init_pose=pose_pred, gt=img, beta=0.5)


                    print(mpd_i.cpu().detach().numpy().round(2))
                    print(mtre_i.squeeze().detach().cpu().numpy().round(2))

                    dcm_records.append({
                        "pt_id": id,
                        "dcm_name": img_name,
                        "mpd": float(mpd_i.detach().cpu()),
                        "mtre": float(mtre_i.nanmean().detach().cpu()),
                    })


                    final_pose = drr(pose_pred, mask_to_channels=True)  # (1, C, H, W)
                    final_pose_plot = drr(pose_pred)
                    final_pose_plot =transforms(final_pose_plot)
                    fp = final_pose[0, 1:].detach().cpu().numpy()  # (C, H, W)
                    CHANNEL_COLORS = {
                        0: np.array([0.0, 0.0, 1.0]),  # aorta - blue
                        1: np.array([0.0, 1.0, 0.0]),  # ncc   - green
                        2: np.array([0.0, 0.5, 1.0]),  # lcc   - light blue
                        3: np.array([1.0, 1.0, 0.0]),  # rcc   - yellow
                    }

                    rgb_fluoro = np.stack([1 - img_plot[0].squeeze(0).squeeze(0).detach().cpu()] * 3, axis=-1)

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

                    plt.figure(figsize=(12, 4))
                    plt.subplot(141)
                    plt.grid(False)
                    plt.title("Target fluoroscopy")
                    plt.imshow(1 - img_plot[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
                    plt.axis('off')

                    plt.subplot(142)
                    plt.grid(False)
                    plt.title("DRR Ppred")
                    plt.imshow(1- final_pose_plot[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
                    plt.scatter(p2.squeeze()[:, 0].detach().cpu().numpy(),
                                p2.squeeze()[:, 1].detach().cpu().numpy(),
                                c='fuchsia', s=160, alpha=1, marker='o')
                    plt.scatter(p2p.squeeze()[:, 0].detach().cpu().numpy(),
                                p2p.squeeze()[:, 1].detach().cpu().numpy(),
                                c='lime', s=160, alpha=1, marker='x', linewidths=3)
                    plt.axis('off')

                    plt.subplot(143)
                    plt.grid(False)
                    plt.title("Bones DiffMap")
                    plt.imshow(rgb_bones)
                    plt.axis('off')

                    plt.subplot(144)
                    plt.grid(False)
                    plt.title("Aorta projection")
                    plt.imshow(rgb_fluoro)
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.05, hspace=0, left=0, right=1, top=0.95, bottom=0)
                    # plt.savefig(os.path.join(dest_path, img_name + f"_{i}.png"),
                    #             bbox_inches='tight', pad_inches=0)
                    plt.close()

                    del final_pose, init_pose, img, input2, bones


#
iter_df = pd.DataFrame(iter_records)
iter_df.to_csv(os.path.join(summary_dir, "all_metrics_final.csv"))

# step 1: one value per (patient, iter) — mean over that patient's frames
per_patient_iter = (iter_df.groupby(["patient", "iter"])[["mpd", "mtre", "alpha", "beta","gamma", "tx", "ty", "tz","time"]]
                    .mean().reset_index())

# step 2: median (and spread) across patients, per iter
iter_summary = per_patient_iter.groupby("iter").agg(
    mpd_mean=("mpd", "mean"),
    mpd_std=("mpd", "std"),
    mtre_mean=("mtre", "mean"),
    mtre_std=("mtre", "std"),
    alpha_mean=("alpha", "mean"),
    alpha_std=("alpha", "std"),
    beta_mean=("beta", "mean"),
    beta_std=("beta", "std"),
    gamma_mean=("gamma", "mean"),
    gamma_std=("gamma", "std"),
    tx_mean=("tx", "mean"),
    tx_std=("tx", "std"),
    ty_mean=("ty", "mean"),
    ty_std=("ty", "std"),
    tz_mean=("tz", "mean"),
    tz_std=("tz", "std"),
    time_mean=("time", "mean"),
    time_std=("time", "std"),
)
iter_summary.to_csv(os.path.join(summary_dir, "summary_final.csv"))
