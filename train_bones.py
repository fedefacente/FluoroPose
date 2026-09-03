from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import wandb
from diffdrr.pose import convert, axis_angle_to_matrix,matrix_to_axis_angle
import diffdrr.data
from diffdrr.drr import DRR
from torchvision.transforms import Compose, Lambda, Normalize

from models import RegressorDecoderHRNetNew
from models import RegressorDecoderHRNet
from models import encoder_decoder, registration_attention
from geodesic import GeodesicSE3
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d, DoubleGeodesicSE3,GradientNormalizedCrossCorrelation2d,NormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
import kornia
from kornia.augmentation import AugmentationSequential
import kornia.augmentation as K
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from kornia.losses import SSIMLoss
from torch.nn import MSELoss
from hu_to_mu import AttenuationMapper, AttenuationMapperRatio
from utils import simulate_catheter_v1
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="0f7f6475")
args = parser.parse_args()
id = args.id

def generate_pytorch_catheter_batch(batch_size=16, img_size=256, num_samples=200, device='cuda'):
    """
    Generate a batch of masks with random Bézier curves.

    Returns:
        masks: [batch_size, img_size, img_size], float32, 0/1
    """

    # 1. Randomize 4 control points for each batch item
    p = torch.randint(20, img_size - 20, (batch_size, 4, 2), device=device).float()

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
        intensities = torch.rand(batch_size, device=device)
        masks *= intensities[:, None, None]

    return masks.to(device)

def transform_hu_to_density(volume):
    volume = volume.float()
    density = volume.clamp(min=-800)
    # shift = torch.empty(1).uniform_(0, 600).to(density.device)
    # soft = torch.empty(1).uniform_(-200, 350).to(density.device)
    # density += shift * ((density < soft).to(density.dtype) )

    density -= density.min()
    density /= density.max()
    return density

def transform_labels_to_density(volume):
    volume = volume.float()
    out = torch.zeros_like(volume)
    out = torch.where(volume == 1, 1.0,  out)   # delivery sheath (polymer)
    out = torch.where(volume == 2, 6.0,  out)   # valve frame
    out = torch.where(volume == 3, 10.0, out)   # Safari wire (stiff, radiopaque)
    return out
def train(
        model,
        reg_model1,
        optimizer,
        scheduler,
        drr,
        drr_bones,
        drr_catheter,
        transforms,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        parameterization,
        convention,
        output_path,
        subject,
        instrument_iter,
        loader,
        aug,
        start_epoch,
        t_iso):
    torch.cuda.empty_cache()
    # geodesic = GeodesicSE3().to(device)
    n_batches_per_epoch_val = 2
    #double = DoubleGeodesic(1020).to(device)
    min_delta = 0.01
    patience = 8
    best_loss = np.inf  # init to infinity
    ncc = MultiscaleNormalizedCrossCorrelation2d([None, 16], [0.5, 0.5]).to(device)
    ncc_grad =  GradientNormalizedCrossCorrelation2d().to(device)
    ncc =  NormalizedCrossCorrelation2d().to(device)
    # ssim =  SSIMLoss(5)
    # contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
    l2_loss = MSELoss()
    volume_anatomy = drr.subject.volume.data.to(device)
    mapper = AttenuationMapperRatio().to('cuda').eval()
    dens = mapper.density(volume_anatomy.clone())
    for epoch in range(start_epoch,n_epochs + 1):
        losses = []
        loss_cnn1 = []
        loss_cnn3 = []
        losses_val = []
        #ncc1mean = []
        #ncc2mean = []
        model.train()
        reg_model1.train()
        mae_loss = nn.L1Loss(reduction='none')

        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):

            sid = -820.0
            # torch.manual_seed(torch.seed())

            #rx = torch.FloatTensor(batch_size, 1).uniform_(-0.7, 0.7).to(device)
            #ry = torch.FloatTensor(batch_size, 1).uniform_(-0.3, 0.3).to(device)
            #rz = torch.FloatTensor(batch_size, 1).uniform_(-0.7, 0.7).to(device)
            #rot = torch.cat([rx, ry, rz], dim=1).to(device)

            #tx = torch.FloatTensor(batch_size, 1).uniform_(-100, 100).to(device)
            #ty = torch.FloatTensor(batch_size, 1).uniform_(-150, 150).to(device)
            #tz = torch.FloatTensor(batch_size, 1).uniform_(-100, 100).to(device)

            rx = torch.normal(mean=0.0, std=0.4, size=(batch_size, 1)).to(device)
            ry = torch.normal(mean=0.0, std=0.2, size=(batch_size, 1)).to(device)
            rz = torch.normal(mean=0.0, std=0.4, size=(batch_size, 1)).to(device)
            rot = torch.cat([rx, ry, rz], dim=1).to(device)

            tx = torch.normal(mean=0, std=30, size=(batch_size, 1)).to(device)
            ty = torch.normal(mean=0, std=75, size=(batch_size, 1)).to(device)
            tz = torch.normal(mean=0, std=30, size=(batch_size, 1)).to(device)

            xyz = torch.cat([tx, ty, tz], dim=1).to(device)
            xyz =xyz  + torch.tensor([np.float32(t_iso[0]), np.float32(sid), np.float32(t_iso[2])], dtype=tx.dtype,device=tx.device)

            '''contrast = contrast_distribution.sample().item()
            drr.density = diffdrr.data.transform_hu_to_density(drr.subject.volume.data,
                                                               bone_attenuation_multiplier=torch.tensor(contrast,
                                                                                                        dtype=torch.float32)).to(
                device).squeeze(0)'''


            # if torch.rand(1).item() > 0.5:
            #     max = (torch.rand(1, device=device) * 0.3) + 0.7
            #     min = (torch.rand(1, device=device) * 0.3)
            #     img = (img * (max - min)) + min

            bones_gt = drr_bones(rot, xyz, parameterization=parameterization, convention=convention)
            # bones_gt=-torch.log(bones_gt.clamp(min=1e-6))
            bones_gt = transforms(bones_gt)



            # img = aug(img)

            with torch.no_grad():
                try:
                    instruments = next(instrument_iter)
                except StopIteration:
                    # If we ran through all files, automatically reshuffle and restart
                    instrument_iter = iter(loader)
                    instruments = next(instrument_iter)

                instruments = instruments.to(device, non_blocking=True).float().squeeze(0)
                # volume = torch.where(instruments > 0, instruments, volume_anatomy)
                # drr.density = transform_hu_to_density(volume).squeeze(0)

                # drr_catheter.density = transform_hu_to_density(instruments).squeeze(0)
                # catheter = drr_catheter(rot, xyz, parameterization='axis_angle', convention=None)
                # catheter = transforms(catheter.sum(dim=1, keepdim=True))
                # alpha = torch.rand(1, device=device).item() * 0.5
                mu = mapper(volume_anatomy.clone(), instruments, dens=dens)
                drr.density = mu.to(drr.density)
                img = drr(rot, xyz, parameterization=parameterization, convention=convention)
                img = transforms(img)
                img = aug(img)
                # catheter = generate_pytorch_catheter_batch(batch_size=1).unsqueeze(0)
                # alpha = torch.rand(1, device=device).item() * 0.5
                # img = torch.where(catheter > 0, catheter * alpha + img, img)
                # # img = torch.where(catheter > 0, catheter * alpha + img, img)



            _,_,bones = model(img)
            # bones = transforms(bones)
            # bones = torch.sigmoid(bones)

            loss = (1 - ncc_grad(bones, bones_gt)) #(1 - ms_ssim(bones, bones_gt, data_range =1) )
            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            adaptive_clip_grad_(reg_model1.parameters())
            optimizer.step()
            scheduler.step()
            losses.append(loss.mean().item())


            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                loss=loss.mean().item(),
            )
        losses = torch.tensor(losses)


        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.nanmean().item():.4f}")

        if (epoch + 1) % n_epochs == 0:
            torch.save(
                {
                    "model_state_dict1": model.state_dict(),
                    "model_state_dict2": reg_model1.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(output_path, f'model_final.pth'),
            )


        elif (epoch + 1) % 100 == 0:
            torch.save(
                {
                    "model_state_dict1": model.state_dict(),
                    "model_state_dict2": reg_model1.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(output_path, f'model_{epoch}.pth'),
            )

def main(
    parameterization="axis_angle",
    convention=None,
    lr=1e-3,
    batch_size=16,
    n_epochs=500,
    n_batches_per_epoch=100,
    resume=False
):


    device = torch.device("cuda")
    csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/{id}_50_vertebrae.csv"
    df = pd.read_csv(csv_file_path)
    fiducials = np.array(df.values)  # Extract as NumPy array
    fiducials = torch.tensor([fiducials]).to(torch.float)

    subject = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/volume.nii.gz",
                                orientation="PA", bone_attenuation_multiplier=1, fiducials=fiducials)
    drr = DRR(
        subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
        sdd=1125,  # Source-to-detector distance (i.e., the C-arm's focal length)
        height=256,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        delx=(0.4*512/256),  # Pixel spacing (in mm)
        renderer="trilinear",
        reverse_x_axis=True
    ).to(device)

    subject2 = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/volume.nii.gz", labelmap = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/vertebrae.nii.gz",
                                labels =1, orientation="PA", bone_attenuation_multiplier=30, fiducials=fiducials)

    drr_bones = DRR(
        subject2,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
        sdd=1125,  # Source-to-detector distance (i.e., the C-arm's focal length)
        height=256,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        delx=(0.4*512/256),  # Pixel spacing (in mm)
        renderer="trilinear",
        reverse_x_axis=True
    ).to(device)

    drr_catheter = DRR(
        subject2,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
        sdd=1125,  # Source-to-detector distance (i.e., the C-arm's focal length)
        height=256,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        delx=(0.4*512/256),  # Pixel spacing (in mm)
        renderer="trilinear",
        reverse_x_axis=True
    ).to(device)

    #drr_bones.density = diffdrr.data.transform_hu_to_density(drr.subject.volume.data,bone_attenuation_multiplier=torch.tensor(10,dtype=torch.float32)).to(device).squeeze(0)

    csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/points_centerline.csv"
    df = pd.read_csv(csv_file_path)
    centerline = np.array(df.values)  # Extract as NumPy array
    centerline = torch.tensor([centerline]).to(torch.float)
    subject_catheter = diffdrr.data.read(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/ct/aorta.nii.gz",
                                         orientation="PA", bone_attenuation_multiplier=1, fiducials=centerline)
    model = RegressorDecoderHRNetNew()
    reg_model1 = registration_attention()

    model = model.to(device)
    reg_model1 = reg_model1.to(device)
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr}])


    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    output_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/model/tavi_bones_new_patients_instance_norm/{id}_hu_to_mu_ratio_bones_edges"
    os.makedirs(output_path, exist_ok=True)

    transforms = Compose(
        [
            # Lambda(lambda x: (x.max() + x.min() - x)),
            Lambda(lambda x: (((x - x.min()) / (x.max() - x.min()+ 1e-6)))),
            #Normalize(mean=0.3080, std=0.1494),
        ]
    )


    aug = AugmentationSequential(
        kornia.augmentation.RandomSaltAndPepperNoise(amount=(0.005, 0.01), salt_vs_pepper=(0.4, 0.6), p=0.5,same_on_batch=False, keepdim=True),
        kornia.augmentation.RandomGamma(gamma=(0.6, 1.8), gain=(1.0, 1.0), same_on_batch=False, p=0.5, keepdim=True),
        kornia.augmentation.RandomGaussianNoise(mean=0.0, std=0.03, same_on_batch=False, p=0.5, keepdim=True),
        kornia.augmentation.RandomSharpness(p=0.5, keepdim=True, same_on_batch=False),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1), p=0.5, keepdim=True,same_on_batch=False),
        kornia.augmentation.RandomPlasmaContrast(roughness=(0.1, 0.5),p=1.0, keepdim=True, same_on_batch=False),
        kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.5), intensity=(-1, 1),p=1.0, keepdim=True, same_on_batch=False),
        kornia.augmentation.RandomPlasmaShadow(roughness=(0.1, 0.7), shade_intensity=(-0.5, 0.5),shade_quantity=(0.0, 0.5), same_on_batch=False, p=1.0, keepdim=False),
        #kornia.augmentation.RandomErasing(p=0.5, keepdim=True, same_on_batch=False, value = 1.0),
        data_keys=["input"],
        same_on_batch=None,
    )

    iso = pd.read_csv(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/isocenter.csv")
    iso = iso[["T8x", "T8y", "T8z"]].to_numpy()

    t_iso = np.array([((iso[0][0]-subject.shape[1]/2)*subject.spacing[0]),((iso[0][1]-subject.shape[2]/2)*subject.spacing[1]),((iso[0][2]-subject.shape[3]/2)*subject.spacing[2])], dtype = np.float32)
    from data_loader import InstrumentDataset
    from torch.utils.data import Dataset, DataLoader

    dataset = InstrumentDataset(f"/lustre/fswork/projects/rech/gfu/uur34ii/data/TAVI/{id}/instruments_volume")

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2
    )

    instrument_iter = iter(loader)

    if resume == False:
        start_epoch = 0
    else:
        model_dict = torch.load(
            f"/lustre/fswork/projects/rech/gfu/uur34ii/model/tavi_bones_new_patients_instance_norm/1e7dd220_hu_to_mu_ratio_bones_edges/model_final.pth")
        model.load_state_dict(model_dict["model_state_dict1"])
        reg_model1.load_state_dict(model_dict["model_state_dict2"])
        start_epoch = 0
        # optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)
        # scheduler.load_state_dict(model_dict["scheduler_state_dict"])

    train(
        model,
        reg_model1,
        optimizer,
        scheduler,
        drr,
        drr_bones,
        drr_catheter,
        transforms,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        parameterization,
        convention,
        output_path,
        subject_catheter,
        instrument_iter,
        loader,
        aug,
        start_epoch,
        t_iso)

if __name__ == "__main__":
    main(resume=False)