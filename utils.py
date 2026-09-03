import torch.nn.functional as F
import torch
from kornia.geometry.transform import center_crop, resize, translate
import math

def sobel(x):
    a = torch.tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]], dtype=torch.float32).view((1, 1, 3, 3)).to('cuda')

    b = torch.tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]], dtype=torch.float32).view((1, 1, 3, 3)).to('cuda')

    # Apply reflection padding before convolution
    x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')

    G_x = F.conv2d(x_padded, a)
    G_y = F.conv2d(x_padded, b)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

    return G



import pandas as pd
import torch
from tqdm import tqdm
from diffdrr.registration import Registration
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d,GradientNormalizedCrossCorrelation2d
beta = 0.5
sim1 = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5]).cuda()
sim2 = GradientNormalizedCrossCorrelation2d(patch_size=11, sigma=10).cuda()
criterion = lambda x, y: beta * sim1(x, y) + (1 - beta) * sim2(x, y)

def optimize(
    reg: Registration,
    ground_truth,
    lr_rotations=5e-2,
    lr_translations=1e2,
    momentum=0,
    dampening=0,
    n_itrs=200,
    optimizer="sgd",  # 'sgd' or `adam`
):
    # Initialize an optimizer with different learning rates
    # for rotations and translations since they have different scales
    if optimizer == "sgd":
        optim = torch.optim.SGD(
            [
                {"params": [reg._rotation], "lr": lr_rotations},
                {"params": [reg._translation], "lr": lr_translations},
            ],
            momentum=momentum,
            dampening=dampening,
            maximize=True,
        )
        optimizer = optimizer.upper()
    elif optimizer == "adam":
        optim = torch.optim.Adam(
            [
                {"params": [reg._rotation], "lr": lr_rotations},
                {"params": [reg._translation], "lr": lr_translations}
            ],
            maximize=True,
        )
        optimizer = optimizer.upper()
    else:
        raise ValueError(f"Unrecognized optimizer {optimizer}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim,
    factor=0.1,
    mode="max",
    )
    params = []
    losses = [criterion(ground_truth, reg()).item()]
    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        # Save the current set of parameters
        alpha, beta, gamma = reg.rotation.squeeze().tolist()
        bx, by, bz = reg.translation.squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

        # Run the optimization loop
        optim.zero_grad()
        estimate = reg()
        loss = criterion(ground_truth, estimate)
        loss.backward()
        optim.step()
        scheduler.step(loss)
        losses.append(loss.item())
        pbar.set_description(f"NCC = {loss.item():06f}")

        # Stop the optimization if the estimated and ground truth images are 99.9% correlated
        if loss > 0.999:
            if momentum != 0:
                optimizer += " + momentum"
            if dampening != 0:
                optimizer += " + dampening"
            tqdm.write(f"{optimizer} converged in {itr + 1} iterations")
            break

    # Save the final estimated pose
    alpha, beta, gamma = reg.rotation.squeeze().tolist()
    bx, by, bz = reg.translation.squeeze().tolist()
    params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    return df

import warnings
import numpy as np
import torchio
import ants
from torchio.data.io import get_sitk_metadata_from_ras_affine
from diffdrr.pose import RigidTransform, convert, make_matrix
def direction(img: torchio.ScalarImage):
    """Volume directions in RAS space (comport with ANTS convention)."""
    *_, direction = get_sitk_metadata_from_ras_affine(img.affine)
    return np.array(direction).reshape(3, 3)


def get_4x4(mat, img, isocenter, invert=False):
    """Get the rigid or affine matrix for warping img_warped -> img."""
    img = torchio.ScalarImage(img)

    transform = ants.read_transform(mat)
    if invert:
        transform = transform.invert()
    R = transform.parameters[:9].reshape(3, 3)
    t = transform.parameters[9:]
    c_old = transform.fixed_parameters
    c_ct_old = isocenter

    global_t = + (R @ (c_ct_old - c_old) + t + c_old)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = global_t

    D = np.eye(4)
    if img.orientation == ("L", "P", "S"):
        D[:3, :3] = np.array(img.direction).reshape(3, 3)
    elif img.orientation == ("R", "A", "S"):
        D[:3, :3] = direction(img)
    else:
        warnings.warn(
            f"Unrecognized orientation {img.orientation}; assuming LPS+ directions. If the corrected pose is completely wrong, check here first."
        )
        D[:3, :3] = np.array(img.direction).reshape(3, 3)

    Tinv = np.eye(4)
    Tinv[:3, 3] = -np.array(img.get_center())

    T = Tinv @ D @ M @ np.linalg.inv(D)
    T = torch.from_numpy(T).to(torch.float32)
    T = RigidTransform(T)

    return project_onto_SO3(T.inverse())


def project_onto_SO3(T: RigidTransform):
    """Convert the upper 3x3 to a matrix in SO(3) (i.e., unitary with det=+1)."""
    M = T.matrix[0]
    A = M[:3, :3]
    At = M[:3, 3]
    U, S, V = A.svd()
    t = A.inverse() @ At
    S = torch.ones_like(S)
    S[..., -1] = (U @ V.mT).det()
    R = torch.einsum("ij, j, jk -> ik", U, S, V.mT)
    t = R @ t
    return RigidTransform(make_matrix(R[None], t[None]))


def _correct_pose(pose, warp, volume,isocenter, invert):
    if warp is None:
        return pose

    # Get the closest SE(3) transformation relating the CT to some reference frame
    T = get_4x4(warp, volume, isocenter, invert).cuda()
    return pose.compose(T)



def resample(
    img: torch.Tensor,
    focal_len: float,
    delx: float,
    x0: float = 0,
    y0: float = 0,
    new_focal_len: float = None,
    new_delx: float = None,
    new_x0: float = None,
    new_y0: float = None,
) -> torch.Tensor:
    """Resample an image with new intrinsic parameters."""
    if new_focal_len is None:
        new_focal_len = focal_len
    if new_delx is None:
        new_delx = delx
    if new_x0 is None:
        new_x0 = x0
    if new_y0 is None:
        new_y0 = y0

    x = img.clone()
    _, _, height, width = x.shape
    shape = torch.tensor([height, width])

    # Translate the image
    translation = torch.tensor([[new_x0 - x0, new_y0 - y0]]) / delx
    x = translate(x, translation.to(x))

    # Crop the image to change the focal length
    focal_scaling = new_focal_len / focal_len
    crop_size = (shape / focal_scaling).to(int).tolist()
    x = center_crop(x, crop_size)
    x = resize(x, (height, width))

    # Pad the image to resize pixels
    pixel_scaling = new_delx / delx
    padding = (shape * (pixel_scaling - 1) / 2).to(int).tolist()
    padding = [padding[1], padding[1], padding[0], padding[0]]
    x = torch.nn.functional.pad(x, padding)
    x = resize(x, (height, width))

    return x


def square(img: torch.Tensor,height,width)-> torch.Tensor:
    # padding real image
    target_size = max(height,width)
    pad_h = target_size - height
    pad_w = target_size - width
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    value = img.max()
    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=value)

    return img


import kornia.geometry as kg

def warp_to_canonical_space(img, K_real, K_can):
    B, C, H_img, W_img = img.shape

    def to_img_coords(K):
        K_adj = K.clone()
        K_adj[:, 1, 2] = H_img - K[:, 1, 2]
        return K_adj

    K_real_adj = to_img_coords(K_real)
    K_can_adj  = to_img_coords(K_can)

    H = K_can_adj @ torch.inverse(K_real_adj)

    img_warped = kg.warp_perspective(
        img, H, (H_img, W_img),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return img_warped


def simulate_catheter_v1(volume_tensor, points_tensor, affine, spacing, final_volume):
    CATHETER_RADIUS_MM = 0.5
    CATHETER_INTENSITY = torch.randint(15000, 30000, (1,)).item()
    N_SPLINE_SAMPLES = 10000
    CATHETER_RADIUS_MM2 = (2.0 + torch.rand(1) * (4.0 - 2.0)).item()
    CATHETER_INTENSITY2 = torch.randint(2000, 10000, (1,)).item()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    shape = torch.tensor(volume_tensor.squeeze(0).shape, dtype=torch.long, device=DEVICE)
    zooms = torch.tensor(spacing, dtype=torch.float32, device=DEVICE)
    # ── RAS → voxel ───────────────────────────────────────────────────────────────

    landmarks_ras = points_tensor.to(DEVICE)  # (N, 3)
    inv_affine = torch.linalg.inv(affine).float()

    # Homogeneous coords: (N, 4)
    B, N, _ = landmarks_ras.shape  # 1, 10, 3
    ones = torch.ones(B, N, 1, dtype=landmarks_ras.dtype, device=landmarks_ras.device)
    pts_h = torch.cat([landmarks_ras, ones], dim=-1)
    vox_points = pts_h @ inv_affine.T  # (1, 10, 4)
    vox_points = vox_points[..., :3]

    aorta_voxels = torch.nonzero(volume_tensor.squeeze(0))  # (N, 3)

    # Highest z in the volume
    z_max_volume = aorta_voxels[:, 2].max()  # scalar

    # Highest z among catheter landmarks
    z_max_points = vox_points.squeeze(0)[:, 2].max()  # scalar

    # How much to translate up (in voxels), leaving room for the tube radius
    radius_vox_z = CATHETER_RADIUS_MM * 4 / zooms[2]  # radius in z voxels
    translation_z = z_max_volume - z_max_points - radius_vox_z
    snapped = vox_points.squeeze(0).clone()
    snapped = apply_stiffness(snapped.squeeze(0), iterations=10)

    snapped_origin = vox_points.squeeze(0).clone()

    # Apply to all landmarks (only z axis)
    if torch.rand(1).item() > 0.2:
        vox_points[:, 2:8, 2] += translation_z

        # Highest x in the volume
        x_max_volume = aorta_voxels[:, 0].max()  # scalar
        x_max_points = vox_points.squeeze(0)[:, 0].max()  # scalar
        radius_vox_x = CATHETER_RADIUS_MM * 4 / zooms[0]  # radius in z voxels
        translation_x = x_max_volume - x_max_points - radius_vox_x
        vox_points[:, 0:2, 0] += translation_x / 2

        # lowers x in the volume
        x_min_volume = aorta_voxels[:, 0].min()  # scalar
        x_min_points = vox_points.squeeze(0)[:, 0].min()  # scalar
        radius_vox_x = CATHETER_RADIUS_MM * 4 / zooms[0]  # radius in z voxels
        translation_x = x_min_volume - x_min_points - radius_vox_x
        vox_points[:, 8:, 0] += translation_x / 2

        vol_bool = (volume_tensor.squeeze(0) == 1)
        wall_mask = (
                vol_bool & (
                ~torch.roll(vol_bool, 1, 0) | ~torch.roll(vol_bool, -1, 0) |
                ~torch.roll(vol_bool, 1, 1) | ~torch.roll(vol_bool, -1, 1) |
                ~torch.roll(vol_bool, 1, 2) | ~torch.roll(vol_bool, -1, 2)
        )
        )
        wall_voxels = torch.nonzero(wall_mask).float().to(DEVICE)  # (W, 3)
        wall_mm = wall_voxels * zooms

        anchor_idx = int(vox_points[:, :, 2].argmax())  # highest z = top of catheter

        snapped = vox_points.squeeze(0).clone()
        order_down = list(range(anchor_idx + 1, snapped.shape[0]))
        random_length_down = torch.randint(1, len(order_down) + 1, (1,)).item()
        order_down = order_down[0:random_length_down]

        order_up = list(range(anchor_idx - 1, -1, -1))
        random_length_up = torch.randint(1, len(order_up) + 1, (1,)).item()
        order_up = order_up[0:random_length_up]

        for prev_idx in order_up:
            prev_mm = snapped[prev_idx] * zooms
            dists_from_prev = torch.sum((wall_mm - prev_mm) ** 2, dim=1)
            best = dists_from_prev.argmin()
            target_mm = wall_mm[best]
            direction = target_mm - prev_mm
            dist_to_target = torch.norm(direction)
            if dist_to_target > 1e-6:
                unit_direction = direction / dist_to_target
            else:
                unit_direction = torch.zeros_like(direction)
            new_point_mm = target_mm - (unit_direction * CATHETER_RADIUS_MM)
            new_voxel_coords = torch.round(new_point_mm / zooms).long()
            snapped[prev_idx] = new_voxel_coords
        for prev_idx in order_down:
            prev_mm = snapped[prev_idx] * zooms
            dists_from_prev = torch.sum((wall_mm - prev_mm) ** 2, dim=1)
            best = dists_from_prev.argmin()
            target_mm = wall_mm[best]
            direction = target_mm - prev_mm
            dist_to_target = torch.norm(direction)
            if dist_to_target > 1e-6:
                unit_direction = direction / dist_to_target
            else:
                unit_direction = torch.zeros_like(direction)
            new_point_mm = target_mm - (unit_direction * CATHETER_RADIUS_MM)
            new_voxel_coords = torch.round(new_point_mm / zooms).long()
            snapped[prev_idx] = new_voxel_coords
        snapped = apply_stiffness(snapped.squeeze(0), iterations=10)
        snapped_origin = snapped.clone()

    else:
        vox_points[:, 2:8, 2] += translation_z

        # Highest x in the volume
        x_max_volume = aorta_voxels[:, 0].max()  # scalar
        x_max_points = vox_points.squeeze(0)[:, 0].max()  # scalar
        radius_vox_x = CATHETER_RADIUS_MM * 4 / zooms[0]  # radius in z voxels
        translation_x = x_max_volume - x_max_points - radius_vox_x
        vox_points[:, 1:2, 0] += translation_x / 2

        # lowers x in the volume
        x_min_volume = aorta_voxels[:, 0].min()  # scalar
        x_min_points = vox_points.squeeze(0)[:, 0].min()  # scalar
        radius_vox_x = CATHETER_RADIUS_MM * 4 / zooms[0]  # radius in z voxels
        translation_x = x_min_volume - x_min_points - radius_vox_x
        vox_points[:, 8:, 0] += translation_x / 2
        snapped = apply_stiffness(snapped.squeeze(0), iterations=10)
        snapped_origin = snapped.clone()

    ##### ADD pigtail
    if torch.rand(1).item() > 0.3:
        P0 = snapped[0, :]
        P1 = snapped[1, :]

        # Tangent vector pointing from the aorta DOWN into the LV
        v = P0 - P1
        v_norm = torch.linalg.norm(v)

        if v_norm > 0:
            v_unit = v / v_norm
        else:
            v_unit = torch.tensor([0.0, 0.0, 1.0], device=DEVICE, dtype=P0.dtype)

        R = torch.randint(15, 30, (1,)).item()

        L_straight = torch.randint(30, 60, (1,)).item()
        P_straight_end = P0 + L_straight * v_unit

        # ==========================================
        # --- 6. Generate Straight Segment ---
        # ==========================================
        num_straight_points = 2

        # PyTorch linspace on the correct device
        straight_distances = torch.linspace(L_straight, 0, num_straight_points, device=DEVICE, dtype=P0.dtype)[:-1]

        # Vectorized point generation (no for-loop required)
        straight_points = P0.unsqueeze(0) + straight_distances.unsqueeze(1) * v_unit.unsqueeze(0)

        # ==========================================
        # ---  Generate Pigtail ---
        # ==========================================
        arbitrary_vec = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=P0.dtype)

        # Check dot product alignment
        if torch.abs(torch.dot(v_unit, arbitrary_vec)) > 0.99:
            arbitrary_vec = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=P0.dtype)

        n = torch.linalg.cross(v_unit, arbitrary_vec)
        n_unit = n / torch.linalg.norm(n)
        theta = torch.rand(1, device=DEVICE, dtype=P0.dtype) * (90.0 * math.pi / 180.0)
        n_unit = n_unit * math.cos(theta) + torch.linalg.cross(v_unit, n_unit) * math.sin(theta)

        C = P_straight_end + R * n_unit
        theta_max = torch.randint(1, 3, (1,)).item() * math.pi
        num_pigtail_points = 20

        # Generate all angles at once
        thetas = torch.linspace(theta_max, 0, num_pigtail_points, device=DEVICE, dtype=P0.dtype)

        # Vectorized spiral radius calculations
        r_spiral = R * (1.0 - 0.8 * (thetas / theta_max))

        # Calculate the entire pigtail curve simultaneously
        pigtail_points = (
                C.unsqueeze(0)
                - (r_spiral * torch.cos(thetas)).unsqueeze(1) * n_unit.unsqueeze(0)
                + (r_spiral * torch.sin(thetas)).unsqueeze(1) * v_unit.unsqueeze(0)
        )

        # Drop the last point to avoid duplicating the connection point
        pigtail_points = pigtail_points[:-1]

        # ==========================================
        # --- 8. Combine the Tensors ---
        # ==========================================
        snapped = torch.cat((pigtail_points, straight_points, snapped), dim=0)

    if torch.rand(1).item() > 0.1:
        curve = catmull_rom_spline(snapped_origin.squeeze(0), n_samples=N_SPLINE_SAMPLES)  # (S, 3)

        curve = curve.unsqueeze(0)  # Shape becomes (1, S, 3)

        # FIX: Add .squeeze(0) so we extract points as (S, 3) before rounding and unique
        centers_raw = torch.round(curve.squeeze(0)).long()  # (S, 3)
        centers = torch.unique(centers_raw, dim=0)  # Shape is now correctly (C, 3)

        # ── Anisotropic ellipsoid footprint ───────────────────────────────────────────

        radius_vox = CATHETER_RADIUS_MM2 / zooms  # (3,) anisotropic
        r_max = int(torch.ceil(radius_vox.max()).item())

        r = torch.arange(-r_max, r_max + 1, device=DEVICE, dtype=torch.float64)
        gx, gy, gz = torch.meshgrid(r, r, r, indexing='ij')  # (2r+1, 2r+1, 2r+1)

        dist = torch.sqrt((gx / radius_vox[0]) ** 2 +
                          (gy / radius_vox[1]) ** 2 +
                          (gz / radius_vox[2]) ** 2)

        offsets = torch.argwhere(dist <= 1.0) - r_max  # (N_off, 3)

        # ── Stamp catheter into volume ────────────────────────────────────────────────
        # Now this Broadcast correctly: (C, 1, 3) + (1, N_off, 3) → (C, N_off, 3) → (C*N_off, 3)
        all_coords = (centers.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)

        # Bounds check
        in_bounds = torch.all((all_coords >= 0) & (all_coords < shape.unsqueeze(0)), dim=1)
        all_coords = torch.unique(all_coords[in_bounds], dim=0)  # (M, 3)

        # index_put_ for in-place scatter
        final_volume.squeeze(0).index_put_(
            (all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]),
            torch.tensor(CATHETER_INTENSITY2, dtype=final_volume.dtype, device=DEVICE),
        )

    curve = catmull_rom_spline(snapped.squeeze(0), n_samples=N_SPLINE_SAMPLES)  # (S, 3)

    curve = curve.unsqueeze(0)  # Shape becomes (1, S, 3)

    # FIX: Add .squeeze(0) so we extract points as (S, 3) before rounding and unique
    centers_raw = torch.round(curve.squeeze(0)).long()  # (S, 3)
    centers = torch.unique(centers_raw, dim=0)  # Shape is now correctly (C, 3)

    # ── Anisotropic ellipsoid footprint ───────────────────────────────────────────

    radius_vox = CATHETER_RADIUS_MM / zooms  # (3,) anisotropic
    r_max = int(torch.ceil(radius_vox.max()).item())

    r = torch.arange(-r_max, r_max + 1, device=DEVICE, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(r, r, r, indexing='ij')  # (2r+1, 2r+1, 2r+1)

    dist = torch.sqrt((gx / radius_vox[0]) ** 2 +
                      (gy / radius_vox[1]) ** 2 +
                      (gz / radius_vox[2]) ** 2)

    offsets = torch.argwhere(dist <= 1.0) - r_max  # (N_off, 3)

    # ── Stamp catheter into volume ────────────────────────────────────────────────
    # Now this Broadcast correctly: (C, 1, 3) + (1, N_off, 3) → (C, N_off, 3) → (C*N_off, 3)
    all_coords = (centers.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)

    # Bounds check
    in_bounds = torch.all((all_coords >= 0) & (all_coords < shape.unsqueeze(0)), dim=1)
    all_coords = torch.unique(all_coords[in_bounds], dim=0)  # (M, 3)

    # index_put_ for in-place scatter
    final_volume.squeeze(0).index_put_(
        (all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]),
        torch.tensor(CATHETER_INTENSITY, dtype=final_volume.dtype, device=DEVICE),
    )


    return final_volume

import torch
import torch.nn as nn
import torch.nn.functional as F
class NormalizedGradientField2d(nn.Module):
    """Squared-form NGF similarity for ``(B, C, H, W)`` images.

    Args:
        eta: edge/noise floor (the eta in the denominator). This is a *real
            hyperparameter*, not a numerical stabilizer -- set it to the
            fluoroscopy noise level, in the units of the gradient magnitude, and
            set it *after* any log-compression of the DRR. With a high
            ``bone_attenuation_multiplier`` the raw bone-edge gradients otherwise
            dominate the magnitude scale eta competes against, biasing the metric
            toward bone alignment and starving softer structures. Default 1e-2
            assumes images in ~[0, 1]; retune for your intensity range.
        grad_method: gradient operator, ``{'central', 'sobel'}``. Both are scaled
            so a unit-slope ramp yields unit gradient magnitude, i.e. eta
            transfers between them. Match this to the operator used in your
            agreement-map module so the two share a gradient convention.
        as_loss: if ``False`` (default) return the NGF *similarity* to maximize,
            matching the DiffDRR metric convention. If ``True`` return
            ``1 - NGF``, a loss in [0, 1] to minimize.
        reduction: ``{'mean', 'none'}`` over the batch. ``'mean'`` returns a
            scalar (DiffDRR default); ``'none'`` returns a per-sample vector.
    """

    def __init__(
            self,
            eta: float = 1e-2,
            grad_method: str = "sobel",
            as_loss: bool = True,
            reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(f"unknown reduction: {reduction!r}")
        self.eta = float(eta)
        self.as_loss = as_loss
        self.reduction = reduction
        kx, ky = self._kernels(grad_method)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    @staticmethod
    def _kernels(method: str) -> tuple[torch.Tensor, torch.Tensor]:
        if method == "central":
            # central finite difference: (right - left) / 2
            kx = torch.tensor(
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]
            )
        elif method == "sobel":
            # /8 so a unit-slope ramp -> unit gradient, matching 'central' scale
            kx = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ) / 8.0
        else:
            raise ValueError(f"unknown grad_method: {method!r}")
        ky = kx.t().contiguous()  # orthogonal axis; choice is irrelevant to (.)^2
        return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)

    def _grad(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # per-channel gradients via grouped conv; replicate pad avoids edge spikes
        c = img.shape[1]
        kx = self.kx.expand(c, 1, 3, 3).to(img.dtype)
        ky = self.ky.expand(c, 1, 3, 3).to(img.dtype)
        img = F.pad(img, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(img, kx, groups=c)
        gy = F.conv2d(img, ky, groups=c)
        return gx, gy

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gx1, gy1 = self._grad(x1)
        gx2, gy2 = self._grad(x2)

        # |grad|_eta = sqrt(|grad|^2 + eta^2); flat regions -> ~eta -> near-zero
        # normalized vectors -> negligible contribution to the sum, by design.
        n1 = torch.sqrt(gx1 * gx1 + gy1 * gy1 + self.eta ** 2)
        n2 = torch.sqrt(gx2 * gx2 + gy2 * gy2 + self.eta ** 2)

        # per-pixel dot product of the two normalized gradient fields
        dot = (gx1 * gx2 + gy1 * gy2) / (n1 * n2)

        # square -> sign-invariant (survives local contrast inversion)
        ngf = (dot * dot).mean(dim=(1, 2, 3))  # per-sample, in [0, 1]

        out = 1.0 - ngf if self.as_loss else ngf
        return out.mean() if self.reduction == "mean" else out


