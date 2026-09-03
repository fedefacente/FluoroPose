import torch
def max_translation(pts_vox, wall_voxels, zooms, dirs, perp_tol_mm=3.0):
    """
    (K,) distance in mm from each landmark to the aorta wall along dirs.
    inf where the ray misses the wall.
    """
    pts_mm  = pts_vox * zooms
    wall_mm = wall_voxels * zooms

    v = wall_mm.unsqueeze(0) - pts_mm.unsqueeze(1)          # (K, V, 3)
    along = (v * dirs.unsqueeze(1)).sum(-1)                 # (K, V) signed distance along the ray
    perp  = (v - along.unsqueeze(-1) * dirs.unsqueeze(1)).norm(dim=-1)  # (K, V) offset from the ray

    hit = (along > 0) & (perp < perp_tol_mm)                # in front, and close enough to the ray
    return torch.where(hit, along, along.new_full((), float('inf'))).amin(dim=1)

def outward_dirs(pts_vox, zooms, jitter_deg=0.0):
    """(K, 3) unit vectors pointing at the outer wall, in mm-space.
    jitter_deg: random rotation around the tangent, one angle for the whole polyline."""
    pts_mm = pts_vox * zooms

    t = torch.zeros_like(pts_mm)
    t[1:-1] = pts_mm[2:] - pts_mm[:-2]
    t[0], t[-1] = pts_mm[1] - pts_mm[0], pts_mm[-1] - pts_mm[-2]
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-6)

    c = pts_mm.mean(dim=0)
    d = pts_mm - c
    d = d - (d * t).sum(1, keepdim=True) * t
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)

    if jitter_deg > 0:
        # Second basis vector of the perpendicular disc
        b = torch.linalg.cross(t, d)
        b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-6)

        ang = (torch.rand(1, device=d.device).item() * 2 - 1) * jitter_deg * math.pi / 180.0
        d = d * math.cos(ang) + b * math.sin(ang)
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)

    return d

def classify_aorta_landmarks(pts_vox: torch.Tensor,
                             zooms: torch.Tensor,
                             vert_thresh: float = 0.6):
    """
    pts_vox: (N, 3) ordered landmarks (tip -> entry).
    Returns boolean masks (asc, arch, desc), each (N,).
    """
    pts_mm = pts_vox * zooms

    # Central-difference tangents
    t = torch.zeros_like(pts_mm)
    t[1:-1] = pts_mm[2:] - pts_mm[:-2]
    t[0]    = pts_mm[1] - pts_mm[0]
    t[-1]   = pts_mm[-1] - pts_mm[-2]
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-6)

    verticality = t[:, 2].abs()            # ~1 in the limbs, ~0 at the apex
    arch = verticality < vert_thresh       # tangent mostly horizontal

    apex_idx = pts_mm[:, 2].argmax()
    idx = torch.arange(pts_vox.shape[0], device=pts_vox.device)

    asc  = (idx <= apex_idx) & ~arch       # limb containing the tip
    desc = (idx >  apex_idx) & ~arch
    return asc, arch, desc

def catmull_rom_spline(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Vectorized Catmull-Rom spline without Python loops."""
    p = torch.cat([points[:1], points, points[-1:]], dim=0)  # (N+2, D)
    n_segs = len(points) - 1
    samps_per = max(n_samples // n_segs, 2)

    # Stack control points for all segments: shape (n_segs, 4, D)
    p_stack = torch.stack([p[:-3], p[1:-2], p[2:-1], p[3:]], dim=1)

    t = torch.linspace(0, 1, samps_per, dtype=points.dtype, device=points.device)
    t2 = t ** 2
    t3 = t ** 3

    # Basis matrix: shape (samps_per, 4)
    c = 0.5 * torch.stack([
        -t3 + 2 * t2 - t,      # p0
        3 * t3 - 5 * t2 + 2,   # p1
        -3 * t3 + 4 * t2 + t,  # p2
        t3 - t2                # p3
    ], dim=1)

    # Multiply basis coefficients by control points
    # c: (samps_per, 4)
    # p_stack: (n_segs, 4, D)
    # output: (n_segs, samps_per, D)
    segments = torch.einsum('si,nid->nsd', c, p_stack)

    # Flatten out the segments
    return segments.reshape(-1, p.shape[-1])


def apply_stiffness(points: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """
    Applies Taubin smoothing to control points.
    Smooths zig-zags without shrinking the overall length or size of the curve.
    """
    p = points.clone().float()

    # Taubin parameters (mu must be slightly larger negatively than lambda is positively)
    lambda_ = 0.5
    mu_ = -0.53

    for _ in range(iterations):
        # 1. Shrink step (smooths out the bumps)
        laplacian = (p[:-2] + p[2:]) - 2 * p[1:-1]
        p[1:-1] = p[1:-1] + lambda_ * 0.5 * laplacian

        # 2. Expand step (pushes the curve back out to its original size)
        laplacian = (p[:-2] + p[2:]) - 2 * p[1:-1]
        p[1:-1] = p[1:-1] + mu_ * 0.5 * laplacian

    return p


import torch
import math


def generate_valve(
        height=45.0,
        r_bottom=13.0,
        r_waist=10.0,
        r_top=16.0,
        waist_z_ratio=0.35,
        res_z=5,
        res_theta=10
):
    """
    Generates a 3D point cloud grid of a flared prosthetic valve.

    Args:
        height: Total height of the valve (e.g., mm).
        r_bottom: Radius at the inflow (ventricular end).
        r_waist: Radius at the narrowest point.
        r_top: Radius at the outflow (aortic end).
        waist_z_ratio: Where the waist occurs as a fraction of total height (0.0 to 1.0).
        res_z: Number of points along the height.
        res_theta: Number of points around the circumference.

    Returns:
        points: Tensor of shape (res_z * res_theta, 3) containing local coordinates.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Create the Z-axis steps (from 0 to height)
    z = torch.linspace(0, height, res_z, device=device)
    waist_z = height * waist_z_ratio

    # 2. Compute the variable Radius R(z)
    # We use two quadratic curves: one from bottom to waist, one from waist to top
    r_z = torch.zeros_like(z)

    # Lower section (Bottom to Waist)
    mask_lower = z <= waist_z
    if mask_lower.any():
        # Normalized t from 0 to 1
        t_lower = z[mask_lower] / waist_z
        # Parabolic interpolation ensuring derivative is 0 at the waist
        r_z[mask_lower] = r_bottom - (r_bottom - r_waist) * (2 * t_lower - t_lower ** 2)

    # Upper section (Waist to Top)
    mask_upper = z > waist_z
    if mask_upper.any():
        # Normalized t from 0 to 1
        t_upper = (z[mask_upper] - waist_z) / (height - waist_z)
        # Parabolic interpolation ensuring derivative is 0 at the waist
        r_z[mask_upper] = r_waist + (r_top - r_waist) * (t_upper ** 2)

    # 3. Create angular steps
    theta = torch.linspace(0, 2 * math.pi, res_theta, device=device)

    # 4. Generate the full 3D grid using meshgrid
    # theta_grid, z_grid will both be shape (res_z, res_theta)
    theta_grid, z_grid = torch.meshgrid(theta, z, indexing='xy')
    r_grid = r_z.repeat(res_theta, 1).T  # Transpose to match (res_z, res_theta)

    # 5. Convert cylindrical to Cartesian coordinates
    x = r_grid * torch.cos(theta_grid)
    y = r_grid * torch.sin(theta_grid)

    # 6. Flatten to a list of 3D points
    points = torch.stack([x.flatten(), y.flatten(), z_grid.flatten()], dim=1)

    device = points.device
    points_per_strut = 10
    # 1. Reshape the flat list back into the 2D grid structure
    P = points.view(res_z, res_theta, 3)
    P_start = P[:-1, :, :]  # (res_z-1, res_theta, 3)

    # 2. Get BOTH target diagonals
    P_target_right = torch.roll(P[1:, :, :], shifts=-1, dims=1)
    P_target_left = torch.roll(P[1:, :, :], shifts=1, dims=1)

    # 3. Interpolate BOTH directions (No mask needed!)
    t = torch.linspace(0, 1, points_per_strut, device=device).view(-1, 1, 1, 1)

    P_start_exp = P_start.unsqueeze(0)

    # Generate lines for both directions
    lines_right = P_start_exp + t * (P_target_right.unsqueeze(0) - P_start_exp)
    lines_left = P_start_exp + t * (P_target_left.unsqueeze(0) - P_start_exp)

    # 4. Flatten and combine
    # Now you have the original nodes + the full set of criss-crossing struts
    all_points = torch.cat([
        points,
        lines_right.reshape(-1, 3),
        lines_left.reshape(-1, 3)
    ], dim=0)

    all_points = torch.unique(all_points, dim=0)
    return all_points


def simulate_catheter(volume_tensor, points_tensor,affine,spacing, final_volume,ventricle):
    CATHETER_RADIUS_MM = (0.5 + torch.rand(1) * (0.5 - 0.3)).item()
    CATHETER_INTENSITY = 1
    N_SPLINE_SAMPLES = 10000
    CATHETER_RADIUS_MM2 = (2.0 + torch.rand(1) * (4.0 - 2.0)).item()
    CATHETER_INTENSITY2 = 2
    VALVE_INTENSITY = 3
    VALVE_THICKNESS_MM = (0.3 + torch.rand(1) * (0.5 - 0.3)).item()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    shape = torch.tensor(volume_tensor.squeeze(0).shape, dtype=torch.long, device=DEVICE)
    zooms = torch.tensor(spacing, dtype=torch.float32, device=DEVICE)
    pigtail = False
    outer_catheter = False
    final_volume = torch.zeros_like(final_volume)
    # ── RAS → voxel ───────────────────────────────────────────────────────────────

    landmarks_ras = points_tensor.to(DEVICE)  # (N, 3)
    inv_affine = torch.linalg.inv(affine).float()

    # Homogeneous coords: (N, 4)
    B, N, _ = landmarks_ras.shape  # 1, 10, 3
    ones = torch.ones(B, N, 1, dtype=landmarks_ras.dtype, device=landmarks_ras.device)
    pts_h = torch.cat([landmarks_ras, ones], dim=-1)
    vox_points = pts_h @ inv_affine.T  # (1, 10, 4)
    vox_points = vox_points[...,:3]

    aorta_voxels = torch.nonzero(volume_tensor.squeeze(0))  # (N, 3)

    # Highest z in the volume
    z_max_volume = aorta_voxels[:, 2].max()  # scalar

    # Highest z among catheter landmarks
    z_max_points = vox_points.squeeze(0)[:, 2].max()  # scalar

    # How much to translate up (in voxels), leaving room for the tube radius
    radius_vox_z = CATHETER_RADIUS_MM*4/ zooms[2]  # radius in z voxels
    translation_z = z_max_volume - z_max_points - radius_vox_z

    # ── Wall shell (both branches need it) ────────────────────────────────────
    vol_bool = (volume_tensor.squeeze(0) == 1)
    wall_mask = vol_bool & (
            ~torch.roll(vol_bool, 1, 0) | ~torch.roll(vol_bool, -1, 0) |
            ~torch.roll(vol_bool, 1, 1) | ~torch.roll(vol_bool, -1, 1) |
            ~torch.roll(vol_bool, 1, 2) | ~torch.roll(vol_bool, -1, 2)
    )
    wall_voxels = torch.nonzero(wall_mask).float().to(DEVICE)  # (V, 3) voxel units
    wall_mm = wall_voxels * zooms

    # ── Classify + outward directions ─────────────────────────────────────────
    pts = vox_points.squeeze(0)
    asc, arch, desc = classify_aorta_landmarks(pts, zooms)
    limbs = asc | desc

    dirs = outward_dirs(pts, zooms, jitter_deg=45)
    up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE, dtype=dirs.dtype)
    dirs[arch] = up  # arch goes straight up

    dist = max_translation(pts, wall_voxels, zooms, dirs)  # (K,) mm, inf on miss
    push = (dist - CATHETER_RADIUS_MM * 2).clamp(min=0)
    push = torch.where(torch.isinf(push), torch.zeros_like(push), push)
    scale = torch.zeros_like(push)
    scale[limbs] = 0.2 + torch.rand(1, device=DEVICE).item() * 0.8
    scale[arch] = torch.rand(1, device=DEVICE).item()
    delta = dirs * (push * scale).unsqueeze(1) / zooms
    vox_points += delta.unsqueeze(0)
    snapped = vox_points.squeeze(0).clone()

    # ── Branch: with or without per-point wall snapping ───────────────────────
    if torch.rand(1).item() > 0.4:
        anchor_idx = int(snapped[ :, 2].argmax())  # highest z = top of catheter
        snapped[ arch, 2] += translation_z
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

        snapped = apply_stiffness(snapped, iterations=10)

    else:
        snapped = apply_stiffness(snapped.squeeze(0), iterations=10)

    ##### ADD pigtail
    if torch.rand(1).item() > 0.3:
        P0 = snapped.squeeze(0)[0, :]
        P1 = snapped.squeeze(0)[1, :]

        # Tangent vector pointing from the aorta DOWN into the LV
        v = P0 - P1
        v_norm = torch.linalg.norm(v)

        if v_norm > 0:
            v_unit = v / v_norm
        else:
            v_unit = torch.tensor([0.0, 0.0, 1.0], device=DEVICE, dtype=P0.dtype)

        R = torch.randint(15, 30, (1,)).item()
        ######### CALCULATE MAX L_straigth with ventricle #############
        mask = (ventricle.squeeze(0) > 0).to(DEVICE)
        shape = torch.tensor(mask.shape, device=DEVICE)

        # 1. Generate points along the ray (from tip to max_search_dist)
        steps = torch.arange(0, 150.0, 1.0, device=DEVICE)
        ray_points = P0.unsqueeze(0) + steps.unsqueeze(1) * v_unit.unsqueeze(0)
        ray_coords = torch.round(ray_points).long()
        ray_coords = torch.clamp(ray_coords,
                                 min=torch.tensor([0, 0, 0], device=DEVICE),
                                 max=shape - 1)

        # 3. Sample the mask to see which ray points actually land inside the ventricle
        is_inside_ventricle = mask[ray_coords[:, 0], ray_coords[:, 1], ray_coords[:, 2]]

        # 4. Get the indices of the points that are INSIDE
        inside_indices = torch.where(is_inside_ventricle)[0]

        # Fallback: If the ray completely misses the ventricle
        if inside_indices.numel() == 0:
            pigtail = False
        else:
                # 5. Extract the actual 3D coordinates of the points inside the ventricle
            inside_coords = ray_coords[inside_indices]

            # 6. Find the point with the lowest Z coordinate (index 2)
            min_z_idx = torch.argmin(inside_coords[:, 2])
            max_z_idx = torch.argmax(inside_coords[:, 2])

            # 7. Map that back to the distance 'L_straight'
            # The 'inside_indices' directly correlates to the 'steps' distance
            best_step_idx = inside_indices[min_z_idx]
            max_l = float(steps[best_step_idx])
            best_step_idx = inside_indices[max_z_idx]
            min_l = float(steps[best_step_idx])
            ###########
            lo = round(min(min_l, max_l))
            hi = round(max(min_l, max_l) - R)

            if torch.rand(1).item() > 0.3 or hi <= lo:
                L_straight = (max(min_l, max_l)) - R  # deterministic branch, also the fallback
            else:
                L_straight = torch.randint(lo, hi, (1,)).item()
            P_straight_end = P0 + L_straight * v_unit

            # ==========================================
            # ---  Generate Straight Segment ---
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
            theta_max = torch.randint(1, 3, (1,)).item()* math.pi
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
            snapped_pigtail = torch.cat((pigtail_points, straight_points, snapped), dim=0)
            pigtail = True

    if torch.rand(1).item() > 0.1:
        curve = catmull_rom_spline(snapped.squeeze(0), n_samples=N_SPLINE_SAMPLES)  # (S, 3)

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
        outer_catheter = True
    if torch.rand(1).item() < 0.99 and pigtail and outer_catheter:
        # =========================================================================
        # --- PROSTHETIC VALVE INTEGRATION ---
        # Insert this right before `return final_volume`
        # =========================================================================

        # 1. Configuration


        # 2. Identify the tip and the tangent vector from the high-res spline
        # curve shape is (1, S, 3). We use the first point as the deployment tip.
        # We look 10 points ahead to get a stable tangent, avoiding micro-kinks at the very edge.
        p0_vox = snapped[0, :]
        p1_vox = snapped[1, :]

        # 3. Convert to physical space (mm) for accurate orthogonal calculations
        # This prevents the valve from warping if your CT/MRI voxels are anisotropic
        p0_mm = p0_vox * zooms
        p1_mm = p1_vox * zooms

        # 4. Generate the local valve envelope
        if torch.rand(1).item() < 0.7:
            if torch.rand(1).item() < 0.5:
                valve_local_mm = generate_valve(
                    height=torch.randint(20, 35, (1,)).item(), r_bottom=torch.randint(5, 10, (1,)).item(), r_waist=torch.randint(2, 5, (1,)).item(), r_top=torch.randint(2, 5, (1,)).item(),
                    waist_z_ratio=(torch.rand(1) * 0.5 + 0.2).item()
                ).to(DEVICE)
            else:
                radius = torch.randint(2, 5, (1,)).item()
                valve_local_mm = generate_valve(
                    height=torch.randint(20, 35, (1,)).item(), r_bottom=radius,
                    r_waist=radius, r_top=radius, waist_z_ratio=(torch.rand(1) * 0.5 + 0.2).item(),
                ).to(DEVICE)
        else:
            if torch.rand(1).item() < 0.5:
                valve_local_mm = generate_valve(
                    height=torch.randint(20, 35, (1,)).item(), r_bottom=torch.randint(8, 12, (1,)).item(), r_waist=torch.randint(8, 12, (1,)).item(), r_top=torch.randint(12, 14, (1,)).item(),
                    waist_z_ratio=(torch.rand(1) * 0.5 + 0.2).item()
                ).to(DEVICE)
            else:
                radius = torch.randint(6, 10, (1,)).item()
                valve_local_mm = generate_valve(
                    height=torch.randint(20, 35, (1,)).item(), r_bottom=radius,
                    r_waist=radius, r_top=radius, waist_z_ratio=(torch.rand(1) * 0.5 + 0.2).item(),
                ).to(DEVICE)


        # 5. Compute the 3x3 Rotation Matrix in physical space
        t_vec = p1_mm - p0_mm
        t_vec = torch.nn.functional.normalize(t_vec, p=2, dim=0)

        # Define UP vector (arbitrary, adjust if patient orientation requires it)
        up_vector = torch.tensor([0.0, 1.0, 0.0], dtype=t_vec.dtype, device=DEVICE)

        # Calculate orthonormal basis
        b_vec = torch.linalg.cross(t_vec, up_vector)
        # If tangent is perfectly parallel to UP, fallback to avoid NaN
        if torch.linalg.norm(b_vec) < 1e-6:
            up_vector = torch.tensor([1.0, 0.0, 0.0], dtype=t_vec.dtype, device=DEVICE)
            b_vec = torch.linalg.cross(t_vec, up_vector)

        b_vec = torch.nn.functional.normalize(b_vec, p=2, dim=0)
        n_vec = torch.linalg.cross(b_vec, t_vec)
        n_vec = torch.nn.functional.normalize(n_vec, p=2, dim=0)

        R = torch.stack([b_vec, n_vec, t_vec], dim=1)  # Shape: (3, 3)

        # 6. Apply rotation and translate to the catheter tip
        valve_world_mm = torch.matmul(valve_local_mm, R.T) + p0_mm

        # 7. Convert back to Voxel Space
        valve_vox = valve_world_mm / zooms
        valve_centers_raw = torch.round(valve_vox).long()
        valve_centers = torch.unique(valve_centers_raw, dim=0)

        # 8. Dilate the valve wires using an anisotropic ellipsoid footprint
        radius_vox_valve = VALVE_THICKNESS_MM / zooms
        r_max_valve = int(torch.ceil(radius_vox_valve.max()).item())

        r_v = torch.arange(-r_max_valve, r_max_valve + 1, device=DEVICE, dtype=torch.float64)
        gx_v, gy_v, gz_v = torch.meshgrid(r_v, r_v, r_v, indexing='ij')

        dist_v = torch.sqrt((gx_v / radius_vox_valve[0]) ** 2 +
                            (gy_v / radius_vox_valve[1]) ** 2 +
                            (gz_v / radius_vox_valve[2]) ** 2)

        offsets_v = torch.argwhere(dist_v <= 1.0) - r_max_valve

        # 9. Stamp the valve into the final volume
        all_valve_coords = (valve_centers.unsqueeze(1) + offsets_v.unsqueeze(0)).reshape(-1, 3)

        # Bounds check to avoid writing outside the image shape
        in_bounds_v = torch.all((all_valve_coords >= 0) & (all_valve_coords < shape.unsqueeze(0)), dim=1)
        all_valve_coords = torch.unique(all_valve_coords[in_bounds_v], dim=0)

        final_volume.squeeze(0).index_put_(
            (all_valve_coords[:, 0], all_valve_coords[:, 1], all_valve_coords[:, 2]),
            torch.tensor(VALVE_INTENSITY, dtype=final_volume.dtype, device=DEVICE),
        )
        # =========================================================================

    if pigtail == True:
        snapped = snapped_pigtail
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