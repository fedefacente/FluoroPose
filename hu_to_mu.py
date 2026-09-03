import torch
import torch.nn as nn


def _to_dhw(x):
    while x.dim() > 3 and x.shape[0] == 1:
        x = x[0]
    return x


class AttenuationMapper(nn.Module):
    """HU -> mu [1/mm] with energy + instrument domain randomisation.
    """

    _CM_TO_MM = 0.1

    def __init__(self, ranges=None):
        super().__init__()
        self.register_buffer("e_kev",
                             torch.tensor([30., 40., 50., 60., 80., 100.]), persistent=False)
        self.register_buffer("mur_soft",
                             torch.tensor([0.3790, 0.2688, 0.2264, 0.2048, 0.1823, 0.1693]), persistent=False)
        self.register_buffer("mur_bone",
                             torch.tensor([1.331, 0.6655, 0.4242, 0.3148, 0.2229, 0.1855]), persistent=False)
        self.ranges = ranges or {
            "energy_kev": (30.0, 100.0),
            "bone_thresh": (150.0, 350.0),
            # Plausible device linear attenuation [1/cm], sampled per pass.
            # Material-agnostic: the exact device is not known, so mu is drawn from
            # a plausible range rather than committed to a material.
            "device_mu": {1: (3.5, 5.0), 2: (0.5, 1.5), 3: (2.0, 5.0)},
        }

    def _interp(self, e, table):
        e = e.clamp(self.e_kev[0], self.e_kev[-1])
        i = torch.searchsorted(self.e_kev, e).clamp(1, self.e_kev.numel() - 1)
        t = (e - self.e_kev[i - 1]) / (self.e_kev[i] - self.e_kev[i - 1])
        return table[i - 1] + t * (table[i] - table[i - 1])

    @torch.no_grad()
    def density(self, hu):
        hu = _to_dhw(hu.float())
        slope = 0.001029 + (hu > 0).to(hu.dtype) * (0.0005886 - 0.001029)
        return torch.clamp(1.030 + hu * slope, min=0.0)

    @torch.no_grad()
    def forward(self, hu, device_labels=None, dens=None,
                bbox=None, rng=None, out=None):
        hu = _to_dhw(hu.float())
        dev = hu.device
        dt = out.dtype if out is not None else hu.dtype
        if dens is None:
            dens = self.density(hu)

        labels = sorted(self.ranges["device_mu"].keys())
        u = torch.rand(3 + len(labels), generator=rng, device=dev)
        U = lambda i, lo, hi: lo + (hi - lo) * u[i]

        energy = U(0, *self.ranges["energy_kev"])
        bone_thresh = U(1, *self.ranges["bone_thresh"])
        keep = (u[2] > 0.1).to(dt)  # 0/1 instrument-presence dropout, as a scalar factor

        s = self._CM_TO_MM
        mur_soft = (self._interp(energy, self.mur_soft) * s).to(dt)
        mur_bone = (self._interp(energy, self.mur_bone) * s).to(dt)

        mu = out if out is not None else torch.empty_like(dens, dtype=dt)
        torch.where(hu > bone_thresh, mur_bone, mur_soft, out=mu)
        mu.mul_(dens)

        if device_labels is not None:
            dl = _to_dhw(device_labels)
            if bbox is not None:
                z0, z1, y0, y1, x0, x1 = bbox
                sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            else:
                sl = (slice(None),) * 3

            dlc = dl[sl]  # native dtype, no cast
            mu_v = mu[sl]

            for k, lab in enumerate(labels):
                lo, hi = self.ranges["device_mu"][lab]     # plausible mu [1/cm]
                mval = (U(3 + k, lo, hi) * s).to(dt)  # random attenuation, 0-d, no sync
                o = (dlc == lab).to(dt)  # binary label mask (1 float temp + transient bool)
                o.mul_(keep)             # drop all instruments with prob 0.1
                # hard insertion: mu -> mval inside the label (o in {0, 1})
                # written as mu + o*(mval - mu) to stay in place and sync-free
                mu_v.addcmul_(o, mval - mu_v)

        return mu.squeeze(0)


class AttenuationMapperRatio(nn.Module):
    """HU -> mu [1/mm] with energy + instrument domain randomisation.

    Device attenuation is defined *relative to tissue* at the sampled energy,
    rather than as an independent absolute value. Each device label draws a
    dimensionless multiplier applied to a per-pass reference tissue attenuation:

        dense/metal labels (bone-like energy response):  mu = r * bone_mu_ref
        radiolucent labels (soft-tissue-like response):  mu = r * soft_mu_ref

    Because bone_mu_ref / soft_mu_ref are evaluated at the sampled energy, the
    device attenuation is energy-coupled *by construction* and its ratio to
    tissue is controlled directly. This keeps the projected device-to-tissue
    contrast within a physically plausible band across the whole diagnostic
    energy range, and removes the unbounded contrast tail that arises when
    device mu is sampled independently of energy.

    The coupling is contrast-driven (a bounded ratio to tissue), not a
    per-material spectral model: the exact device is unknown, so its contrast
    relative to tissue is randomised rather than committed to a material.
    """

    _CM_TO_MM = 0.1
    # Reference HU at which the per-pass tissue attenuations are evaluated for
    # the device multiplier (scalar, sync-free). Representative cortical bone.
    _HU_REF = 700.0

    def __init__(self, ranges=None):
        super().__init__()
        self.register_buffer("e_kev",
                             torch.tensor([30., 40., 50., 60., 80., 100.]), persistent=False)
        self.register_buffer("mur_soft",
                             torch.tensor([0.3790, 0.2688, 0.2264, 0.2048, 0.1823, 0.1693]), persistent=False)
        self.register_buffer("mur_bone",
                             torch.tensor([1.331, 0.6655, 0.4242, 0.3148, 0.2229, 0.1855]), persistent=False)
        self.ranges = ranges or {
            "energy_kev": (30.0, 100.0),
            "bone_thresh": (150.0, 350.0),
            # Device attenuation as a dimensionless multiplier on a reference
            # tissue attenuation at the sampled energy. "base" selects which
            # tissue's energy response the device tracks:
            #   "bone" -> dense/metal instruments (high-Z, bone-like fall-off)
            #   "soft" -> radiolucent instruments (polymer/light, soft-like)
            # Ranges chosen so projected device-to-bone contrast stays within a
            # fluoroscopically plausible band (verified by Monte Carlo).
            "device_mu": {
                1: {"base": "bone", "ratio": (4.0, 9.0)},  # dense metal
                2: {"base": "soft", "ratio": (1.5, 6.0)},  # radiolucent / light
                3: {"base": "bone", "ratio": (2.5, 8.0)},  # mixed / catch-all
            },
        }

    def _interp(self, e, table):
        e = e.clamp(self.e_kev[0], self.e_kev[-1])
        i = torch.searchsorted(self.e_kev, e).clamp(1, self.e_kev.numel() - 1)
        t = (e - self.e_kev[i - 1]) / (self.e_kev[i] - self.e_kev[i - 1])
        return table[i - 1] + t * (table[i] - table[i - 1])

    @torch.no_grad()
    def density(self, hu):
        hu = _to_dhw(hu.float())
        slope = 0.001029 + (hu > 0).to(hu.dtype) * (0.0005886 - 0.001029)
        return torch.clamp(1.030 + hu * slope, min=0.0)

    @torch.no_grad()
    def forward(self, hu, device_labels=None, dens=None,
                bbox=None, rng=None, out=None):
        hu = _to_dhw(hu.float())
        dev = hu.device
        dt = out.dtype if out is not None else hu.dtype
        if dens is None:
            dens = self.density(hu)

        labels = sorted(self.ranges["device_mu"].keys())
        u = torch.rand(3 + len(labels), generator=rng, device=dev)
        U = lambda i, lo, hi: lo + (hi - lo) * u[i]

        energy = U(0, *self.ranges["energy_kev"])
        bone_thresh = U(1, *self.ranges["bone_thresh"])
        keep = (u[2] > 0.1).to(dt)  # 0/1 instrument-presence dropout, as a scalar factor

        s = self._CM_TO_MM
        soft_e = self._interp(energy, self.mur_soft)  # 1/cm, mass-atten * ref (pre-density)
        bone_e = self._interp(energy, self.mur_bone)
        mur_soft = (soft_e * s).to(dt)
        mur_bone = (bone_e * s).to(dt)

        mu = out if out is not None else torch.empty_like(dens, dtype=dt)
        torch.where(hu > bone_thresh, mur_bone, mur_soft, out=mu)
        mu.mul_(dens)

        if device_labels is not None:
            dl = _to_dhw(device_labels)
            if bbox is not None:
                z0, z1, y0, y1, x0, x1 = bbox
                sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            else:
                sl = (slice(None),) * 3

            dlc = dl[sl]  # native dtype, no cast
            mu_v = mu[sl]

            # Per-pass scalar reference tissue attenuations at the sampled
            # energy and reference HU (0-d, sync-free). Device mu is a bounded
            # multiple of these, so it tracks energy through the tissue tables.
            # density() at HU_REF (>0 -> uses the positive-HU slope).
            _slope_ref = 0.001029 + (self._HU_REF > 0) * (0.0005886 - 0.001029)
            dens_ref = max(1.030 + self._HU_REF * _slope_ref, 0.0)
            bone_mu_ref = (bone_e * s * dens_ref).to(dt)
            soft_mu_ref = (soft_e * s * dens_ref).to(dt)

            for k, lab in enumerate(labels):
                spec = self.ranges["device_mu"][lab]
                lo, hi = spec["ratio"]
                base = bone_mu_ref if spec["base"] == "bone" else soft_mu_ref
                mval = (U(3 + k, lo, hi) * base).to(dt)  # ratio * tissue mu, 0-d, no sync
                o = (dlc == lab).to(dt)  # binary label mask
                o.mul_(keep)  # drop all instruments with prob 0.1
                # hard insertion: mu -> mval inside the label (o in {0, 1})
                mu_v.addcmul_(o, mval - mu_v)

        return mu.squeeze(0)