from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage, signal
from skimage.segmentation import watershed
## logger for progress messages
log = logging.getLogger("vertebrae")
## label numbers: 1 = l5 ... 5 = l1, 6 = t12 ... 17 = t1, 18 = c7 ... 24 = c1
NAMES = ([""] + ["L5", "L4", "L3", "L2", "L1"] + [f"T{i}" for i in range(12, 0, -1)]
         + [f"C{i}" for i in range(7, 0, -1)])
ID = {n: i for i, n in enumerate(NAMES) if n}
## neighbours: s26 = touching even by a corner, s6 = touching by a face only
S26 = np.ones((3, 3, 3), dtype=bool)
S6 = ndimage.generate_binary_structure(3, 1)

## all settings in one place (sizes in mm / mm3 / cm3, ct values in hu)
CFG = {
    "bone_hu": 150.0, "speck_mm3": 30.0, "soft_bone_fraction": 0.50,
    "near_mm": 5.0, "corridor_pad_mm": 6.0, "bridge_cap_mm3": 500.0,
    "enclosure": 0.80, "max_rel_to_host": 0.5, "min_labeled_shell": 0.5,
    "suspect_size_lo": 0.60, "suspect_size_hi": 1.80, "suspect_logres": 0.30,
    "suspect_pool_frac": 0.20, "band_gap_close": 3,
    "profile_sigma_mm": 2.5, "profile_radius_mm": 10.0, "min_body_height_mm": 12.0,
    "max_body_height_mm": 45.0, "dp_beta": 2.0, "pin_lo_mm": 8.0, "pin_hi_mm": 35.0,
    "seed_margin_mm": 3.0, "seed_radius_mm": 14.0, "halo_mm": 2.5, "band_pad_mm": 4.0,
    "arch_root_mm": 3.0, "arch_neck_mm": 1.5, "arch_cost": "core",
    "arch_overhang_mm": 16.0,
    "imb_half_mm": 14.0, "imb_strip_mm": 6.0, "imb_canal_mm": 12.0,
    "imb_max_shift_cm3": 6.0,
    "polish_trigger_mm": 3.5, "polish_min_contact_mm2": 80.0, "polish_collar_mm": 6.0,
    "polish_seed_shell_mm": 2.5, "polish_max_shift_frac": 0.15,
    "polish_min_gain_mm": 0.3, "polish_priority": "hu",
    "reclaim_neck_mm": 10.0, "reclaim_link_mm": 10.0,
    "mv_neck_mm": 1.2, "mv_anchor_mm": 40.0, "mv_max_shift_frac": 0.25,
    "mv_elong": 2.5,
    "skel_cut_mm": 1.8,
    "sigma_mm": 1.2, "max_dev_mm": 1.5, "vol_tol": 0.01, "min_dice": 0.97,
    "smooth_pad_mm": 5.0,
    "crop_margin_mm": 25.0, "island_max_cm3": 2.0, "island_host_frac": 0.6,
    "hole_max_cm3": 3.0, "hole_min_hu": 100.0, "pit_max_mm3": 200.0, "pit_close_mm": 1.5,
    "max_growth": 0.03, "speck_neck_radii_mm": (1.0, 1.5), "speck_max_mm3": 150.0,
    "speck_host_frac": 0.5, "speck_thick_max_mm3": 50.0, "separate_bone_max_cm3": 2.0,
    "separate_bone_lines": 10, "speck_rounds": 4, "spine_crop_mm": 60.0,
}

## loop over every vertebra together with its (padded) bounding box
def _boxes(seg, pad=0):
    objs = ndimage.find_objects(seg)
    for l in range(1, 25):
        if l <= len(objs) and objs[l - 1] is not None:
            yield l, tuple(slice(max(s.start - pad, 0), s.stop + pad) for s in objs[l - 1])

## compare every voxel with its neighbour along x, y and z
def _face_pairs(seg):
    for ax in range(3):
        hi = tuple(slice(1, None) if k == ax else slice(None) for k in range(3))
        lo = tuple(slice(None, -1) if k == ax else slice(None) for k in range(3))
        yield ax, hi, lo, seg[hi], seg[lo]

## how many problems an audit found (lower = better)
def _bad(s):
    return s["n_fragmented"] + s["n_empty"] + s["n_size"] + s["n_order"]



## read the ai labels and the ct (same grid, ct clipped to -1024..3071 hu)
def load_case(case_dir: Path, ct_root: Path):
    seg_dir = case_dir / "segmentations"
    vfiles = sorted(seg_dir.glob("vertebrae_*.nii.gz")) if seg_dir.is_dir() else []
    if vfiles:
        img = nib.load(str(vfiles[0]))
        seg = np.zeros(img.shape, dtype=np.uint8)
        for f in vfiles:
            name = f.name[len("vertebrae_"):-len(".nii.gz")]
            if name in ID:
                seg[np.asanyarray(nib.load(str(f)).dataobj) > 0] = ID[name]
    else:
        if seg_dir.is_dir() and any(seg_dir.glob("*.nii.gz")):
            raise ValueError(
                f"{case_dir}: segmentations/ has organ masks but no vertebrae_*.nii.gz - "
                "this looks like the AbdomenAtlas organ folder. Point --pred_dir at "
                "SuPreM's vertebrae inference output instead.")
        img = nib.load(str(case_dir / "combined_labels.nii.gz"))
        seg = np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)
    if seg.max() > 24:
        raise ValueError(f"{case_dir}: label values up to {seg.max()} - not the 1..24 vertebra coding")
    ct_img = nib.load(str(ct_root / case_dir.name / "ct.nii.gz"))
    ct = np.clip(np.asanyarray(ct_img.dataobj), -1024, 3071).astype(np.int16)
    if ct.shape != seg.shape:
        raise ValueError(f"{case_dir.name}: CT {ct.shape} and labels {seg.shape} differ")
    return img, seg, ct

## write combined_labels.nii.gz + one file per vertebra in segmentations/
def save_case(out_dir: Path, case: str, seg: np.ndarray, img):
    d = out_dir / case
    (d / "segmentations").mkdir(parents=True, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), img.affine, hdr), str(d / "combined_labels.nii.gz"))
    for lid in range(1, 25):
        nib.save(nib.Nifti1Image((seg == lid).astype(np.uint8), img.affine, hdr),
                 str(d / "segmentations" / f"vertebrae_{NAMES[lid]}.nii.gz"))

## box around the main spine, ignoring far away stray blobs
def spine_bbox(seg, f=4):
    small = np.zeros(tuple(-(-n // f) for n in seg.shape), dtype=bool)
    for a in range(f):
        for b in range(f):
            for c in range(f):
                v = seg[a::f, b::f, c::f] > 0
                small[:v.shape[0], :v.shape[1], :v.shape[2]] |= v
    small = ndimage.binary_dilation(small, structure=S26, iterations=2)
    lab, n = ndimage.label(small, structure=S26)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.nonzero(sizes >= 0.05 * sizes.max())[0]
    pts = np.argwhere(np.isin(lab, keep))
    return (np.maximum(pts.min(axis=0) * f - f, 0),
            np.minimum(pts.max(axis=0) * f + 2 * f, np.array(seg.shape) - 1))

## split a mask into pieces (corners count as touching) + their sizes
def components(mask):
    lab, n = ndimage.label(mask, structure=S26)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab, sizes

## vertebra names from c1 down to l5
TOP_DOWN = NAMES[:0:-1]

## grow a box by a few voxels, clipped to the image
def _bbox_pad(sl, pad_vox, shape):
    return tuple(slice(max(s.start - int(p), 0), min(s.stop + int(p), n))
                 for s, p, n in zip(sl, pad_vox, shape))

## same as components(), but with cc3d - its piece order is kept on purpose
def _components(mask):
    cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    counts = np.bincount(cc.ravel())
    counts[0] = 0
    return cc, counts

## per vertebra: broken into pieces? too small? out of order?
## the counts decide whether a step is kept or undone
def audit(seg, affine, vox_mm3):
    rows = []
    objs = ndimage.find_objects(seg)
    for name in TOP_DOWN:
        lid = ID[name]
        sl = objs[lid - 1] if lid - 1 < len(objs) else None
        if sl is None:
            rows.append({"name": f"vertebrae_{name}", "voxels": 0, "volume_cm3": 0.0,
                         "components": 0, "largest_fraction": 0.0, "flags": ["EMPTY"]})
            continue
        mask = seg[sl] == lid
        vox = int(mask.sum())
        _, counts = _components(mask)
        com = ndimage.center_of_mass(mask)
        w = affine @ np.array([com[0] + sl[0].start, com[1] + sl[1].start,
                               com[2] + sl[2].start, 1.0])
        ncomp = int((counts > 0).sum())
        rows.append({"name": f"vertebrae_{name}", "voxels": vox,
                     "volume_cm3": round(vox * vox_mm3 / 1000.0, 1),
                     "components": ncomp,
                     "largest_fraction": round(float(counts.max()) / vox, 4),
                     "si_world": float(w[2]),
                     "flags": [f"FRAGMENTED({ncomp})"] if ncomp > 1 else []})
    si = [(r["name"], r["si_world"]) for r in rows if "si_world" in r]
    if len(si) >= 3:
        dec = si[-1][1] < si[0][1]
        for i in range(1, len(si)):
            step = si[i][1] - si[i - 1][1]
            if (dec and step > 0) or (not dec and step < 0):
                next(r for r in rows if r["name"] == si[i][0])["flags"].append("ORDER")
    present = [r for r in rows if r["voxels"] > 0]
    for i in range(1, len(present) - 1):
        smaller = min(present[i - 1]["voxels"], present[i + 1]["voxels"])
        if smaller > 0:
            ratio = round(present[i]["voxels"] / smaller, 2)
            present[i]["size_ratio_vs_neighbors"] = ratio
            if ratio < 0.6:
                present[i]["flags"].append(f"SIZE({ratio})")
    summary = {k: sum(1 for r in rows if any(f.startswith(pre) for f in r["flags"]))
               for k, pre in [("n_fragmented", "FRAG"), ("n_empty", "EMPTY"),
                              ("n_size", "SIZE"), ("n_order", "ORDER")]}
    return rows, summary

## step 1: loose pieces - tiny -> delete, not bone -> delete,
## bone close by -> reconnect, bone far away -> set aside for later
def stage1_triage(seg, ct, zooms, vox_mm3, records):
    out = seg.copy()
    bone = ct >= CFG["bone_hu"]
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        main = cc == main_id
        union_sl = ndimage.find_objects((cc > 0).astype(np.uint8))[0]
        pad_vox = np.ceil(CFG["near_mm"] / np.asarray(zooms)) + 2
        union_sl = _bbox_pad(union_sl, pad_vox, seg.shape)
        dist_main = ndimage.distance_transform_edt(~main[union_sl], sampling=zooms)
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id:
                continue
            comp = cc == comp_id
            vol = float(counts[comp_id]) * vox_mm3
            hu = ct[comp]
            rec = {"stage": 1, "label": NAMES[lid], "vol_mm3": round(vol, 1),
                   "hu_mean": round(float(hu.mean()), 1),
                   "bone_fraction": round(float((hu >= CFG["bone_hu"]).mean()), 3)}
            if vol < CFG["speck_mm3"]:
                out[comp] = 0
                rec["action"] = "REMOVED_SPECK"
            elif rec["bone_fraction"] <= CFG["soft_bone_fraction"]:
                out[comp] = 0
                rec["action"] = "REMOVED_SOFT"
            else:
                d = float(dist_main[comp[union_sl]].min())
                rec["dist_to_main_mm"] = round(d, 2)
                if d <= CFG["near_mm"]:
                    added = _bridge(out, bone, main, comp, lid, zooms)
                    if added is not None:
                        out[comp] = lid
                        out[added] = lid
                        rec["action"] = "BRIDGED"
                        rec["bridge_added_mm3"] = round(float(added.sum()) * vox_mm3, 1)
                    else:
                        rec["action"] = "KEPT_POSSIBLE_FRAGMENT"
                else:
                    out[comp] = 0
                    rec["action"] = "REMOVED_FAR_REASSIGN_CANDIDATE"
            records.append(rec)
    return out

## reconnect a piece to its vertebra through a short path of ct bone
def _bridge(seg, bone, main, comp, lid, zooms, neck_mm=None):
    neck_mm = CFG["near_mm"] if neck_mm is None else neck_mm
    nz_c = ndimage.find_objects(comp.astype(np.uint8))[0]
    nz_m = ndimage.find_objects(main.astype(np.uint8))[0]
    joint = tuple(slice(min(a.start, b.start), max(a.stop, b.stop))
                  for a, b in zip(nz_c, nz_m))
    pad_vox = np.ceil(CFG["corridor_pad_mm"] / np.asarray(zooms)) + 1
    joint = _bbox_pad(joint, pad_vox, seg.shape)
    seg_j, bone_j = seg[joint], bone[joint]
    comp_j, main_j = comp[joint], main[joint]
    d_comp = ndimage.distance_transform_edt(~comp_j, sampling=zooms)
    d_main = ndimage.distance_transform_edt(~main_j, sampling=zooms)
    neck = (d_comp <= neck_mm) & (d_main <= neck_mm)
    domain = (bone_j & ((seg_j == 0) | (seg_j == lid)) & neck) | comp_j | main_j
    reach = ndimage.binary_propagation(main_j, mask=domain, structure=S26)
    if not (reach & comp_j).any():
        return None
    fill = reach & (seg_j == 0) & bone_j & neck
    if float(fill.sum()) * abs(np.prod(zooms)) > CFG["bridge_cap_mm3"]:
        return None
    corridor = np.zeros_like(comp)
    corridor[joint] = fill
    return corridor

## a piece mostly wrapped by another vertebra takes that vertebra's label
def stage2a_islands(seg, vox_mm3, records):
    out = seg.copy()
    vol = {int(l): int((seg == l).sum()) for l in np.unique(seg) if l != 0}
    for lid in sorted(vol):
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id:
                continue
            comp = cc == comp_id
            sl = _bbox_pad(ndimage.find_objects(comp.astype(np.uint8))[0],
                           np.array([2, 2, 2]), seg.shape)
            comp_l = comp[sl]
            shell = ndimage.binary_dilation(comp_l, structure=S26) & ~comp_l
            neigh = out[sl][shell]
            labeled = neigh[neigh != 0]
            if neigh.size == 0 or labeled.size / neigh.size < CFG["min_labeled_shell"]:
                continue
            ids, cnt = np.unique(labeled, return_counts=True)
            host = int(ids[cnt.argmax()])
            frac = float(cnt.max()) / labeled.size
            rec = {"stage": "2a", "label": NAMES[lid], "host": NAMES[host],
                   "vol_mm3": round(float(counts[comp_id]) * vox_mm3, 1),
                   "enclosure": round(frac, 3)}
            if (host != lid and frac >= CFG["enclosure"]
                    and counts[comp_id] <= CFG["max_rel_to_host"] * vol.get(host, 0)):
                out[comp] = host
                rec["action"] = "ISLAND_REASSIGNED"
            else:
                rec["action"] = "ISLAND_FLAG_ONLY"
            records.append(rec)
    return out

## vertebrae that look wrong (broken, odd size, lost a lot of bone) - only these get redrawn
def find_suspects(seg, raw, affine, vox_mm3):
    rows, _ = audit(seg, affine, vox_mm3)
    info = {r["name"].replace("vertebrae_", ""): r for r in rows if r["voxels"] > 0}
    order = [n for n in TOP_DOWN if n in info]
    suspects = set()
    logv = {n: np.log(max(info[n]["voxels"], 1)) for n in order}
    for i, n in enumerate(order):
        w = [logv[order[j]] for j in range(max(0, i - 2), min(len(order), i + 3)) if j != i]
        if w and abs(logv[n] - float(np.median(w))) > CFG["suspect_logres"]:
            suspects.add(n)
    for n, r in info.items():
        ratio = r.get("size_ratio_vs_neighbors")
        if ratio is not None and (ratio < CFG["suspect_size_lo"] or ratio > CFG["suspect_size_hi"]):
            suspects.add(n)
        if r["components"] > 1 or "ORDER" in r["flags"]:
            suspects.add(n)
    for lid in np.unique(raw):
        if lid == 0:
            continue
        raw_n = int((raw == lid).sum())
        pool_n = int(((raw == lid) & (seg == 0)).sum())
        if raw_n and pool_n / raw_n >= CFG["suspect_pool_frac"]:
            suspects.add(NAMES[int(lid)])
    idx = {n: i for i, n in enumerate(order)}
    s_idx = sorted(idx[n] for n in suspects if n in idx)
    bands = []
    for i in s_idx:
        if bands and i - bands[-1][-1] <= CFG["band_gap_close"] + 1:
            bands[-1] = list(range(bands[-1][0], i + 1))
        else:
            bands.append([i])
    return [[order[j] for j in range(b[0], b[-1] + 1)] for b in bands], sorted(suspects), order

## ct brightness on disks across the spine - every disc shows up as a dip
def column_profile(seg, raw, ct, zooms):
    bone = ct >= CFG["bone_hu"]
    M = (seg > 0) | ((raw > 0) & bone)
    Mb = M & bone
    nz = seg.shape[2]
    cx = np.full(nz, np.nan); cy = np.full(nz, np.nan)
    for z in np.nonzero(M.any(axis=(0, 1)))[0]:
        sl2 = Mb[:, :, z] if Mb[:, :, z].any() else M[:, :, z]
        lab, n = ndimage.label(sl2)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        pts = np.nonzero(lab == sizes.argmax())
        cx[z], cy[z] = pts[0].mean(), pts[1].mean()
    w = max(int(40.0 / zooms[2]) | 1, 5)
    ker = np.ones(w) / w
    ok = ~np.isnan(cx)
    cxs, cys = cx.copy(), cy.copy()
    cxs[ok] = np.convolve(np.pad(cx[ok], w // 2, mode="edge"), ker, "valid")
    cys[ok] = np.convolve(np.pad(cy[ok], w // 2, mode="edge"), ker, "valid")
    zmm = np.asarray(zooms)
    Pmm = np.c_[cxs * zmm[0], cys * zmm[1], np.arange(nz) * zmm[2]]
    idx = np.nonzero(ok)[0]
    if idx.size < 5:
        return cxs, cys, np.zeros(nz)
    T = np.zeros_like(Pmm)
    grad = np.gradient(ndimage.gaussian_filter1d(Pmm[idx], sigma=8.0 / zmm[2], axis=0), axis=0)
    grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-6)
    T[idx] = grad
    r = CFG["profile_radius_mm"]
    aa, bb = np.meshgrid(np.arange(-r, r + 1.2, 1.2), np.arange(-r, r + 1.2, 1.2))
    keep = aa ** 2 + bb ** 2 <= r ** 2
    offs = np.c_[aa[keep], bb[keep]]
    rho = np.full(nz, np.nan)
    hu = ct.astype(np.float32)
    for z in idx:
        t = T[z]
        ref = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0.0, 1, 0])
        u = np.cross(t, ref); u /= np.linalg.norm(u)
        v = np.cross(t, u)
        pts_mm = Pmm[z] + offs[:, :1] * u + offs[:, 1:2] * v
        vals = ndimage.map_coordinates(hu, (pts_mm / zmm).T, order=1, mode="nearest")
        rho[z] = float(np.clip(vals, 0, 800).mean())
    rho_s = ndimage.gaussian_filter1d(np.where(np.isnan(rho), 0.0, rho),
                                      sigma=max(CFG["profile_sigma_mm"] / zooms[2], 0.8))
    return cxs, cys, rho_s

## pick the expected number of disc dips, keeping vertebra heights even
def disc_minima(rho_s, z_lo, z_hi, n_cuts, zooms, pin_lo, pin_hi):
    seg_rho = rho_s[z_lo:z_hi]
    if seg_rho.size < 5 or seg_rho.max() <= 0 or n_cuts <= 0:
        return [] if n_cuts == 0 else None
    dz = zooms[2]
    peaks, props = signal.find_peaks(-seg_rho, distance=max(int(6.0 / dz), 2),
                                     prominence=1e-6)
    if peaks.size < n_cuts:
        return None
    z = peaks.astype(float) * dz
    p = props["prominences"] / props["prominences"].max()
    m, K = peaks.size, n_cuts
    min_h, max_h = CFG["min_body_height_mm"], CFG["max_body_height_mm"]
    span = (z_hi - z_lo) * dz
    NEG = -1e18
    first_ok = (z >= CFG["pin_lo_mm"]) & (z <= CFG["pin_hi_mm"]) if pin_lo else np.ones(m, bool)
    last_ok = ((z >= span - CFG["pin_hi_mm"]) & (z <= span - CFG["pin_lo_mm"])
               if pin_hi else np.ones(m, bool))
    if not first_ok.any() or not last_ok.any():
        return None
    dp = np.full((K, m, m), NEG)
    bk = np.zeros((K, m, m), dtype=np.int32)
    for j in range(m):
        if first_ok[j]:
            dp[0, j, :] = p[j]
    for k in range(1, K):
        for j in range(m):
            for i in range(j):
                h = z[j] - z[i]
                if not (min_h <= h <= max_h):
                    continue
                if k == 1:
                    best_prev, best_i2 = dp[0, i, 0], 0
                else:
                    best_prev, best_i2 = NEG, 0
                    for i2 in range(i):
                        h_prev = z[i] - z[i2]
                        if not (min_h <= h_prev <= max_h):
                            continue
                        s = dp[k - 1, i, i2] - CFG["dp_beta"] * np.log(h / h_prev) ** 2
                        if s > best_prev:
                            best_prev, best_i2 = s, i2
                if best_prev > NEG / 2 and best_prev + p[j] > dp[k, j, i]:
                    dp[k, j, i] = best_prev + p[j]
                    bk[k, j, i] = best_i2
    flat = dp[K - 1].copy()
    flat[~last_ok, :] = NEG
    if flat.max() <= NEG / 2:
        return None
    j, i = np.unravel_index(int(flat.argmax()), flat.shape)
    picks = [j]
    k = K - 1
    while k >= 1:
        picks.append(i)
        j, i, k = i, int(bk[k, j, i]), k - 1
    return sorted(int(peaks[q]) + z_lo for q in set(picks))

## regrow the back part (arch) of each vertebra from its two pedicles
def arch_repartition(labels_ws, body_zone, A, band_ids, zooms, vox_mm3):
    body_lab = np.where(body_zone, labels_ws, 0).astype(np.int32)
    roots = np.zeros(A.shape, dtype=np.int32)
    contested = np.zeros(A.shape, dtype=bool)
    pad_vox = np.ceil((CFG["arch_root_mm"] + 2.0) / np.asarray(zooms)).astype(int) + 1
    root_it = max(int(np.ceil(CFG["arch_root_mm"] / min(zooms))), 1)
    rootless = []
    for lid in band_ids:
        body = body_lab == lid
        if not body.any():
            rootless.append(NAMES[lid])
            continue
        sl = _bbox_pad(ndimage.find_objects(body.astype(np.uint8))[0],
                       pad_vox, A.shape)
        reach = ndimage.binary_dilation(body[sl], structure=S6,
                                        iterations=root_it,
                                        mask=A[sl] | body[sl])
        d_eu = ndimage.distance_transform_edt(~body[sl], sampling=zooms)
        r = reach & A[sl] & (d_eu <= CFG["arch_root_mm"] + max(zooms))
        if not r.any():
            rootless.append(NAMES[lid])
            continue
        sub = roots[sl]
        contested[sl] |= (sub > 0) & (sub != lid) & r
        sub[r & (sub == 0)] = lid
    roots[contested] = 0
    rec = {"arch_mode": f"pedicle_roots_{CFG['arch_cost']}",
           "arch_roots_cm3": {NAMES[l]: round(float((roots == l).sum())
                                                   * vox_mm3 / 1000.0, 2)
                              for l in band_ids if (roots == l).any()},
           "arch_contested_mm3": round(float(contested.sum()) * vox_mm3, 1)}
    if rootless:
        rec["arch_rootless"] = rootless
    if not (roots > 0).any():
        rec["arch_skipped"] = "no pedicle roots found"
        return labels_ws, rec
    if CFG["arch_cost"] == "hier":
        bb = _bbox_pad(ndimage.find_objects(A.astype(np.uint8))[0],
                       np.array([2, 2, 2]), A.shape)
        Ab = A[bb]
        solid = Ab | body_zone[bb]
        t2 = np.zeros(Ab.shape, dtype=np.float32)
        zo2 = (zooms[1], zooms[2])
        for x in range(Ab.shape[0]):
            sl2 = solid[x]
            if sl2.any():
                t2[x] = ndimage.distance_transform_edt(sl2, sampling=zo2)
        lab = roots[bb].astype(np.int32)
        tmax = float(t2[Ab].max()) if Ab.any() else 0.0
        tiers = [t for t in (8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.6,
                             1.2, 0.8, 0.4) if t < tmax] + [0.0]
        for t in tiers:
            mask_t = (Ab & (t2 >= t)) | (lab > 0)
            lab = watershed(np.zeros(Ab.shape, dtype=np.uint8), markers=lab,
                            mask=mask_t)
        labels_arch = np.zeros(A.shape, dtype=np.int32)
        labels_arch[bb] = lab
    elif CFG["arch_cost"] == "core":
        edtA = ndimage.distance_transform_edt(A, sampling=zooms).astype(np.float32)
        core = A & ((edtA >= CFG["arch_neck_mm"]) | (roots > 0))
        lab1 = watershed(np.zeros(A.shape, dtype=np.uint8), markers=roots,
                         mask=core)
        labels_arch = watershed(np.zeros(A.shape, dtype=np.uint8), markers=lab1,
                                mask=A)
    elif CFG["arch_cost"] == "edt":
        edtA = ndimage.distance_transform_edt(A, sampling=zooms).astype(np.float32)
        labels_arch = watershed(-edtA, markers=roots, mask=A)
    else:
        labels_arch = watershed(np.zeros(A.shape, dtype=np.uint8),
                                markers=roots, mask=A)
    sel = labels_arch > 0
    changed = int((sel & (labels_arch != labels_ws)).sum())
    out = np.where(sel, labels_arch.astype(labels_ws.dtype), labels_ws)
    rec["arch_phase2_changed_mm3"] = round(changed * vox_mm3, 1)
    rec["arch_unrooted_mm3"] = round(float((A & ~sel).sum()) * vox_mm3, 1)
    return out, rec

## redraw the borders of one suspicious stretch: each vertebra grows through bone,
## they meet in the dark gaps (the discs)
def arbitrate_band(seg, raw, ct, band, neighbors, cxs, cys, rho_s, zooms,
                   vox_mm3, qa):
    bone = ct >= CFG["bone_hu"]
    band_ids = [ID[n] for n in band]
    band_mask = np.isin(seg, band_ids)
    pool = (np.isin(raw, band_ids)) & (seg == 0) & bone
    zsel = np.nonzero((band_mask | pool).any(axis=(0, 1)))[0]
    pad = int(CFG["band_pad_mm"] / zooms[2]) + 1
    z0 = max(int(zsel[0]) - pad, 0)
    z1 = min(int(zsel[-1]) + pad + 1, seg.shape[2])
    rec = {"levels": band, "z_range_crop": [int(z0), int(z1)]}
    qa["bands"].append(rec)
    def centroid_z(name):
        zs = np.nonzero((seg == ID[name]).any(axis=(0, 1)))[0]
        return float(zs.mean()) if zs.size else None
    ups = sorted((ID[n], centroid_z(n)) for n in band if centroid_z(n) is not None)
    z_up = True if len(ups) < 2 else ups[-1][1] >= ups[0][1]
    rec["z_increases_toward_head"] = bool(z_up)
    lo_name, hi_name = (neighbors if z_up else neighbors[::-1])
    w_lo = centroid_z(lo_name) if lo_name else None
    w_hi = centroid_z(hi_name) if hi_name else None
    win_lo = int(w_lo) if w_lo is not None else int(zsel[0]) + int(4.0 / zooms[2])
    win_hi = int(w_hi) if w_hi is not None else int(zsel[-1]) - int(4.0 / zooms[2])
    n_expect = (len(band_ids) - 1) + (w_lo is not None) + (w_hi is not None)
    found = disc_minima(rho_s, win_lo, win_hi, n_expect, zooms,
                        w_lo is not None, w_hi is not None)
    if found is None:
        qa["flags"].append(f"DISC_COUNT_UNRESOLVED band={band}: flag only, no edit")
        rec["skipped"] = True
        return None
    lo_cut = found.pop(0) if w_lo is not None else z0
    hi_cut = found.pop(-1) if w_hi is not None else z1
    cuts_z = [int(lo_cut)] + [int(f) for f in found] + [int(hi_cut)]
    rec["disc_cuts_z"] = cuts_z
    zmm = np.asarray(zooms, dtype=np.float64)
    nzc = seg.shape[2]
    okc = ~np.isnan(cxs)
    Pmm = np.c_[np.where(okc, cxs, 0) * zmm[0], np.where(okc, cys, 0) * zmm[1],
                np.arange(nzc) * zmm[2]]
    T = np.zeros_like(Pmm)
    idxc = np.nonzero(okc)[0]
    g = np.gradient(ndimage.gaussian_filter1d(Pmm[idxc], sigma=8.0 / zmm[2], axis=0), axis=0)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    T[idxc] = g
    step_len = np.zeros(nzc)
    step_len[idxc[1:]] = np.linalg.norm(np.diff(Pmm[idxc], axis=0), axis=1)
    s_axis = np.cumsum(step_len)
    s_cuts = [float(s_axis[c]) for c in cuts_z]
    tilt_pad = int(10.0 / zooms[2])
    z0e, z1e = max(cuts_z[0] - tilt_pad, 0), min(cuts_z[-1] + tilt_pad + 1, nzc)
    slab = slice(z0e, z1e)
    seg_s, bone_s = seg[:, :, slab], bone[:, :, slab]
    band_s, pool_s = band_mask[:, :, slab], pool[:, :, slab]
    xx, yy = np.meshgrid(np.arange(seg.shape[0], dtype=np.float32),
                         np.arange(seg.shape[1], dtype=np.float32), indexing="ij")
    s_f = np.empty(seg_s.shape, dtype=np.float32)
    for z in range(z0e, z1e):
        dx = (xx - np.float32(cxs[z] if okc[z] else 0)) * np.float32(zmm[0])
        dy = (yy - np.float32(cys[z] if okc[z] else 0)) * np.float32(zmm[1])
        corr = np.clip(dx * np.float32(T[z, 0]) + dy * np.float32(T[z, 1]), -6.0, 6.0)
        s_f[:, :, z - z0e] = np.float32(s_axis[z]) + corr
    ids_by_z = sorted(band_ids) if z_up else sorted(band_ids, reverse=True)
    seg_ranges = list(zip(s_cuts[:-1], s_cuts[1:]))
    assign = dict(zip(ids_by_z, seg_ranges))
    s_lo_all, s_hi_all = s_cuts[0], s_cuts[-1]
    markers = np.zeros(seg_s.shape, dtype=np.int32)
    r2 = CFG["seed_radius_mm"] ** 2
    margin = CFG["seed_margin_mm"]
    d2_slab = np.empty(seg_s.shape, dtype=np.float32)
    for z in range(z0e, z1e):
        dx = (xx - np.float32(cxs[z] if okc[z] else 0)) * np.float32(zmm[0])
        dy = (yy - np.float32(cys[z] if okc[z] else 0)) * np.float32(zmm[1])
        d2_slab[:, :, z - z0e] = dx * dx + dy * dy
    eligible_s = band_s | pool_s
    for lid, (sa, sb) in assign.items():
        seeds = (bone_s & eligible_s & (d2_slab <= r2)
                 & (s_f >= sa + margin) & (s_f <= sb - margin))
        if not seeds.any():
            er = ndimage.binary_erosion(seg_s == lid, structure=S6, iterations=2)
            seeds = er if er.any() else (seg_s == lid)
            qa["flags"].append(f"SEED_FALLBACK_{NAMES[lid]}")
        markers[seeds] = lid
    rec["seed_cm3"] = {NAMES[l]: round(float((markers == l).sum()) * vox_mm3 / 1000.0, 1)
                       for l in assign}
    st = ndimage.generate_binary_structure(3, 1)
    halo_it = int(max(np.round(CFG["halo_mm"] / np.asarray(zooms))))
    grown = ndimage.binary_dilation(band_s | pool_s, structure=st, iterations=max(halo_it, 1))
    halo = grown & (seg_s == 0) & bone_s & ~pool_s
    rb2 = (CFG["seed_radius_mm"] + 4.0) ** 2
    body_zone = d2_slab <= rb2
    in_s = (s_f >= s_lo_all) & (s_f <= s_hi_all)
    in_s_arch = ((s_f >= s_lo_all - CFG["arch_overhang_mm"])
                 & (s_f <= s_hi_all + CFG["arch_overhang_mm"]))
    D = (band_s | pool_s | halo) & np.where(body_zone, in_s, in_s_arch)
    markers[~D] = 0
    labels_ws = watershed(np.zeros(D.shape, dtype=np.uint8), markers=markers, mask=D)
    A = D & ~body_zone
    if A.any():
        labels_ws, arch_rec = arch_repartition(labels_ws, body_zone & D, A,
                                               list(assign), zooms, vox_mm3)
        rec.update(arch_rec)
        for nm in arch_rec.get("arch_rootless", []):
            qa["flags"].append(f"ARCH_NO_ROOT_{nm}")
    out = seg.copy()
    out[band_mask] = 0
    out_s = out[:, :, slab]
    sel = labels_ws > 0
    out_s[sel] = labels_ws[sel].astype(seg.dtype)
    rec["unreached_pool_mm3"] = round(float((D & ~sel & (seg_s == 0)).sum()) * vox_mm3, 1)
    cleared = band_mask & (out == 0)
    rec["cleared_out_of_band_mm3"] = round(float(cleared.sum()) * vox_mm3, 1)
    rec["cleared_arch_mm3"] = round(float((cleared[:, :, slab] & ~body_zone).sum())
                                    * vox_mm3, 1)
    return out

## the midline strip at the back of the spine, where the spinous processes are
def _spinous_frame(seg, ct, zooms):
    cx, cy = _centerline_xy(seg)
    if cx is None:
        return None
    nz = seg.shape[2]
    rb = 18.0
    counts = np.zeros((nz, 25), dtype=np.int64)
    y_lo = np.full(nz, np.nan)
    y_hi = np.full(nz, np.nan)
    d_all = np.full(nz, np.nan)
    d_bod = np.full(nz, np.nan)
    for z in range(nz):
        i, j = np.nonzero(seg[:, :, z] > 0)
        if i.size < 20:
            continue
        d2 = (((i - cx[z]) * zooms[0]) ** 2 + ((j - cy[z]) * zooms[1]) ** 2)
        sel = d2 <= rb * rb
        if sel.sum() < 20:
            continue
        labs = seg[i[sel], j[sel], z]
        counts[z] = np.bincount(labs, minlength=25)
        y_lo[z], y_hi[z] = np.percentile(j[sel], 5), np.percentile(j[sel], 95)
        d_all[z], d_bod[z] = j.mean(), j[sel].mean()
    ok = ~np.isnan(y_lo)
    if ok.sum() < 5:
        return None
    psign = 1 if np.nanmean(d_all - d_bod) >= 0 else -1
    idx = np.arange(nz)
    yB = np.interp(idx, idx[ok], (y_hi if psign > 0 else y_lo)[ok])
    yB = ndimage.median_filter(yB, 11)
    off = CFG["imb_canal_mm"] / zooms[1]
    y0 = np.round(yB + off if psign > 0 else yB - off).astype(int)
    bands = {}
    for lid in range(1, 25):
        c = counts[:, lid].astype(np.float64)
        if c.sum() < 100:
            continue
        cw = np.cumsum(c) / c.sum()
        bands[lid] = (int(np.searchsorted(cw, 0.04)),
                      int(np.searchsorted(cw, 0.96)))
    return {"cx": cx, "cy": cy, "psign": psign, "y0": y0, "bands": bands}

## how much bone in that strip sits above its own vertebra (should be ~0)
def _imbrication_cm3(seg, ct, zooms, vox_mm3, lids=None, tol_mm=5.0,
                     frame=None):
    fr = frame if frame is not None else _spinous_frame(seg, ct, zooms)
    if fr is None:
        return 0.0, {}
    if lids is None:
        lids = [int(v) for v in np.unique(seg) if v != 0]
    bone = ct >= CFG["bone_hu"]
    cx, psign, y0 = fr["cx"], fr["psign"], fr["y0"]
    ztop = {l: fr["bands"][l][1] for l in lids if l in fr["bands"]}
    tol = int(round(tol_mm / zooms[2]))
    hw = max(int(round(7.0 / zooms[0])), 2)
    upv = {lid: 0 for lid in ztop}
    for z in range(seg.shape[2]):
        active = [l for l, zt in ztop.items() if z > zt + tol]
        if not active:
            continue
        xc = int(round(cx[z]))
        xs = slice(max(xc - hw, 0), xc + hw + 1)
        ys = (slice(y0[z], None) if psign > 0
              else slice(0, max(y0[z] + 1, 0)))
        s2 = seg[xs, ys, z]
        b2 = bone[xs, ys, z]
        v = s2[(s2 > 0) & b2]
        if not v.size:
            continue
        bc = np.bincount(v, minlength=25)
        for l in active:
            upv[l] += int(bc[l])
    per = {NAMES[l]: round(n * vox_mm3 / 1e3, 2)
           for l, n in upv.items() if n * vox_mm3 > 20.0}
    total = round(float(sum(n for n in upv.values())) * vox_mm3 / 1e3, 2)
    return total, per

## spinous processes droop down - their label may only flow downward from where they start
def stage2g_imbrication(seg, ct, affine, zooms, vox_mm3, qa):
    fr = _spinous_frame(seg, ct, zooms)
    if fr is None:
        qa["imbrication"] = {"skipped": "no frame"}
        return seg
    cx, psign, y0 = fr["cx"], fr["psign"], fr["y0"]
    bone = ct >= CFG["bone_hu"]
    u0, up0 = _imbrication_cm3(seg, ct, zooms, vox_mm3, frame=fr)
    hw = max(int(round(CFG["imb_half_mm"] / zooms[0])), 2)
    sw = int(round(CFG["imb_strip_mm"] / zooms[1]))
    zs = np.nonzero(seg.any(axis=(0, 1)))[0]
    if zs.size == 0:
        qa["imbrication"] = {"skipped": "empty"}
        return seg
    nz_ = seg.shape[2]
    bx = np.full(nz_, np.nan)
    wide = max(int(round(25.0 / zooms[0])), 3)
    for z in range(int(zs[0]), int(zs[-1]) + 1):
        xc = int(round(cx[z]))
        xs = slice(max(xc - wide, 0), xc + wide + 1)
        ys = (slice(y0[z], seg.shape[1]) if psign > 0
              else slice(0, max(y0[z] + 1, 0)))
        bb = bone[xs, ys, z] & (seg[xs, ys, z] > 0)
        if bb.sum() >= 10:
            bx[z] = xs.start + np.nonzero(bb)[0].mean()
    okb = ~np.isnan(bx)
    if okb.sum() >= 5:
        idxz = np.arange(nz_)
        bx = np.interp(idxz, idxz[okb], bx[okb])
        bx = ndimage.gaussian_filter1d(bx, 4.0)
    else:
        bx = cx
    out = seg.copy()
    prev = None
    unreached = 0
    for z in range(int(zs[-1]), int(zs[0]) - 1, -1):
        xc = int(round(bx[z]))
        xs = slice(max(xc - hw, 0), xc + hw + 1)
        if psign > 0:
            cys = slice(min(y0[z], seg.shape[1]), seg.shape[1])
            sys_ = slice(min(y0[z], seg.shape[1]),
                         min(y0[z] + sw, seg.shape[1]))
        else:
            cys = slice(0, max(y0[z] + 1, 0))
            sys_ = slice(max(y0[z] + 1 - sw, 0), max(y0[z] + 1, 0))
        c2 = np.zeros(seg.shape[:2], dtype=bool)
        c2[xs, cys] = bone[xs, cys, z]
        s2m = np.zeros(seg.shape[:2], dtype=bool)
        s2m[xs, sys_] = bone[xs, sys_, z]
        lab2 = out[:, :, z]
        m2 = c2 & (lab2 > 0)
        if not m2.any():
            prev = None
            continue
        seed2 = s2m & m2
        if seed2.any():
            ii, jj = np.nonzero(seed2)
            lv = lab2[ii, jj].astype(int)
            zb = np.array([fr["bands"].get(l, (z, z))[0] for l in lv])
            zt = np.array([fr["bands"].get(l, (z, z))[1] for l in lv])
            tol2 = int(round(6.0 / zooms[2]))
            bad_ = (z < zb - tol2) | (z > zt + tol2)
            seed2[ii[bad_], jj[bad_]] = False
        grow2 = m2 & ~seed2
        mk = np.where(seed2, lab2, 0).astype(np.int32)
        if prev is not None:
            take = (mk == 0) & m2 & (prev > 0)
            mk[take] = prev[take]
        if (mk > 0).any():
            ws = watershed(np.zeros(mk.shape, dtype=np.uint8), markers=mk,
                           mask=m2)
        else:
            ws = mk
        new2 = np.where(grow2 & (ws > 0), ws.astype(lab2.dtype), lab2)
        unreached += int((grow2 & (ws == 0)).sum())
        out[:, :, z] = new2
        prev = np.where(m2, new2, 0)
    diff = out != seg
    if diff.any():
        padv = np.ceil(10.0 / np.asarray(zooms)).astype(int)
        bb = _bbox_pad(ndimage.find_objects(diff.astype(np.uint8))[0],
                       padv, seg.shape)
        sub = majority_filter(out[bb].copy(), iters=1)
        sub = absorb_orphans(sub, vox_mm3, max_mm3=1500.0, delete_below=0.0)
        keep = out[bb] > 0
        out[bb] = np.where(keep, np.where(sub > 0, sub, out[bb]), 0)
    u1, up1 = _imbrication_cm3(out, ct, zooms, vox_mm3, frame=fr)
    mix_before, mix_after, changed, deltas, s_before, s_after = _shift_report(
        seg, out, bone, zooms, cx, vox_mm3, affine)
    rec = {"changed_cm3": round(changed * vox_mm3 / 1e3, 2),
           "unreached_mm3": round(unreached * vox_mm3, 1),
           "upv_cm3_before": u0, "upv_cm3_after": u1,
           "upv_per_level_before": up0, "upv_per_level_after": up1,
           "badcut_pieces_before": mix_before,
           "badcut_pieces_after": mix_after,
           "volume_deltas_cm3": deltas}
    qa["imbrication"] = rec
    over = [n for n, d in deltas.items()
            if abs(d) > CFG["imb_max_shift_cm3"]]
    ok_meter = (u1 < u0 - 0.2) or (u0 <= 0.2 and u1 <= u0 + 0.05
                                   and changed * vox_mm3 / 1e3 <= 2.0)
    rec["gate"] = {"ok_meter": bool(ok_meter),
                   "audit_bad_before_after": [_bad(s_before), _bad(s_after)]}
    if (not ok_meter or mix_after > mix_before
            or _bad(s_after) > _bad(s_before) or over):
        rec["reverted_all"] = True
        if over:
            rec["over_shift"] = over
        qa["flags"].append("IMBRICATION_REVERTED_ALL")
        return seg
    return out

## redraw every suspicious stretch, keep each one only if the audit got better
def stage2b_arbitrate(seg, raw, ct, affine, zooms, vox_mm3, qa):
    bands, suspects, present = find_suspects(seg, raw, affine, vox_mm3)
    qa["suspects"] = suspects
    if not bands:
        return seg
    cxs, cys, rho_s = column_profile(seg, raw, ct, zooms)
    def badness(s, names):
        rows_, _ = audit(s, affine, vox_mm3)
        bad = 0
        for r in rows_:
            nm = r["name"].replace("vertebrae_", "")
            if nm in names:
                bad += sum(1 for f in r["flags"]
                           if f.startswith(("SIZE", "FRAG", "EMPTY", "ORDER")))
                ratio = r.get("size_ratio_vs_neighbors")
                if ratio is not None and ratio > CFG["suspect_size_hi"]:
                    bad += 1
        return bad
    out = seg
    for band in bands:
        if {"C1", "C2"} & set(band):
            qa["flags"].append(f"ATLAS_AXIS_SKIP band={band}: no disc plane at C1/C2")
            continue
        i0, i1 = present.index(band[0]), present.index(band[-1])
        above = present[i0 - 1] if i0 > 0 else None
        below = present[i1 + 1] if i1 + 1 < len(present) else None
        band_lids = [ID[n] for n in band]
        u0, _ = _imbrication_cm3(out, ct, zooms, vox_mm3, band_lids)
        cand = arbitrate_band(out, raw, ct, band, (below, above), cxs, cys, rho_s,
                              zooms, vox_mm3, qa)
        if cand is None:
            continue
        b0, b1 = badness(out, set(band)), badness(cand, set(band))
        u1, _ = _imbrication_cm3(cand, ct, zooms, vox_mm3, band_lids)
        qa["bands"][-1]["badness_before_after"] = [b0, b1]
        qa["bands"][-1]["imbrication_upv_cm3_before_after"] = [u0, u1]
        if b1 >= b0:
            qa["flags"].append(f"REVERTED band={band}: badness {b0}->{b1}")
        else:
            out = cand
    return out

## all spots where two vertebrae touch
def _adjacent_interfaces(seg):
    out = {}
    for ax, _, _, x, y in _face_pairs(seg):
        m = (x > 0) & (y > 0) & (x != y)
        if not m.any():
            continue
        i = np.nonzero(m)
        a = np.minimum(x[m], y[m]).astype(np.int32)
        b = np.maximum(x[m], y[m]).astype(np.int32)
        pts = np.c_[i[0], i[1], i[2]]
        pts[:, ax] += 1
        key = a.astype(np.int64) * 100 + b
        for k in np.unique(key):
            sel = key == k
            out.setdefault((int(k) // 100, int(k) % 100), []).append(pts[sel])
    return {k: np.vstack(v) for k, v in out.items()}

## how flat a border is - a very flat one is probably a fake straight cut
def _planarity_mm(pts, zooms):
    if pts is None or len(pts) < 50:
        return None
    Q = pts * np.asarray(zooms)
    Q = Q - Q.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return float(np.sqrt(((Q @ Vt[2]) ** 2).mean()))

## border voxels between two given vertebrae
def _pair_pts(seg, a, b):
    pts = []
    for ax, _, _, x, y in _face_pairs(seg):
        m = ((x == a) & (y == b)) | ((x == b) & (y == a))
        if m.any():
            i = np.nonzero(m)
            pp = np.c_[i[0], i[1], i[2]]
            pp[:, ax] += 1
            pts.append(pp)
    return np.vstack(pts) if pts else None

## move suspiciously flat borders onto the darkest line of the gap
def stage2c_interface_polish(seg, ct, zooms, vox_mm3, qa):
    out = seg.copy()
    pairs = _adjacent_interfaces(out)
    adj = {tuple(sorted((ID[TOP_DOWN[k]], ID[TOP_DOWN[k + 1]])))
           for k in range(len(TOP_DOWN) - 1)}
    for (a, b) in sorted(pairs):
        if (a, b) not in adj:
            continue
        pts = pairs[(a, b)]
        p0 = _planarity_mm(pts, zooms)
        area = len(pts) * (vox_mm3 ** (2 / 3))
        rec = {"pair": f"{NAMES[b]}|{NAMES[a]}",
               "contact_mm2": round(area, 0),
               "planarity_before": None if p0 is None else round(p0, 2)}
        qa["polish"].append(rec)
        if (p0 is None or area < CFG["polish_min_contact_mm2"]
                or p0 >= CFG["polish_trigger_mm"]):
            rec["action"] = "skip"
            continue
        lo = np.maximum(pts.min(0) - 2, 0)
        hi = np.minimum(pts.max(0) + 3, np.asarray(out.shape))
        pad = np.ceil((CFG["polish_collar_mm"] + CFG["polish_seed_shell_mm"] + 2)
                      / np.asarray(zooms)).astype(int)
        sl = tuple(slice(max(int(l - p), 0), min(int(h + p), n))
                   for l, h, p, n in zip(lo, hi, pad, out.shape))
        sub = out[sl]
        iface = np.zeros(sub.shape, dtype=bool)
        iface[tuple((pts - [s.start for s in sl]).T)] = True
        d_if = ndimage.distance_transform_edt(~iface, sampling=zooms)
        collar = d_if <= CFG["polish_collar_mm"]
        shell = (d_if > CFG["polish_collar_mm"]) & \
                (d_if <= CFG["polish_collar_mm"] + CFG["polish_seed_shell_mm"])
        ab = (sub == a) | (sub == b)
        R = collar & ab
        seeds = np.where(shell & ab, sub, 0).astype(np.int32)
        if not (seeds == a).any() or not (seeds == b).any() or not R.any():
            rec["action"] = "skip_no_seeds"
            continue
        if CFG["polish_priority"] == "hu":
            prio = -ndimage.gaussian_filter(
                np.clip(ct[sl].astype(np.float32), -200, 1500),
                sigma=[0.8 / z for z in zooms])
        else:
            prio = ndimage.gaussian_gradient_magnitude(
                ct[sl].astype(np.float32), sigma=[1.0 / z for z in zooms])
        ws = watershed(prio, markers=seeds, mask=R | (seeds > 0))
        cand = sub.copy()
        m_new = R & (ws > 0)
        cand[m_new] = ws[m_new].astype(cand.dtype)
        if _polish_gate(rec, sub, cand, a, b, zooms, "ws", p0):
            out[sl] = cand
    return out

## accept a polished border only if it got better and nothing broke
def _polish_gate(rec, sub, cand, a, b, zooms, tag, p0):
    m = (cand == a) | (cand == b)
    swapped = int(((cand != sub) & m).sum())
    n_min = min(int((sub == a).sum()), int((sub == b).sum()))
    p1 = _planarity_mm(_pair_pts(cand, a, b), zooms)
    rec[f"planarity_after_{tag}"] = None if p1 is None else round(p1, 2)
    rec[f"swapped_vox_{tag}"] = swapped
    ok = (p1 is not None and swapped > 0
          and swapped <= CFG["polish_max_shift_frac"] * max(n_min, 1)
          and p1 > p0 + CFG["polish_min_gain_mm"])
    if ok:
        for lab in (a, b):
            _, c_new = _components(cand == lab)
            _, c_old = _components(sub == lab)
            if int((c_new > 0).sum()) > int((c_old > 0).sum()):
                ok = False
                rec["fragment_guard"] = NAMES[lab]
                break
    rec["action"] = tag if ok else f"revert_{tag}"
    return bool(ok)

## centre of the spine in every slice
def _centerline_xy(seg):
    nz = seg.shape[2]
    cx = np.full(nz, np.nan); cy = np.full(nz, np.nan)
    for z in np.nonzero((seg > 0).any(axis=(0, 1)))[0]:
        pts = np.nonzero(seg[:, :, z])
        cx[z], cy[z] = pts[0].mean(), pts[1].mean()
    ok = ~np.isnan(cx)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return None, None
    for arr in (cx, cy):
        arr[ok] = ndimage.gaussian_filter1d(arr[ok], 10)
        m = np.isnan(arr)
        arr[m] = np.interp(np.nonzero(m)[0], idx, arr[idx])
    return cx, cy

## shift every slice so the spine sits in the middle
def _shift2d_stack(vol, sx, sy):
    out = np.zeros_like(vol)
    nx, ny, nz = vol.shape
    for z in range(nz):
        dx, dy = int(sx[z]), int(sy[z])
        xs0, xs1 = max(0, -dx), min(nx, nx - dx)
        ys0, ys1 = max(0, -dy), min(ny, ny - dy)
        out[xs0:xs1, ys0:ys1, z] = vol[xs0 + dx:xs1 + dx, ys0 + dy:ys1 + dy, z]
    return out

## one sure spot per vertebra (its body), used as a reference
def _mv_anchors(seg, ct, zooms, cx, cy):
    rb = (CFG["seed_radius_mm"] + 4.0)
    anchors = np.zeros_like(seg)
    it = max(int(np.ceil(CFG["mv_anchor_mm"] / min(zooms))), 1)
    pad = np.ceil((CFG["mv_anchor_mm"] + 4.0) / np.asarray(zooms)).astype(int)
    objs = ndimage.find_objects(seg)
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        sl = _bbox_pad(objs[lid - 1], pad, seg.shape)
        lab = seg[sl] == lid
        xx = (np.arange(sl[0].start, sl[0].stop, dtype=np.float32)[:, None, None]
              - cx[None, None, sl[2].start:sl[2].stop]) * zooms[0]
        yy = (np.arange(sl[1].start, sl[1].stop, dtype=np.float32)[None, :, None]
              - cy[None, None, sl[2].start:sl[2].stop]) * zooms[1]
        core = lab & (xx * xx + yy * yy <= rb * rb)
        if not core.any():
            core = ndimage.binary_erosion(lab, structure=S6, iterations=2)
        if not core.any():
            continue
        reach = ndimage.binary_dilation(core, structure=S6, iterations=it,
                                        mask=lab)
        anchors[sl][reach] = lid
    return anchors

## recolour a voxel only when the front, side and top views all agree
def stage2e_multiview_recolor(seg, ct, affine, zooms, vox_mm3, qa):
    cx, cy = _centerline_xy(seg)
    if cx is None:
        qa["multiview"] = {"flipped_mm3": 0.0}
        return seg
    bone = ct >= CFG["bone_hu"]
    anchors = _mv_anchors(seg, ct, zooms, cx, cy)
    sx = np.round(cx - np.median(cx)).astype(int)
    sy = np.round(cy - np.median(cy)).astype(int)
    bone_s = _shift2d_stack(bone.astype(np.uint8), sx, sy).astype(bool)
    anch_s = _shift2d_stack(anchors, sx, sy)
    votes = np.zeros(seg.shape, dtype=np.uint8)
    conflict = np.zeros(seg.shape, dtype=bool)
    SENT = 255
    for ax in range(3):
        n_sl = seg.shape[ax]
        zo2 = tuple(z for i, z in enumerate(zooms) if i != ax)
        for k in range(n_sl):
            idx = [slice(None)] * 3
            idx[ax] = k
            idx = tuple(idx)
            b2 = bone_s[idx]
            if not b2.any():
                continue
            edt2 = ndimage.distance_transform_edt(b2, sampling=zo2)
            core2 = edt2 >= CFG["mv_neck_mm"]
            if not core2.any():
                continue
            lab2, n2 = ndimage.label(core2)
            if n2 == 0:
                continue
            a2 = anch_s[idx]
            sel = (a2 > 0) & (lab2 > 0)
            comp_map = np.full(n2 + 1, SENT, dtype=np.int32)
            if sel.any():
                pairs = np.unique(np.stack([lab2[sel], a2[sel]]).astype(np.int64),
                                  axis=1)
                comps, first = np.unique(pairs[0], return_index=True)
                ncnt = np.bincount(pairs[0], minlength=n2 + 1)
                ii, jj = np.nonzero(lab2)
                cids = lab2[ii, jj]
                for c, f in zip(comps, first):
                    if ncnt[c] != 1:
                        continue
                    m = cids == c
                    if m.sum() < 12:
                        continue
                    pts = np.c_[ii[m] * zo2[0], jj[m] * zo2[1]]
                    pts = pts - pts.mean(0)
                    cov = pts.T @ pts / len(pts)
                    w = np.linalg.eigvalsh(cov)
                    if w[1] >= CFG["mv_elong"] ** 2 * max(w[0], 1e-6):
                        comp_map[c] = int(pairs[1, f])
            comp_map[0] = 0
            markers = comp_map[lab2]
            if not (markers > 0).any() or not ((markers > 0) & (markers != SENT)).any():
                continue
            ws2 = watershed(np.zeros_like(markers, dtype=np.uint8),
                            markers=markers, mask=b2)
            v2 = votes[idx]
            c2 = conflict[idx]
            val = ws2.astype(np.uint8)
            good = (val > 0) & (val != SENT)
            c2 |= good & (v2 > 0) & (v2 != val)
            newv = good & (v2 == 0)
            v2[newv] = val[newv]
            votes[idx] = v2
            conflict[idx] = c2
    votes = _shift2d_stack(votes, -sx, -sy)
    conflict = _shift2d_stack(conflict.astype(np.uint8), -sx, -sy).astype(bool)
    cand = (seg > 0) & (votes > 0) & ~conflict & (votes != seg)
    if not cand.any():
        qa["multiview"] = {"flipped_mm3": 0.0}
        return seg
    out = seg.copy()
    out[cand] = votes[cand]
    flips = {}
    pts_from = seg[cand]
    pts_to = votes[cand]
    for a, b in zip(pts_from, pts_to):
        flips[(int(a), int(b))] = flips.get((int(a), int(b)), 0) + 1
    reverted_pairs = []
    for (a, b), n in sorted(flips.items()):
        n_min = min(int((seg == a).sum()), int((seg == b).sum()))
        undo = False
        if n > CFG["mv_max_shift_frac"] * max(n_min, 1):
            undo = True
        else:
            _, c_new = _components(out == a)
            _, c_old = _components(seg == a)
            if int((c_new > 0).sum()) > int((c_old > 0).sum()):
                undo = True
        if undo:
            m = cand & (seg == a) & (votes == b)
            out[m] = a
            reverted_pairs.append(f"{NAMES[a]}->{NAMES[b]}")
    qa["multiview"] = {
        "flipped_mm3": round(float((out != seg).sum()) * vox_mm3, 1),
        "pair_flips_mm3": {f"{NAMES[a]}->{NAMES[b]}":
                           round(n * vox_mm3, 1) for (a, b), n in flips.items()},
        "reverted_pairs": reverted_pairs}
    _, s_before = audit(seg, affine, vox_mm3)
    _, s_after = audit(out, affine, vox_mm3)
    if _bad(s_after) > _bad(s_before):
        qa["multiview"]["reverted_all"] = True
        qa["flags"].append("MULTIVIEW_REVERTED_ALL")
        return seg
    return out

## area of borders that cut through thick bone (should be small)
def _masscut_mm2(seg, edtb, vox_mm3, thresh=2.5):
    m = np.zeros(seg.shape, dtype=bool)
    for ax, hi, lo, x, y in _face_pairs(seg):
        b = (x > 0) & (y > 0) & (x != y)
        m[hi] |= b
        m[lo] |= b
    return float((m & (edtb >= thresh)).sum()) * (vox_mm3 ** (2 / 3))

## distance to the label border inside a 2d slice
def _boundary_edt2(mm, edt2):
    bmask = np.zeros(mm.shape, dtype=bool)
    for a2 in range(2):
        sh = [slice(None)] * 2
        sl_ = [slice(None)] * 2
        sh[a2] = slice(1, None)
        sl_[a2] = slice(None, -1)
        xx, yy = mm[tuple(sh)], mm[tuple(sl_)]
        bb = (xx > 0) & (yy > 0) & (xx != yy)
        bmask[tuple(sh)] |= bb
        bmask[tuple(sl_)] |= bb
    if not bmask.any():
        return 0.0
    return float(np.mean(edt2[bmask]))

## before / after numbers used to keep or undo a step
def _shift_report(seg, out, bone, zooms, cx, vox_mm3, affine):
    cx_med = int(round(np.median(cx)))
    mix_before = _badcut_pieces(seg, bone, zooms, cx_med)
    mix_after = _badcut_pieces(out, bone, zooms, cx_med)
    changed = int((out != seg).sum())
    deltas = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        d = (int((out == lid).sum()) - int((seg == lid).sum())) * vox_mm3 / 1e3
        if abs(d) > 0.05:
            deltas[NAMES[lid]] = round(d, 2)
    _, s_before = audit(seg, affine, vox_mm3)
    _, s_after = audit(out, affine, vox_mm3)
    return mix_before, mix_after, changed, deltas, s_before, s_after

## count solid bone pieces that carry two labels
def _badcut_pieces(seg, bone, zooms, cx_med, span=10, thresh=2.6):
    n_bad = 0
    for x in range(cx_med - span, cx_med + span + 1):
        if not (0 <= x < seg.shape[0]):
            continue
        b2 = bone[x] | (seg[x] > 0)
        if not b2.any():
            continue
        edt2 = ndimage.distance_transform_edt(b2, sampling=(zooms[1], zooms[2]))
        l2, n2 = ndimage.label((edt2 >= 1.2) & (seg[x] > 0))
        s2 = seg[x]
        for c in range(1, n2 + 1):
            m = l2 == c
            if m.sum() * zooms[1] * zooms[2] < 60:
                continue
            vals = s2[m]
            ids, cnt = np.unique(vals[vals > 0], return_counts=True)
            if ids.size < 2 or (cnt / cnt.sum()).min() < 0.15:
                continue
            if _boundary_edt2(np.where(m, s2, 0), edt2) >= thresh:
                n_bad += 1
    return n_bad

## a solid bone piece with two labels gets one label
## (or the border moves to its thinnest spot)
def stage2f_skeleton_relabel(seg, ct, affine, zooms, vox_mm3, qa):
    import gc
    gc.collect()
    nzs = np.nonzero(seg)
    if nzs[0].size == 0:
        qa["skeleton"] = {"skipped": "empty"}
        return seg
    padv = np.ceil(8.0 / np.asarray(zooms)).astype(int)
    tight = tuple(slice(max(int(c.min() - p), 0), min(int(c.max() + p + 1), n))
                  for c, p, n in zip(nzs, padv, seg.shape))
    seg_orig = seg
    seg = seg[tight].copy()
    ct = np.ascontiguousarray(ct[tight])
    gc.collect()
    bone = ct >= CFG["bone_hu"]
    edtb = ndimage.distance_transform_edt(bone, sampling=zooms).astype(np.float32)
    gc.collect()
    mc_before = _masscut_mm2(seg, edtb, vox_mm3)
    cx, cy = _centerline_xy(seg)
    if cx is None:
        qa["skeleton"] = {"skipped": "no centerline"}
        return seg
    sx = np.round(cx - np.median(cx)).astype(int)
    sy = np.round(cy - np.median(cy)).astype(int)
    seg_h = _shift2d_stack(seg, sx, sy)
    bone_h = _shift2d_stack(bone.astype(np.uint8), sx, sy).astype(bool)
    out_h = seg_h.copy()
    for ax in range(3):
        n_sl = seg.shape[ax]
        zo2 = tuple(z for i, z in enumerate(zooms) if i != ax)
        for k in range(n_sl):
            idx = [slice(None)] * 3
            idx[ax] = k
            idx = tuple(idx)
            s2 = out_h[idx]
            lab2d = s2 > 0
            if not lab2d.any():
                continue
            edt2 = ndimage.distance_transform_edt(
                bone_h[idx] | lab2d, sampling=zo2)
            l2, n2 = ndimage.label((edt2 >= CFG["mv_neck_mm"]) & lab2d)
            if n2 == 0:
                continue
            for pid in range(1, n2 + 1):
                m = l2 == pid
                npx = int(m.sum())
                if npx * zo2[0] * zo2[1] < 60:
                    continue
                vals = s2[m]
                ids, cnt = np.unique(vals, return_counts=True)
                sel = ids > 0
                ids, cnt = ids[sel], cnt[sel]
                if ids.size < 2:
                    continue
                order = np.argsort(cnt)[::-1]
                major = int(ids[order[0]])
                minority_frac = 1.0 - cnt[order[0]] / cnt.sum()
                mm = np.where(m, s2, 0)
                bedt = _boundary_edt2(mm, edt2)
                if bedt < CFG["skel_cut_mm"]:
                    continue
                if minority_frac <= 0.40:
                    s2[m & (mm != major) & (mm > 0)] = major
                    continue
                mk = np.zeros_like(mm)
                for lid_ in ids:
                    part = mm == lid_
                    er = ndimage.binary_erosion(part, iterations=3)
                    mk[er if er.any() else part] = lid_
                ws = watershed(-edt2.astype(np.float32), markers=mk, mask=m)
                cand2 = np.where(m, ws, mm).astype(mm.dtype)
                if _boundary_edt2(np.where(m, cand2, 0), edt2) < bedt - 0.2:
                    s2[m] = cand2[m].astype(s2.dtype)
    out = _shift2d_stack(out_h, -sx, -sy)
    del out_h, seg_h, bone_h
    gc.collect()
    keep = seg > 0
    out = np.where(keep, np.where(out > 0, out, seg), 0).astype(seg.dtype)
    del keep
    out = majority_filter(out, iters=2)
    gc.collect()
    out = absorb_orphans(out, vox_mm3, max_mm3=600.0, delete_below=0.0)
    for lid in [int(v) for v in np.unique(out) if v != 0]:
        cc_, counts_ = _components(out == lid)
        if (counts_ > 0).sum() <= 1:
            continue
        main_ = int(counts_.argmax())
        for comp_id in np.nonzero(counts_)[0]:
            if comp_id == main_ or counts_[comp_id] * vox_mm3 > 600.0:
                continue
            comp_ = cc_ == comp_id
            if counts_[comp_id] * vox_mm3 < CFG["speck_mm3"]:
                sh_ = ndimage.binary_dilation(comp_, structure=S26) & ~comp_
                v_ = out[sh_]
                v_ = v_[v_ > 0]
                if v_.size:
                    ids_, cnt_ = np.unique(v_, return_counts=True)
                    out[comp_] = int(ids_[cnt_.argmax()])
                    continue
            if (seg[comp_] != lid).any():
                out[comp_] = seg[comp_]
    gc.collect()
    mc_after = _masscut_mm2(out, edtb, vox_mm3)
    del edtb
    gc.collect()
    mix_before, mix_after, changed, deltas, s_before, s_after = _shift_report(
        seg, out, bone, zooms, cx, vox_mm3, affine)
    rec = {"changed_mm3": round(changed * vox_mm3, 1),
           "masscut_before_mm2": round(mc_before, 0),
           "masscut_after_mm2": round(mc_after, 0),
           "badcut_pieces_before": mix_before,
           "badcut_pieces_after": mix_after,
           "volume_deltas_cm3": deltas}
    qa["skeleton"] = rec
    over = [n for n, d in deltas.items() if abs(d) > 4.0]
    if mix_after > mix_before or _bad(s_after) > _bad(s_before) or over:
        rec["reverted_all"] = True
        if over:
            rec["over_shift"] = over
        qa["flags"].append("SKELETON_REVERTED_ALL")
        return seg_orig
    out_full = seg_orig.copy()
    out_full[tight] = out
    return out_full

## bone set aside in step 1 goes back to the vertebra it is joined to through bone
def stage2d_reclaim_pool(seg, raw, ct, zooms, vox_mm3, qa):
    bone = ct >= CFG["bone_hu"]
    pool = (raw > 0) & (seg == 0) & bone
    cc, counts = _components(pool)
    st2 = ndimage.generate_binary_structure(2, 1)
    summary = {}
    for comp_id in np.nonzero(counts)[0]:
        vol = float(counts[comp_id]) * vox_mm3
        if vol < CFG["speck_mm3"]:
            continue
        comp = cc == comp_id
        objs = ndimage.find_objects(comp.astype(np.uint8))[0]
        rl = raw[comp]
        ids_r, cnt_r = np.unique(rl[rl > 0], return_counts=True)
        rec = {"stage": "2d", "vol_mm3": round(vol, 1),
               "raw_label": NAMES[int(ids_r[cnt_r.argmax()])]}
        ring_votes = {}
        for z in range(objs[2].start, objs[2].stop):
            cz = comp[:, :, z]
            if not cz.any():
                continue
            lab2, _ = ndimage.label(bone[:, :, z], structure=st2)
            for h in np.unique(lab2[cz]):
                if h == 0:
                    continue
                labs = seg[:, :, z][lab2 == h]
                for l, c in zip(*np.unique(labs[labs > 0], return_counts=True)):
                    ring_votes[int(l)] = ring_votes.get(int(l), 0) + int(c)
        ring = max(ring_votes, key=ring_votes.get) if ring_votes else 0
        pad_vox = np.ceil((CFG["reclaim_link_mm"] + 2.0) / np.asarray(zooms)).astype(int)
        slc = _bbox_pad(objs, pad_vox, seg.shape)
        it = max(int(np.ceil(CFG["reclaim_link_mm"] / min(zooms))), 1)
        reach = ndimage.binary_dilation(comp[slc], structure=S6,
                                        iterations=it,
                                        mask=bone[slc] | comp[slc])
        labs = seg[slc][reach]
        ids3, cnt3 = np.unique(labs[labs > 0], return_counts=True)
        geo = int(ids3[cnt3.argmax()]) if ids3.size else 0
        rec["ring_vote"] = (NAMES[ring] or "-")
        rec["geodesic_vote"] = (NAMES[geo] or "-")
        lid = ring if ring else geo
        if lid == 0:
            rec["action"] = "UNLINKED_NOT_VERTEBRA"
            qa["flags"].append(
                f"POOL_UNRESOLVED_{rec['raw_label']}_{int(vol)}mm3")
            qa["records"].append(rec)
            continue
        mcc, mcounts = _components(seg == lid)
        main = mcc == int(mcounts.argmax())
        added = _bridge(seg, bone, main, comp, lid, zooms,
                        neck_mm=CFG["reclaim_neck_mm"])
        seg[comp] = lid
        if added is not None:
            seg[added] = lid
            rec["action"] = "RECLAIMED_BRIDGED"
            rec["bridge_added_mm3"] = round(float(added.sum()) * vox_mm3, 1)
        else:
            rec["action"] = "RECLAIMED_KEPT_SEPARATE"
        nm = NAMES[lid]
        summary[nm] = round(summary.get(nm, 0.0) + vol / 1000.0, 2)
        qa["records"].append(rec)
    qa["reclaim_cm3_by_level"] = summary
    return absorb_orphans(seg, vox_mm3, max_mm3=300.0, delete_below=0.0)

## smooth jagged borders: a border voxel takes its neighbours' most common label
def majority_filter(seg, iters=3):
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    out = seg.copy()
    for _ in range(iters):
        diff = np.zeros(out.shape, dtype=bool)
        for ax, hi, lo, x, y in _face_pairs(out):
            b = (x > 0) & (y > 0) & (x != y)
            diff[hi] |= b
            diff[lo] |= b
        diff &= out > 0
        pts = np.nonzero(diff)
        n = pts[0].size
        if n == 0:
            break
        counts = np.zeros((n, 25), dtype=np.uint8)
        shp = out.shape
        for dx, dy, dz in offs:
            xx = np.clip(pts[0] + dx, 0, shp[0] - 1)
            yy = np.clip(pts[1] + dy, 0, shp[1] - 1)
            zz = np.clip(pts[2] + dz, 0, shp[2] - 1)
            counts[np.arange(n), out[xx, yy, zz]] += 1
        cur = out[pts]
        lab_counts = counts[:, 1:]
        mode = lab_counts.argmax(axis=1) + 1
        swap = (mode != cur) & (lab_counts.max(axis=1) > counts[np.arange(n), cur])
        if not swap.any():
            break
        out[pts[0][swap], pts[1][swap], pts[2][swap]] = mode[swap].astype(out.dtype)
    return out

## small leftover pieces join the label around them
def absorb_orphans(seg, vox_mm3, max_mm3=1200.0, delete_below=100.0):
    out = seg.copy()
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id or counts[comp_id] * vox_mm3 > max_mm3:
                continue
            comp = cc == comp_id
            sl = _bbox_pad(ndimage.find_objects(comp.astype(np.uint8))[0],
                           np.array([2, 2, 2]), seg.shape)
            comp_l = comp[sl]
            shell = ndimage.binary_dilation(comp_l, iterations=2) & ~comp_l
            votes = out[sl][shell]
            votes = votes[(votes != 0) & (votes != lid)]
            if votes.size >= 5:
                ids, cnt = np.unique(votes, return_counts=True)
                out[comp] = int(ids[cnt.argmax()])
            elif counts[comp_id] * vox_mm3 < delete_below:
                out[comp] = 0
    return out

## smooth the surfaces a little (max 1.5 mm), undo it for any vertebra that changes too much
def stage3_smooth(seg, ct, zooms, vox_mm3, qa):
    seg = majority_filter(seg, iters=3)
    seg = absorb_orphans(seg, vox_mm3)
    bone = ct >= CFG["bone_hu"]
    pad_vox = np.ceil(CFG["smooth_pad_mm"] / np.asarray(zooms))
    objs = ndimage.find_objects(seg)
    per_label = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        sl = _bbox_pad(objs[lid - 1], pad_vox, seg.shape)
        m = seg[sl] == lid
        others = (seg[sl] > 0) & ~m
        m1 = ndimage.binary_fill_holes(m)
        for ax in (0, 1):
            fill2d = np.zeros_like(m1)
            for k in range(m1.shape[ax]):
                idx = [slice(None)] * 3
                idx[ax] = k
                fill2d[tuple(idx)] = ndimage.binary_fill_holes(m1[tuple(idx)])
            m1 |= fill2d & ~others
        m_f = ndimage.binary_fill_holes(m1)
        m_f &= ~others
        d_in = ndimage.distance_transform_edt(m_f, sampling=zooms)
        d_out = ndimage.distance_transform_edt(~m_f, sampling=zooms)
        sdt = (d_in - d_out).astype(np.float32)
        s = ndimage.gaussian_filter(sdt, sigma=[CFG["sigma_mm"] / z for z in zooms])
        target = int(m_f.sum())
        lo, hi = -2.0, 2.0
        tau = 0.0
        for _ in range(40):
            tau = 0.5 * (lo + hi)
            n = int((s > tau).sum())
            if abs(n - target) <= CFG["vol_tol"] * target:
                break
            lo, hi = (tau, hi) if n > target else (lo, tau)
        per_label[lid] = (sl, m_f, sdt, s, float(tau))
        qa["smooth"].append({"label": NAMES[lid],
                             "holes_filled_mm3": round(float((m_f & ~m).sum()) * vox_mm3, 1),
                             "tau_mm": round(float(tau), 3)})
    out = np.zeros_like(seg)
    step = max(64, int(120 / zooms[2]))
    for c0 in range(0, seg.shape[2], step):
        c1 = min(c0 + step, seg.shape[2])
        score = np.full((seg.shape[0], seg.shape[1], c1 - c0), -np.inf, dtype=np.float32)
        lab = np.zeros((seg.shape[0], seg.shape[1], c1 - c0), dtype=seg.dtype)
        for lid, (sl, m_f, sdt, s, tau) in per_label.items():
            i0, i1 = max(sl[2].start, c0), min(sl[2].stop, c1)
            if i0 >= i1:
                continue
            zloc = slice(i0 - sl[2].start, i1 - sl[2].start)
            s_l, sdt_l, m_l = s[:, :, zloc], sdt[:, :, zloc], m_f[:, :, zloc]
            band = np.abs(sdt_l) <= CFG["max_dev_mm"]
            claim = np.where(band, s_l > tau, m_l)
            marg = np.where(band, s_l - tau, np.where(m_l, 1e3, -1e3)).astype(np.float32)
            claim &= (m_l | bone[sl[0], sl[1], i0:i1])
            sc = score[sl[0], sl[1], i0 - c0:i1 - c0]
            lb = lab[sl[0], sl[1], i0 - c0:i1 - c0]
            upd = claim & (marg > sc)
            sc[upd] = marg[upd]
            lb[upd] = lid
        out[:, :, c0:c1] = lab
    for lid, (sl, m_f, sdt, s, tau) in per_label.items():
        new = out[sl] == lid
        denom = int(new.sum()) + int(m_f.sum())
        dice = 2 * int((new & m_f).sum()) / denom if denom else 1.0
        _, c_new = _components(new)
        _, c_old = _components(m_f)
        rec = next(r for r in qa["smooth"] if r["label"] == NAMES[lid])
        rec["dice_vs_presmooth"] = round(dice, 4)
        if dice < CFG["min_dice"] or int((c_new > 0).sum()) > int((c_old > 0).sum()):
            out[sl][m_f] = lid
            out[sl][(out[sl] == lid) & ~m_f] = 0
            rec["reverted"] = True
    return absorb_orphans(out, vox_mm3)


## pieces per vertebra - used to check the cleanup did not break anything
def _pieces(seg, vox):
    out = {}
    for l, sl in _boxes(seg):
        _, sz = components(seg[sl] == l)
        out[NAMES[l]] = int((sz * vox > 20).sum())
    return out

## bits hanging by a 1-voxel thread that sit on another vertebra take its label
def _split_threaded_pieces(out, vox, moved):
    for l, sl in _boxes(out, 3):
        cur = out[sl]
        m = cur == l
        core = ndimage.binary_erosion(m, structure=S26)
        lab, sizes = components(core)
        if (sizes > 0).sum() <= 1:
            continue
        main = lab == sizes.argmax()
        near_main = ndimage.binary_dilation(main, structure=S26, iterations=3)
        for cid in np.nonzero(sizes)[0]:
            if cid == sizes.argmax():
                continue
            piece = ndimage.binary_dilation(lab == cid, structure=S26, iterations=2,
                                            mask=m & ~near_main) | ((lab == cid) & m)
            n = int(piece.sum())
            if n == 0 or n * vox / 1e3 > CFG["island_max_cm3"]:
                continue
            shell = ndimage.binary_dilation(piece, structure=S26) & ~piece
            v = cur[shell]
            v = v[(v != 0) & (v != l)]
            if v.size < 5:
                continue
            ids, cnt = np.unique(v, return_counts=True)
            if cnt.max() / v.size >= CFG["island_host_frac"]:
                host = int(ids[cnt.argmax()])
                cur[piece] = host
                moved.append({"from": NAMES[l], "to": NAMES[host], "mm3": round(n * vox, 1),
                              "how": "was hanging by a thin thread"})

## small loose pieces mostly surrounded by another vertebra take its label
def fix_stray_pieces(seg, vox, rep):
    out = seg.copy()
    moved = []
    for _ in range(2):
        _split_threaded_pieces(out, vox, moved)
    for _ in range(2):
        for l, sl in _boxes(out, 2):
            lab, sizes = components(out[sl] == l)
            if (sizes > 0).sum() <= 1:
                continue
            main = sizes.argmax()
            for cid in np.nonzero(sizes)[0]:
                if cid == main or sizes[cid] * vox / 1e3 > CFG["island_max_cm3"]:
                    continue
                comp = lab == cid
                shell = ndimage.binary_dilation(comp, structure=S26) & ~comp
                v = out[sl][shell]
                v = v[(v != 0) & (v != l)]
                if v.size < 5:
                    continue
                ids, n = np.unique(v, return_counts=True)
                host = int(ids[n.argmax()])
                if n.max() / v.size >= CFG["island_host_frac"]:
                    out[sl][comp] = host
                    moved.append({"from": NAMES[l], "to": NAMES[host],
                                  "mm3": round(float(sizes[cid] * vox), 1)})
    rep["stray_pieces_moved"] = moved
    return out

## a ball of r mm made of voxels (voxels are not cubes)
def _ball(radius_mm, zooms):
    r = [max(int(np.floor(radius_mm / z)), 1) for z in zooms]
    g = np.ogrid[tuple(slice(-k, k + 1) for k in r)]
    return sum((gi * z / radius_mm) ** 2 for gi, z in zip(g, zooms)) <= 1.0

## fill holes and small dents - only where the ct is bone and not next to another vertebra
def fill_holes_and_pits(seg, ct, zooms, rep):
    vox = float(np.prod(zooms))
    out = seg.copy()
    ball = _ball(CFG["pit_close_mm"], zooms)
    pad = max(ball.shape) + 2
    filled = {}
    skipped = []
    for l, sl in _boxes(seg, pad):
        cur = out[sl]
        m = cur == l
        others = (cur > 0) & ~m
        near_other = ndimage.binary_dilation(others, structure=S26)
        cand = ndimage.binary_fill_holes(m) & ~m
        for ax in range(3):
            mm = np.moveaxis(m, ax, 0)
            f2 = np.zeros_like(mm)
            for k in range(mm.shape[0]):
                if mm[k].any():
                    f2[k] = ndimage.binary_fill_holes(mm[k])
            cand |= np.moveaxis(f2, 0, ax) & ~m
        cand &= (cur == 0) & ~near_other
        lab, sizes = components(cand)
        hu = ndimage.mean(ct[sl].astype(np.float32), labels=lab,
                          index=np.arange(1, len(sizes))) if len(sizes) > 1 else []
        keep = [i + 1 for i, h in enumerate(hu)
                if h >= CFG["hole_min_hu"] and sizes[i + 1] * vox / 1e3 <= CFG["hole_max_cm3"]]
        cand = np.isin(lab, keep) if keep else np.zeros_like(cand)
        closed = ndimage.binary_closing(m, structure=ball)
        pit = closed & ~m & (cur == 0) & ~near_other & (ct[sl] >= CFG["bone_hu"])
        plab, psz = components(pit)
        big = np.nonzero(psz * vox > CFG["pit_max_mm3"])[0]
        if big.size:
            pit &= ~np.isin(plab, big)
        add = cand | pit
        n = int(add.sum())
        if n == 0:
            continue
        if n > CFG["max_growth"] * int(m.sum()):
            skipped.append({"vertebra": NAMES[l], "would_add_mm3": round(n * vox, 1)})
            continue
        cur[add] = l
        filled[NAMES[l]] = round(n * vox, 1)
    rep["filled_mm3"] = filled
    rep["fill_skipped_too_large"] = skipped
    return out

## step 2: the three fixes above; all undone if a vertebra ends up in more pieces
def cleanup(seg, ct, zooms, rep):
    vox = float(np.prod(zooms))
    before = _pieces(seg, vox)
    s1 = fix_stray_pieces(seg, vox, rep)
    s2 = fill_holes_and_pits(s1, ct, zooms, rep)
    after = _pieces(s2, vox)
    worse = [k for k in after if after[k] > before.get(k, 0)]
    rep["pieces_per_vertebra_before"] = {k: v for k, v in before.items() if v > 1}
    rep["pieces_per_vertebra_after"] = {k: v for k, v in after.items() if v > 1}
    rep["changed_mm3"] = round(float((s2 != seg).sum()) * vox, 1)
    if worse:
        rep["result"] = f"REVERTED (more pieces in {worse})"
        return seg
    rep["result"] = "ACCEPTED"
    return s2

## shrink a vertebra - parts that come loose were hanging on a thin neck
def _neck_parts(m, shrink, zooms):
    core = ndimage.binary_erosion(m, structure=shrink)
    lab, n = ndimage.label(core, structure=S6)
    if n <= 1:
        return []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = sizes.argmax()
    _, near = ndimage.distance_transform_edt(lab == 0, sampling=zooms, return_indices=True)
    own = np.where(m, lab[tuple(near)], 0)
    return [own == c for c in np.nonzero(sizes)[0] if c != main]

## pieces touching the vertebra only by a corner or an edge
def _face_parts(m):
    lab, n = ndimage.label(m, structure=S6)
    if n <= 1:
        return []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = sizes.argmax()
    return [lab == c for c in np.nonzero(sizes)[0] if c != main]

## face-connected pieces per vertebra (what a 3d viewer shows)
def _loose_pieces(seg):
    out = {}
    for l, sl in _boxes(seg):
        n = ndimage.label(seg[sl] == l, structure=S6)[1]
        if n > 1:
            out[NAMES[l]] = int(n)
    return out

## a speck sitting on another vertebra takes its label, otherwise it is deleted
def _fix_speck(cur, p, l, vox, rep, why):
    ring = cur[ndimage.binary_dilation(p, structure=S6) & ~p]
    outside = ring[ring != l]
    other = outside[outside != 0]
    mm3 = round(float(p.sum()) * vox, 1)
    if other.size and other.size >= CFG["speck_host_frac"] * outside.size:
        ids, n = np.unique(other, return_counts=True)
        host = int(ids[n.argmax()])
        cur[p] = host
        rep["recoloured"].append({"from": NAMES[l], "to": NAMES[host], "mm3": mm3, "why": why})
    else:
        cur[p] = 0
        rep["deleted_mm3"][NAMES[l]] = round(rep["deleted_mm3"].get(NAMES[l], 0.0) + mm3, 1)
        rep["deleted_pieces"] += 1

## is a piece joined to its vertebra through bone? (reads the ct along lines across the gap)
def _bone_across_gap(ct, piece, main, zooms):
    dist, near = ndimage.distance_transform_edt(~main, sampling=zooms, return_indices=True)
    pts = np.argwhere(piece)
    d = dist[tuple(pts.T)]
    best, gap = -1e9, float(d.min())
    for q in pts[np.argsort(d)[:CFG["separate_bone_lines"]]]:
        b = near[:, q[0], q[1], q[2]]
        n = max(int(np.ceil(np.abs(b - q).max() * 2)), 2)
        line = np.unique(np.linspace(q, b, n + 1).round().astype(int), axis=0)
        gap_pts = [tuple(x) for x in line if not piece[tuple(x)] and not main[tuple(x)]]
        if not gap_pts:
            return True, gap, None
        best = max(best, float(min(ct[x] for x in gap_pts)))
    return best >= CFG["bone_hu"], gap, best

## step 3: specks on thin necks, corner joins, thick-slice floaters and separate bones;
## repeated until nothing changes
def remove_specks(seg, zooms, rep, ct=None):
    vox = float(np.prod(zooms))
    out = seg.copy()
    max_vox = CFG["speck_max_mm3"] / vox
    thick = max(zooms) > max(CFG["speck_neck_radii_mm"])
    thick_vox = CFG["speck_thick_max_mm3"] / vox
    sep_vox = CFG["separate_bone_max_cm3"] * 1e3 / vox
    rep.update(loose_pieces_before=_loose_pieces(seg), recoloured=[], deleted_mm3={},
               deleted_pieces=0, separate_bone=[])
    for _ in range(CFG["speck_rounds"]):
        n0 = len(rep["recoloured"]) + rep["deleted_pieces"]
        for l, sl in _boxes(out, 4):
            cur = out[sl]
            for r in CFG["speck_neck_radii_mm"]:
                for p in _neck_parts(cur == l, _ball(r, zooms), zooms):
                    if p.sum() <= max_vox:
                        _fix_speck(cur, p, l, vox, rep, f"neck < {2 * r:g} mm")
            if thick:
                for p in _neck_parts(cur == l, S6, zooms):
                    ring = cur[ndimage.binary_dilation(p, structure=S26) & ~p]
                    if p.sum() <= thick_vox and not ((ring != 0) & (ring != l)).any():
                        _fix_speck(cur, p, l, vox, rep, "joined through one thick slice")
            for p in _face_parts(cur == l):
                if p.sum() <= max_vox:
                    _fix_speck(cur, p, l, vox, rep, "corner / edge join")
            if ct is not None:
                lab, n = ndimage.label(cur == l, structure=S6)
                if n > 1:
                    sizes = np.bincount(lab.ravel())
                    sizes[0] = 0
                    main = lab == sizes.argmax()
                    for c in np.nonzero(sizes)[0]:
                        p = lab == c
                        if c == sizes.argmax() or sizes[c] > sep_vox:
                            continue
                        ring = cur[ndimage.binary_dilation(p, structure=S26) & ~p]
                        if ((ring != 0) & (ring != l)).any():
                            continue
                        linked, gap, hu = _bone_across_gap(ct[sl], p, main, zooms)
                        if not linked:
                            rep["separate_bone"].append(
                                {"from": NAMES[l], "mm3": round(float(sizes[c]) * vox, 1),
                                 "gap_mm": round(gap, 1), "best_gap_hu": round(hu, 0)})
                            _fix_speck(cur, p, l, vox, rep, "separate bone (no bone across the gap)")
        if len(rep["recoloured"]) + rep["deleted_pieces"] == n0:
            break
    rep["loose_pieces_after"] = _loose_pieces(out)
    rep["changed_mm3"] = round(float((out != seg).sum()) * vox, 1)
    return out

## box around the spine (+ 60 mm) so steps 2-3 run on a small crop
def _spine_crop(seg_full, affine):
    zooms = nib.affines.voxel_sizes(affine)
    lo, hi = spine_bbox(seg_full)
    pad = np.maximum(np.round(CFG["spine_crop_mm"] / zooms).astype(int), 2)
    lo, hi = np.maximum(lo - pad, 0), np.minimum(hi + pad + 1, seg_full.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

## step 1: the big repair, stage by stage
def run_repair(seg_full, ct_full, img):
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    vox_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
    pad = np.maximum(np.round(CFG["crop_margin_mm"] / np.asarray(zooms)).astype(int), 2)
    nz = np.nonzero(seg_full)
    lo = np.maximum(np.array([c.min() for c in nz]) - pad, 0)
    hi = np.minimum(np.array([c.max() for c in nz]) + pad + 1, seg_full.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    seg = seg_full[sl].copy()
    ct = np.ascontiguousarray(ct_full[sl])
    aff = img.affine.copy()
    aff[:3, 3] = (img.affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]
    qa = {"records": [], "bands": [], "flags": [], "polish": [], "smooth": []}
    ## keep the ai's original labels to compare against
    raw = seg.copy()
    ## loose pieces, then pieces wrapped by another vertebra
    seg = stage1_triage(seg, ct, zooms, vox_mm3, qa["records"])
    seg = stage2a_islands(seg, vox_mm3, qa["records"])
    ## redraw suspicious borders at the discs, then polish flat borders
    seg = stage2b_arbitrate(seg, raw, ct, aff, zooms, vox_mm3, qa)
    seg = stage2c_interface_polish(seg, ct, zooms, vox_mm3, qa)
    ## smooth, then give set-aside bone back to its vertebra
    seg = stage3_smooth(seg, ct, zooms, vox_mm3, qa)
    seg = stage2d_reclaim_pool(seg, raw, ct, zooms, vox_mm3, qa)
    ## 3-view recolour, mixed pieces, spinous processes
    seg = stage2e_multiview_recolor(seg, ct, aff, zooms, vox_mm3, qa)
    seg = stage2f_skeleton_relabel(seg, ct, aff, zooms, vox_mm3, qa)
    seg = stage2g_imbrication(seg, ct, aff, zooms, vox_mm3, qa)
    out_full = np.zeros(seg_full.shape, dtype=np.uint8)
    out_full[sl] = seg
    return out_full

## one patient: load -> repair -> cleanup -> specks -> save
def process_case(case_dir: Path, ct_root: Path, out_dir: Path, specks=True):
    t0 = time.time()
    case = case_dir.name
    ## load the ai labels + ct
    img, seg_full, ct_full = load_case(case_dir, ct_root)
    log.info("%s: part 1 - repair", case)
    ## step 1: big repair on the whole spine
    fixed = run_repair(seg_full, ct_full, img)
    del seg_full
    log.info("%s: parts 2-3 - cleanup, specks", case)
    ## crop around the spine for the last two steps
    sl = _spine_crop(fixed, img.affine)
    zooms = tuple(float(z) for z in nib.affines.voxel_sizes(img.affine))
    seg, ct = np.ascontiguousarray(fixed[sl]), np.ascontiguousarray(ct_full[sl])
    del ct_full
    ## step 2: strays, holes and dents
    clean = cleanup(seg, ct, zooms, {})
    sp = {}
    ## step 3: specks (skip with --no_specks)
    if specks:
        clean = remove_specks(clean, zooms, sp, ct)
    fixed[sl] = clean
    ## put the crop back and save the labels
    save_case(out_dir, case, fixed, img)
    print(f"\n{case}  ({time.time() - t0:.0f} s)")
    if sp:
        print(f"  specks: recoloured {len(sp['recoloured'])}, deleted {sp['deleted_pieces']} "
              f"| vertebrae in >1 piece: {len(sp['loose_pieces_before'])} -> {len(sp['loose_pieces_after'])}")

## read the command line and run every patient
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Repair SuPreM vertebra labels using the CT.")
    ap.add_argument("--pred_dir", required=True, type=Path,
                    help="SuPreM vertebra predictions: <case>/segmentations or combined_labels")
    ap.add_argument("--ct_root", required=True, type=Path, help="holds <case>/ct.nii.gz")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--case", action="append", help="case id(s); default: all")
    ap.add_argument("--no_specks", action="store_true", help="skip part 3 (speck removal)")
    a = ap.parse_args()
    cases = sorted(p for p in a.pred_dir.iterdir()
                   if (p / "combined_labels.nii.gz").exists() or (p / "segmentations").is_dir())
    if a.case:
        want = {c for v in a.case for c in v.split(",")}
        cases = [c for c in cases if c.name in want]
    for c in cases:
        process_case(c, a.ct_root, a.out_dir, specks=not a.no_specks)

## run main() when the file is started directly
if __name__ == "__main__":
    main()
