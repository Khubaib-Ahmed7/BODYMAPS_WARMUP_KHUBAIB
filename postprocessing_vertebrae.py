"""postprocessing_vertebrae.py - repair SuPreM vertebra labels with the CT, in one command.

    python postprocessing_vertebrae.py --pred_dir AbdomenAtlasDemoPredict \
        --ct_root AbdomenAtlasDemo --out_dir Final [--case BDMAP_00000031] \
        [--no_hao_check] [--no_specks]

Input : <pred_dir>/<case>/segmentations/vertebrae_*.nii.gz or combined_labels.nii.gz
                                                   (1=L5 ... 5=L1, 6=T12 ... 17=T1,
                                                    18=C7 ... 23=C2, 24=C1)
        <ct_root>/<case>/ct.nii.gz                 (same grid, HU)
Output: <out_dir>/<case>/combined_labels.nii.gz + segmentations/vertebrae_*.nii.gz
        <out_dir>/<case>/review.nii.gz             (1 = Pro and Hao disagree on a body,
                                                    2 = body colour vs anatomy disagree)
        <out_dir>/<case>/landmarks.nii.gz          (ribs coloured by the body they
                                                    attach to, sacrum = 25)
        <out_dir>/<case>/report.json               (cleanup, specks, checks)
        <out_dir>/<case>/pro_qa.json               (every ShapeKit-Pro stage)

Chain (this file replaces pro_postprocessing_vertebrae.py -> finalize_vertebrae.py,
with check_names.py and the Hao pipeline as checks):

  PART 1   ShapeKit-Pro    boundary repair: stages 1, 2a-2g, 3 (see its section).
  PART 2   cleanup         stray pieces take the colour of the vertebra they sit on;
                           holes and pits are filled where the CT is bone.
  PART 3   specks          a piece hanging on by a neck thinner than ~3 mm, or by a
                           corner / edge only, and smaller than speck_max_mm3, takes the
                           colour of the vertebra it sits on, or is deleted. On thick
                           slices also small pieces joined only through one slice.
  PART 4   checks          report only, they never change the labels:
                           a. Hao agreement - the Hao pipeline is run on the same input
                              and its bodies L2..T5 are compared with ours;
                           b. anatomy - vertebrae counted from the sacrum (= L5) and
                              the ribs (lowest = T12, highest = T1).

numpy, scipy, nibabel, cc3d (connected-components-3d), scikit-image.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage, signal
from skimage.segmentation import watershed


# ==========================================================================
# HAO + SHARED HELPERS (names, geometry, cores, ribs, sacrum, I/O)
# Run here as an independent second opinion (PART 4a); finalize and the name
# check use its helpers.
#
#  Pipeline (each editing stage is kept only if its meters do not get worse):
#
#    1. CLEANUP     specks / non-bone pieces deleted; a wrong-coloured piece sitting on
#                   another vertebra takes that vertebra's colour; bone blobs floating
#                   away from the spine are deleted.
#    2. GEOMETRY    spine centre line; mean CT brightness on disks PERPENDICULAR to it
#                   (discs show up as dips even on a curved spine); the body's back edge
#                   per slice (front of the spinal canal).            [idea: ShapeKit-Pro]
#    3. BODIES      vertebral bodies found WITHOUT the AI's colours: shrink the mask until
#                   touching bodies separate into thick "cores", keep cores that stay put
#                   across two shrink sizes.                           [idea: Hao Yu]
#    4. LANDMARKS   from the CT: the sacrum under the lowest body = L5; the lowest body
#                   with a rib = T12; the highest = T1.               [our anatomy check]
#    5. NAMING      the bodies are counted from the landmarks (L5 from the sacrum, T12
#                   from the lowest rib); without a landmark, the count that agrees best
#                   with the AI is used. Names always go up by exactly 1 per body.
#    6. RE-CUT      where a body holds the wrong colour, or the colour change misses a
#                   clearly visible disc, body voxels are re-split along the DARK DISC
#                   SURFACE, column by column (not a flat cut). Arches untouched here.
#    7. SPINOUS     thoracic spinous processes hang down onto the vertebra below, so the
#                   midline posterior strip is re-coloured top-to-bottom: the root takes
#                   the colour of the body at its level and colour only flows DOWNWARD.
#                                                                      [idea: ShapeKit-Pro]
#    8. AUDIT       pieces per label, order, size vs neighbours, borders vs discs,
#                   spinous upward-violation; written to report.json.
#
#  Nothing is added outside the AI's mask; bone is only deleted in stage 1 (specks,
#  non-bone, far floating blobs). Rare anatomy (11/13 rib pairs, extra lumbar
#  vertebra) is not handled. numpy, scipy, nibabel only.
#
# ==========================================================================
log = logging.getLogger("vertebrae")

NAMES = ([""] + ["L5", "L4", "L3", "L2", "L1"] + [f"T{i}" for i in range(12, 0, -1)]
         + [f"C{i}" for i in range(7, 0, -1)])            # NAMES[1] = "L5" ... NAMES[24] = "C1"
ID = {n: i for i, n in enumerate(NAMES) if n}
L5, T12, T1 = ID["L5"], ID["T12"], ID["T1"]
S26 = np.ones((3, 3, 3), dtype=bool)

CFG = {
    "bone_hu": 150.0,
    # 1 cleanup
    "speck_mm3": 30.0, "min_bone_frac": 0.5, "far_mm": 10.0,
    # 2 geometry
    "profile_radius_mm": 10.0, "profile_sigma_mm": 2.5, "centre_smooth_mm": 40.0,
    "body_cyl_mm": 18.0,
    # discs
    "disc_min_hu": 25.0, "disc_clear_hu": 50.0, "disc_sep_mm": 6.0,
    # 3 cores
    "core_radii_mm": (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0), "core_step_mm": 0.5,
    "core_drift_mm": 3.0, "core_min_mm3": 100.0, "core_axis_mm": 25.0, "core_gap_mm": 10.0,
    "missing_body_ratio": 1.6,
    # 4 landmarks
    "rib_reach_mm": 45.0, "rib_medial_mm": 45.0, "rib_attach_mm": 12.0, "rib_len_mm": 35.0,
    "rib_ignore_near_mm": 4.0,    # bone this close to a vertebra is ignored (DISH bridges)
    "rib_min_mm3": 150.0, "rib_max_cm3": 60.0,
    "rib_crop_lat_mm": 90.0, "rib_crop_ant_mm": 50.0, "rib_crop_post_mm": 70.0,
    "sac_search_mm": 50.0, "sac_near_mm": 20.0, "sac_gap_mm": 25.0,
    "sac_min_width_mm": 105.0, "sac_min_cm3": 40.0,
    # 6 re-cut
    "border_tol_mm": 5.0, "purity_min": 0.90, "back_margin_mm": 2.0,
    # 7 spinous
    "corridor_half_mm": 7.0, "canal_mm": 12.0, "strip_mm": 6.0, "upv_tol_mm": 5.0,
    "spinous_max_change_cm3": 6.0,   # per vertebra (as Pro's per-level bound)
    # io
    "crop_margin_mm": 60.0,
}


# =================================================================== I/O ====
def load_case(case_dir: Path, ct_root: Path):
    """Vertebra labels come from SuPreM's vertebrae output: either
    segmentations/vertebrae_<NAME>.nii.gz (preferred, unambiguous by name) or a
    combined_labels.nii.gz in the 1=L5 ... 24=C1 coding. An AbdomenAtlas organ
    folder (aorta, liver, ... and no vertebra masks) is refused, because its
    combined_labels uses organ codes."""
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


def save_case(out_dir: Path, case: str, seg: np.ndarray, img):
    d = out_dir / case
    (d / "segmentations").mkdir(parents=True, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), img.affine, hdr), str(d / "combined_labels.nii.gz"))
    for lid in range(1, 25):
        nib.save(nib.Nifti1Image((seg == lid).astype(np.uint8), img.affine, hdr),
                 str(d / "segmentations" / f"vertebrae_{NAMES[lid]}.nii.gz"))


def axis_frame(affine):
    """Slices must be axial on array axis 2 (true for AbdomenAtlas)."""
    codes = nib.aff2axcodes(affine)
    if codes[0] not in "LR" or codes[1] not in "AP" or codes[2] not in "SI":
        raise ValueError(f"expected axes (L/R, A/P, S/I), got {codes}")
    return {"z_up": codes[2] == "S", "ant": 1 if codes[1] == "A" else -1}


def spine_bbox(seg, f=4):
    """Box around the main spine piece only (far stray blobs must not blow up the
    crop); found on a 4x-subsampled mask so it is cheap."""
    small = np.zeros(tuple(-(-n // f) for n in seg.shape), dtype=bool)
    for a in range(f):
        for b in range(f):
            for c in range(f):
                v = seg[a::f, b::f, c::f] > 0
                small[:v.shape[0], :v.shape[1], :v.shape[2]] |= v
    # vertebrae may not touch each other (clean gaps at the discs), so join pieces
    # that are close (dilate one block) and keep every piece >= 5% of the largest;
    # only small far-away blobs are left out of the crop
    small = ndimage.binary_dilation(small, structure=S26, iterations=2)
    lab, n = ndimage.label(small, structure=S26)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.nonzero(sizes >= 0.05 * sizes.max())[0]
    pts = np.argwhere(np.isin(lab, keep))
    return (np.maximum(pts.min(axis=0) * f - f, 0),
            np.minimum(pts.max(axis=0) * f + 2 * f, np.array(seg.shape) - 1))


def components(mask):
    lab, n = ndimage.label(mask, structure=S26)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab, sizes


# ============================================================ 1 CLEANUP ====
def stage1_cleanup(seg, ct, zooms, rep):
    """Every non-main piece of every label is judged:
       touching another vertebra (most of its labelled border) -> takes that colour;
       tiny or mostly not bone                                  -> deleted;
       far (> far_mm) from its own label and touching nothing   -> deleted;
       otherwise kept (a real process piece)."""
    vox = float(np.prod(zooms))
    out = seg.copy()
    acts = {"recoloured": 0, "deleted_speck_or_soft": 0, "deleted_floating": 0, "kept": 0}
    objs = ndimage.find_objects(seg)
    for lid in range(1, 25):
        if lid > len(objs) or objs[lid - 1] is None:
            continue
        pad = 3 + int(np.ceil(CFG["far_mm"] / min(zooms)))
        sl = tuple(slice(max(s.start - pad, 0), s.stop + pad) for s in objs[lid - 1])
        lab, sizes = components(out[sl] == lid)
        if (sizes > 0).sum() <= 1:
            continue
        main = lab == sizes.argmax()
        near_main = ndimage.distance_transform_edt(~main, sampling=zooms) <= CFG["far_mm"]
        for cid in np.nonzero(sizes)[0]:
            if cid == sizes.argmax():
                continue
            comp = lab == cid
            shell = ndimage.binary_dilation(comp, structure=S26) & ~comp
            ring = out[sl][shell]
            other = ring[(ring != 0) & (ring != lid)]
            bone = float(np.mean(ct[sl][comp] >= CFG["bone_hu"]))
            if other.size and other.size >= 0.5 * (ring != 0).sum():
                ids, n = np.unique(other, return_counts=True)
                out[sl][comp] = int(ids[n.argmax()])
                acts["recoloured"] += 1
            elif sizes[cid] * vox < CFG["speck_mm3"] or bone < CFG["min_bone_frac"]:
                out[sl][comp] = 0
                acts["deleted_speck_or_soft"] += 1
            elif not (comp & near_main).any() and not other.size:
                out[sl][comp] = 0
                acts["deleted_floating"] += 1
            else:
                acts["kept"] += 1
    rep["stage1_cleanup"] = acts
    return out


# =========================================================== 2 GEOMETRY ====
class Geo:
    pass


def _fill(a):
    a = np.array(a, dtype=np.float64)
    ok = ~np.isnan(a)
    if ok.sum() < 2:
        return a
    idx = np.nonzero(ok)[0]
    a[:] = np.interp(np.arange(len(a)), idx, a[idx])
    return a


def build_geometry(seg, ct, zooms, fr):
    g = Geo()
    g.zooms, g.fr = zooms, fr
    nz = seg.shape[2]
    dx, dy, dz = zooms
    M = seg > 0
    bone = ct >= CFG["bone_hu"]
    cx = np.full(nz, np.nan)
    cy = np.full(nz, np.nan)
    present = M.any(axis=(0, 1))
    for z in np.nonzero(present)[0]:
        sl = M[:, :, z] & bone[:, :, z]
        if not sl.any():
            sl = M[:, :, z]
        lab, n = ndimage.label(sl)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        i, j = np.nonzero(lab == sizes.argmax())
        cx[z], cy[z] = i.mean(), j.mean()
    g.zmin, g.zmax = int(np.nonzero(present)[0].min()), int(np.nonzero(present)[0].max())
    w = max(int(CFG["centre_smooth_mm"] / dz) | 1, 3)
    for a in (cx, cy):
        ok = ~np.isnan(a)
        a[ok] = ndimage.uniform_filter1d(a[ok], w, mode="nearest")
    g.cx, g.cy = _fill(cx), _fill(cy)
    P = np.c_[g.cx * dx, g.cy * dy, np.arange(nz) * dz]
    T = np.gradient(ndimage.gaussian_filter1d(P, sigma=8.0 / dz, axis=0), axis=0)
    g.T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-6)
    # perpendicular-disk mean HU profile
    r = CFG["profile_radius_mm"]
    aa, bb = np.meshgrid(np.arange(-r, r + 1.0, 1.0), np.arange(-r, r + 1.0, 1.0))
    keep = aa ** 2 + bb ** 2 <= r * r
    offs = np.c_[aa[keep], bb[keep]]
    rho = np.zeros(nz)
    hu = ct.astype(np.float32)
    for z in range(g.zmin, g.zmax + 1):
        t = g.T[z]
        ref = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0.0, 1, 0])
        u = np.cross(t, ref)
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        pts = P[z] + offs[:, :1] * u + offs[:, 1:2] * v
        vals = ndimage.map_coordinates(hu, (pts / np.asarray(zooms)).T, order=1, mode="nearest")
        rho[z] = float(np.clip(vals, 0, 800).mean())
    g.rho = ndimage.gaussian_filter1d(rho, sigma=max(CFG["profile_sigma_mm"] / dz, 0.8))
    g.rho[:g.zmin] = 0
    g.rho[g.zmax + 1:] = 0
    # body back edge = front of the spinal canal (hole inside the vertebral ring)
    g.post = -fr["ant"]                       # +1 if posterior = increasing j
    front = np.full(nz, np.nan)
    for z in range(g.zmin, g.zmax + 1):
        m2 = M[:, :, z]
        holes = ndimage.binary_fill_holes(m2) & ~m2
        if not holes.any():
            continue
        lab, n = ndimage.label(holes)
        best = None
        for k in range(1, n + 1):
            i, j = np.nonzero(lab == k)
            area = i.size * dx * dy
            d = np.hypot((i.mean() - g.cx[z]) * dx, (j.mean() - g.cy[z]) * dy)
            if 40 <= area <= 700 and d <= 30 and (best is None or d < best[0]):
                best = (d, j)
        if best is not None:
            j = best[1]
            front[z] = j.min() if g.post > 0 else j.max()     # canal edge nearest the body
    if np.isnan(front).all():                 # fallback: back of an 18 mm cylinder
        for z in range(g.zmin, g.zmax + 1):
            i, j = np.nonzero(M[:, :, z])
            sel = ((i - g.cx[z]) * dx) ** 2 + ((j - g.cy[z]) * dy) ** 2 <= CFG["body_cyl_mm"] ** 2
            if sel.sum() > 20:
                front[z] = np.percentile(j[sel], 95 if g.post > 0 else 5)
    g.back = ndimage.median_filter(_fill(front), size=11, mode="nearest")
    return g


def up(z, g):
    """Height in mm, increasing toward the head."""
    return (np.asarray(z, dtype=float) if g.fr["z_up"] else -np.asarray(z, dtype=float)) * g.zooms[2]


def body_region(seg, g, z0, z1):
    """Labelled voxels in front of the body's back edge, slices z0..z1."""
    jj = np.arange(seg.shape[1])[None, :, None]
    back = g.back[None, None, z0:z1 + 1]
    return (seg[:, :, z0:z1 + 1] > 0) & (g.post * (jj - back) * g.zooms[1] <= CFG["back_margin_mm"])


# ================================================================ discs ====
def find_dips(g):
    sig = -g.rho
    peaks, props = signal.find_peaks(sig, prominence=CFG["disc_min_hu"] * 0.4,
                                     distance=max(int(round(CFG["disc_sep_mm"] / g.zooms[2])), 1))
    return [{"z": int(p), "prom": float(pr)} for p, pr in zip(peaks, props["prominences"])
            if g.rho[p] > 0]


def disc_between(dips, za, zb):
    lo, hi = sorted((za, zb))
    inside = [d for d in dips if lo < d["z"] < hi]
    if not inside:
        return None
    best = max(inside, key=lambda d: d["prom"])
    return best if best["prom"] >= CFG["disc_min_hu"] else None


# ================================================================ 3 CORES ===
def _cores_at(depth, r, seg, ct, g, vox, off):
    """Cores at shrink radius r inside the tube box; `off` = box corner, so
    centres come back in crop coordinates."""
    lab, sizes = components(depth > r)
    objs = ndimage.find_objects(lab)
    recs = {}
    for cid in np.nonzero(sizes * vox >= CFG["core_min_mm3"])[0]:
        sl = objs[cid - 1]
        pts = np.argwhere(lab[sl] == cid) + np.array([s.start for s in sl])
        cen = pts.mean(axis=0) + off
        zc = int(round(cen[2]))
        if np.hypot((cen[0] - g.cx[zc]) * g.zooms[0], (cen[1] - g.cy[zc]) * g.zooms[1]) > CFG["core_axis_mm"]:
            continue
        flat = np.ravel_multi_index(pts.T, lab.shape)
        if np.mean(ct.ravel()[flat] >= CFG["bone_hu"]) < 0.5:
            continue
        ev = np.bincount(seg.ravel()[flat], minlength=25)[:25]
        recs[int(cid)] = {"flat": flat, "cen": cen, "size": len(flat), "ev": ev}
    return lab, recs


def find_cores(seg, ct, g):
    """Hao's idea: label-independent bodies = thick cores of the mask that survive
    one-to-one (<= core_drift_mm movement) at two shrink sizes. The shrink size
    that gives the most cores with no two in one body is used."""
    zooms, vox = g.zooms, float(np.prod(g.zooms))
    dom = (seg > 0) & (ct >= -250)
    tube = CFG["core_axis_mm"] + 25.0
    ii = np.arange(seg.shape[0])[:, None]
    jj = np.arange(seg.shape[1])[None, :]
    for z in range(seg.shape[2]):
        if z < g.zmin or z > g.zmax:
            dom[:, :, z] = False
            continue
        dom[:, :, z] &= ((np.abs(ii - g.cx[z]) * zooms[0] <= tube)
                         & (np.abs(jj - g.cy[z]) * zooms[1] <= tube))
    o = ndimage.find_objects(dom.astype(np.uint8))
    if not o or o[0] is None:
        return [], None
    sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in o[0])
    off = np.array([s.start for s in sl])
    depth = ndimage.distance_transform_edt(dom[sl], sampling=zooms).astype(np.float32)
    seg_t, ct_t = seg[sl], ct[sl]
    best = None
    for r in CFG["core_radii_mm"]:
        _, a = _cores_at(depth, r, seg_t, ct_t, g, vox, off)
        labb, b = _cores_at(depth, r + CFG["core_step_mm"], seg_t, ct_t, g, vox, off)
        stable = []
        for rec in a.values():
            kids, n = np.unique(labb.ravel()[rec["flat"]], return_counts=True)
            kids = [(int(k), int(m)) for k, m in zip(kids, n) if int(k) in b]
            if len(kids) != 1:
                continue
            kid, overlap = kids[0]
            ch = b[kid]
            if overlap < 0.95 * ch["size"] or ch["size"] < 0.1 * rec["size"]:
                continue
            if np.linalg.norm((ch["cen"] - rec["cen"]) * np.asarray(zooms)) > CFG["core_drift_mm"]:
                continue
            stable.append(dict(rec, kid=kid))
        kids = [s["kid"] for s in stable]
        stable = [s for s in stable if kids.count(s["kid"]) == 1]
        hs = sorted(float(up(s["cen"][2], g)) for s in stable)
        regular = all(y - x >= CFG["core_gap_mm"] for x, y in zip(hs[:-1], hs[1:]))
        key = (regular, len(stable), -r)
        if best is None or key > best[0]:
            best = (key, r, stable)
    (_, _, _), r, stable = best
    stable.sort(key=lambda s: float(up(s["cen"][2], g)))
    cores = []
    for s in stable:
        ev = s["ev"].copy()
        ev[0] = 0
        ai = int(ev.argmax())
        second = int(np.argsort(ev)[-2]) if (ev > 0).sum() > 1 else 0
        cores.append({"z": float(s["cen"][2]), "x": float(s["cen"][0]), "y": float(s["cen"][1]),
                      "ai": ai, "purity": round(float(ev[ai] / max(ev.sum(), 1)), 3),
                      "second": second})
    return cores, r


# ============================================================ 4 LANDMARKS ===
def _core_of_height(h, cores, g):
    """Index of the body whose slab (midpoints to its neighbours) holds height h."""
    hs = [float(up(c["z"], g)) for c in cores]
    k = int(np.argmin(np.abs(np.array(hs) - h)))
    return k


def find_ribs(seg, ct, g, cores):
    """Bone outside the vertebra mask, whose inner end is within rib_attach_mm of a
    vertebra and which runs > rib_reach_mm out sideways = a rib on that body."""
    dx, dy, dz = g.zooms
    vox = dx * dy * dz
    zs = sorted([int(round(cores[0]["z"])), int(round(cores[-1]["z"]))])
    pitch = abs(cores[-1]["z"] - cores[0]["z"]) / max(len(cores) - 1, 1)
    z0 = max(int(zs[0] - pitch), 0)
    z1 = min(int(zs[1] + pitch) + 1, seg.shape[2])
    xs, ys = g.cx[z0:z1], g.cy[z0:z1]
    x0 = max(int(xs.min() - CFG["rib_crop_lat_mm"] / dx), 0)
    x1 = min(int(xs.max() + CFG["rib_crop_lat_mm"] / dx) + 1, seg.shape[0])
    a_mm, p_mm = CFG["rib_crop_ant_mm"], CFG["rib_crop_post_mm"]
    lo_mm, hi_mm = (p_mm, a_mm) if g.fr["ant"] > 0 else (a_mm, p_mm)
    y0 = max(int(ys.min() - lo_mm / dy), 0)
    y1 = min(int(ys.max() + hi_mm / dy) + 1, seg.shape[1])
    sseg = seg[x0:x1, y0:y1, z0:z1]
    # Bone hugging the vertebrae (DISH / osteophyte bridges, joint edges) can weld
    # every rib into one huge piece; ignore a thin shell around the vertebra mask
    # so each rib stays its own piece. Rib heads are then a few mm from the mask.
    it = max(int(round(CFG["rib_ignore_near_mm"] / min(g.zooms))), 1)
    near_v = ndimage.binary_dilation(sseg > 0, structure=S26, iterations=it)
    other = (ct[x0:x1, y0:y1, z0:z1] >= CFG["bone_hu"]) & ~near_v
    del near_v
    lab, sizes = components(other)
    objs = ndimage.find_objects(lab)
    diag = {"pieces": int((sizes > 0).sum()), "too_small": 0, "too_big": 0,
            "short_or_not_lateral": 0, "below_spine": 0, "inner_end_far": 0,
            "not_touching_vertebra": 0, "accepted": 0}
    low_h = float(up(cores[0]["z"], g)) - 0.6 * pitch * dz
    rad = np.ceil(CFG["rib_attach_mm"] / np.asarray(g.zooms)).astype(int)
    ribs = []
    mask = np.zeros(seg.shape, dtype=np.uint8)
    diag["too_small"] = int(((sizes > 0) & (sizes * vox < CFG["rib_min_mm3"])).sum())
    diag["too_big"] = int((sizes * vox > CFG["rib_max_cm3"] * 1e3).sum())
    ok_size = (sizes * vox >= CFG["rib_min_mm3"]) & (sizes * vox <= CFG["rib_max_cm3"] * 1e3)
    for cid in np.nonzero(ok_size)[0]:
        sl = objs[cid - 1]
        pts = np.argwhere(lab[sl] == cid) + np.array([s.start for s in sl])
        za = pts[:, 2] + z0
        lat = (pts[:, 0] + x0 - g.cx[za]) * dx
        ap = (pts[:, 1] + y0 - g.cy[za]) * dy * g.fr["ant"]
        ext = np.linalg.norm((pts.max(axis=0) - pts.min(axis=0)) * np.asarray(g.zooms))
        if np.abs(lat).max() < CFG["rib_reach_mm"] or ext < CFG["rib_len_mm"]:
            diag["short_or_not_lateral"] += 1
            continue
        if float(up(za, g).min()) < low_h - 10.0:
            diag["below_spine"] += 1                    # reaches below the spine: pelvis
            continue
        dist = np.hypot(lat, ap)
        m = int(dist.argmin())
        if dist[m] > CFG["rib_medial_mm"]:
            diag["inner_end_far"] += 1
            continue
        p = pts[m] + np.array([x0, y0, z0])
        box = tuple(slice(max(int(p[k]) - rad[k], 0), min(int(p[k]) + rad[k] + 1, seg.shape[k]))
                    for k in range(3))
        nb = np.argwhere(seg[box] > 0)
        if nb.size == 0:
            diag["not_touching_vertebra"] += 1
            continue
        nb = nb + np.array([s.start for s in box])
        dmm = np.linalg.norm((nb - p) * np.asarray(g.zooms), axis=1)
        if dmm.min() > CFG["rib_attach_mm"]:
            diag["not_touching_vertebra"] += 1
            continue
        diag["accepted"] += 1
        k = _core_of_height(float(up(p[2], g)), cores, g)
        ribs.append({"body": k, "side": "+x" if lat[m] > 0 else "-x",
                     "reach_mm": round(float(np.abs(lat).max()), 1)})
        full = pts + np.array([x0, y0, z0])
        mask[tuple(full.T)] = 100 + k            # renamed to vertebra colours later
    return ribs, mask, diag


def find_sacrum(seg, ct, g, cores):
    """Wide bone directly under the lowest body = sacrum -> that body is L5."""
    dx, dy, dz = g.zooms
    c0 = cores[0]
    pitch = abs(cores[1]["z"] - c0["z"]) if len(cores) > 1 else 25.0 / dz
    down = -1 if g.fr["z_up"] else 1
    zb = int(round(c0["z"] + down * 0.5 * pitch))          # about the bottom of the body
    n = int(round(CFG["sac_search_mm"] / dz))
    zz = [zb + down * k for k in range(0, n)]
    zz = [z for z in zz if 0 <= z < seg.shape[2]]
    if len(zz) < 0.6 * n:
        return {"status": "OUT_OF_VIEW"}, None
    za, zc = min(zz), max(zz) + 1
    cx, cy = g.cx[g.zmin if g.fr["z_up"] else g.zmax], g.cy[g.zmin if g.fr["z_up"] else g.zmax]
    x0, x1 = max(int(cx - 110 / dx), 0), min(int(cx + 110 / dx) + 1, seg.shape[0])
    y0, y1 = max(int(cy - 70 / dy), 0), min(int(cy + 70 / dy) + 1, seg.shape[1])
    other = (ct[x0:x1, y0:y1, za:zc] >= CFG["bone_hu"]) & (seg[x0:x1, y0:y1, za:zc] == 0)
    lab, sizes = components(other)
    ii, jj = np.ogrid[x0:x1, y0:y1]
    near = ((ii - cx) * dx) ** 2 + ((jj - cy) * dy) ** 2 <= CFG["sac_near_mm"] ** 2
    ids = set(np.unique(lab[near]).tolist()) - {0}
    if not ids:
        return {"status": "ABSENT"}, None
    m = np.isin(lab, list(ids))
    pts = np.argwhere(m)
    width = float((pts[:, 0].max() - pts[:, 0].min() + 1) * dx)
    vol = float(m.sum() * dx * dy * dz / 1e3)
    near_pts = np.argwhere(m & near[:, :, None])
    top = (near_pts[:, 2].min() if down > 0 else near_pts[:, 2].max()) + za
    gap = abs(top - zb) * dz
    ok = width >= CFG["sac_min_width_mm"] and vol >= CFG["sac_min_cm3"] and gap <= CFG["sac_gap_mm"]
    mask = np.zeros(seg.shape, dtype=np.uint8)
    mask[tuple((pts + np.array([x0, y0, za])).T)] = 25
    return {"status": "PRESENT" if ok else "NOT_SACRUM_LIKE", "width_mm": round(width, 1),
            "cm3": round(vol, 1), "gap_mm": round(float(gap), 1)}, mask


# =============================================================== 5 NAMING ===
def name_bodies(cores, ribs, sacrum, g):
    """Count the bodies from the landmarks. Bodies are split into runs wherever a
    body seems to be missing (gap > missing_body_ratio x local spacing); each run is
    named from its landmark, or by the count that best matches the AI."""
    n = len(cores)
    hs = np.array([float(up(c["z"], g)) for c in cores])
    gaps = np.diff(hs)
    runs, cur = [], [0]
    for i, gp in enumerate(gaps):
        nb = [gaps[j] for j in range(max(0, i - 2), min(len(gaps), i + 3)) if j != i]
        if nb and gp > CFG["missing_body_ratio"] * float(np.median(nb)):
            runs.append(cur)
            cur = []
        cur.append(i + 1)
    runs.append(cur)
    ai = [c["ai"] for c in cores]
    sides = {}
    for r in ribs:
        sides.setdefault(r["body"], set()).add(r["side"])
    rib_bodies = sorted(b for b, sd in sides.items() if len(sd) == 2)   # ribs on both sides
    anchors = []
    if sacrum and sacrum.get("status") == "PRESENT":
        anchors.append({"rule": "sacrum under the lowest body -> L5", "body": 0, "name": L5})
    if rib_bodies:
        lo, hi = rib_bodies[0], rib_bodies[-1]
        if lo > 0:
            anchors.append({"rule": "lowest body with a rib -> T12", "body": lo, "name": T12})
        if hi < n - 1 and hi - lo >= 6:
            anchors.append({"rule": "highest body with a rib -> T1", "body": hi, "name": T1})
    names = list(ai)
    decisions = []
    for run in runs:
        offs = [a["name"] - a["body"] for a in anchors if a["body"] in run]

        def agree(o):
            return sum(1 for i in run if o + i == ai[i])
        if offs:
            o = max(set(offs), key=lambda v: (offs.count(v), agree(v)))
            why = "landmarks" + ("" if len(set(offs)) == 1 else " (disagreeing; best-supported kept)")
        else:
            o = max({ai[i] - i for i in run}, key=agree)
            why = "no landmark in this run: count that best matches the AI"
        for i in run:
            if 1 <= o + i <= 24:
                names[i] = o + i
        decisions.append({"bodies": [NAMES[ai[i]] for i in run],
                          "named": [NAMES[names[i]] for i in run], "by": why})
    return names, anchors, decisions


# =========================================== 6 RENAME + RE-CUT (bodies) ===
def border_meter(seg, g, dips):
    """For consecutive labels along the spine: border vs clear disc (mm), and body
    purity. Used by gates and the audit."""
    nz = seg.shape[2]
    reg = body_region(seg, g, 0, nz - 1)
    counts = np.zeros((nz, 25), np.int64)
    for z in range(g.zmin, g.zmax + 1):
        v = seg[:, :, z][reg[:, :, z]]
        if v.size:
            counts[z] = np.bincount(v, minlength=25)[:25]
    cen = {}
    for l in range(1, 25):
        c = counts[:, l].astype(float)
        if c.sum() > 0:
            cen[l] = float((c * np.arange(nz)).sum() / c.sum())
    order = sorted(cen, key=lambda l: float(up(cen[l], g)))
    pairs = []
    for a, b in zip(order[:-1], order[1:]):
        za, zb = int(round(cen[a])), int(round(cen[b]))
        step = 1 if zb >= za else -1
        walk = list(range(za, zb + step, step))
        ca, cb = counts[walk, a], counts[walk, b]
        fb = next((k for k in range(len(walk)) if cb[k] > ca[k]), None)
        border = None
        if fb is not None:
            la = max((k for k in range(fb) if ca[k] > cb[k]), default=0)
            border = (walk[la] + walk[fb]) / 2
        d = disc_between(dips, za, zb)
        off = None
        if d is not None and d["prom"] >= CFG["disc_clear_hu"] and border is not None:
            off = round(abs(border - d["z"]) * g.zooms[2], 1)
        pairs.append({"lower": NAMES[a], "upper": NAMES[b], "border_offset_mm": off})
    return pairs, counts


def recut_pair(seg, ct, g, za, zb, la, lb, dips):
    """Body voxels between two body centres: above the dark disc surface -> lb,
    below -> la. The surface is the darkest z-smoothed CT point per (x, y) column,
    searched around the disc on the profile (or the midpoint), then smoothed."""
    lo, hi = sorted((int(round(za)), int(round(zb))))
    dz = g.zooms[2]
    d = disc_between(dips, za, zb)
    zc = d["z"] if d is not None else (lo + hi) // 2
    w = max(int(round(min(8.0, 0.35 * (hi - lo) * dz) / dz)), 1)
    w0, w1 = max(zc - w, lo), min(zc + w, hi)
    reg = body_region(seg, g, lo, hi)
    cols = reg.any(axis=2)
    if not cols.any():
        return seg
    ii, jj = np.nonzero(cols)
    i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
    hu = ndimage.gaussian_filter1d(ct[i0:i1, j0:j1, w0:w1 + 1].astype(np.float32),
                                   sigma=max(1.5 / dz, 0.6), axis=2)
    zs = ndimage.median_filter(hu.argmin(axis=2) + w0, size=5)
    zz = np.arange(lo, hi + 1)[None, None, :]
    above = (zz > zs[:, :, None]) if g.fr["z_up"] else (zz < zs[:, :, None])
    out = seg.copy()
    blk = out[i0:i1, j0:j1, lo:hi + 1]
    r = reg[i0:i1, j0:j1, :]
    blk[r & above] = lb
    blk[r & ~above] = la
    return out


def absorb_orphans(seg, lids):
    """Non-main pieces of these labels take the colour that surrounds them."""
    out = seg
    for lid in lids:
        o = ndimage.find_objects((out == lid).astype(np.uint8))
        if not o or o[0] is None:
            continue
        sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in o[0])
        lab, sizes = components(out[sl] == lid)
        if (sizes > 0).sum() <= 1:
            continue
        for cid in np.nonzero(sizes)[0]:
            if cid == sizes.argmax():
                continue
            comp = lab == cid
            shell = ndimage.binary_dilation(comp, structure=S26) & ~comp
            v = out[sl][shell]
            v = v[(v != 0) & (v != lid)]
            if v.size:
                ids, n = np.unique(v, return_counts=True)
                out[sl][comp] = int(ids[n.argmax()])
    return out


def quick_meter(seg, vox):
    """Cheap problem count used by the gates: missing names in the middle of the
    spine (weighted 3: a vanished vertebra is a naming error), vertebrae in
    pieces, and vertebrae far smaller/larger than their neighbours."""
    vol = np.bincount(seg.ravel(), minlength=25)[:25].astype(float) * vox / 1e3
    present = [l for l in range(1, 25) if vol[l] > 0.5]
    missing = [NAMES[l] for l in range(min(present), max(present) + 1)
               if l not in present and l < 23] if present else []
    size = []
    for i, l in enumerate(present):
        if l >= 23:
            continue
        nb = [vol[present[j]] for j in range(max(0, i - 2), min(len(present), i + 3))
              if j != i and present[j] < 23]
        if nb:
            r = vol[l] / float(np.median(nb))
            if r < 0.6 or r > 1.6:
                size.append(NAMES[l])
    frag = []
    objs = ndimage.find_objects(seg)
    for l in present:
        if l <= len(objs) and objs[l - 1] is not None:
            _, sz = components(seg[objs[l - 1]] == l)
            if (sz * vox > 50).sum() > 1:
                frag.append(NAMES[l])
    return {"missing": missing, "size": size, "fragmented": frag,
            "score": 3 * len(missing) + len(size) + len(frag)}


def relabel_arches(seg, g, z0, z1, labels):
    """Back parts (behind the body's back edge) on slices z0..z1 whose colour is in
    `labels` take the colour of the BODY on the same slice, where one body clearly
    owns that slice. Pedicles, laminae and transverse processes sit at their own
    body's height; disc slices (no clear owner) are left alone, and the hanging
    spinous processes are fixed afterwards by stage 7."""
    out = seg.copy()
    reg = body_region(seg, g, z0, z1)
    lab_set = np.array(sorted(labels), dtype=np.uint8)
    for k, z in enumerate(range(z0, z1 + 1)):
        v = seg[:, :, z][reg[:, :, k]]
        v = v[v > 0]
        if v.size < 30:
            continue
        cnt = np.bincount(v, minlength=25)
        owner = int(cnt.argmax())
        if cnt[owner] < 0.7 * v.size:
            continue
        back = (seg[:, :, z] > 0) & ~reg[:, :, k] & np.isin(seg[:, :, z], lab_set)
        out[:, :, z][back] = owner
    return out


def stage6_rename_recut(seg, ct, g, cores, names, dips, rep):
    """(a) RENAME: every AI colour that is the majority colour of exactly one body
    takes that body's new name, everywhere (arches included). This alone fixes a
    whole-spine naming shift.
    (b) RE-CUT: neighbouring bodies whose names differ by exactly 1 and that are not
    clean (a body holds a second colour, or the colour change misses a clear disc)
    are re-split along the dark disc surface. Consecutive such pairs form a group;
    within a group the back parts follow the body on the same slice. Each group is
    kept only if the problem count (missing names x3, pieces, sizes) does not rise.
    Bodies that are clean and correctly named are not touched."""
    vox = float(np.prod(g.zooms))
    lut = np.arange(256, dtype=np.uint8)
    maj = {}
    for i, c in enumerate(cores):
        maj.setdefault(c["ai"], []).append(i)
    mapping = {}
    for a, idx in maj.items():
        if len(idx) == 1 and names[idx[0]] != a:
            mapping[NAMES[a]] = NAMES[names[idx[0]]]
            lut[a] = names[idx[0]]
    out = lut[seg]
    pairs_before, counts = border_meter(out, g, dips)
    off = {(p["lower"], p["upper"]): p["border_offset_mm"] for p in pairs_before}
    todo = []
    for i in range(len(cores) - 1):
        if names[i + 1] - names[i] != 1:          # not neighbours (e.g. C7 core -> C2 core)
            continue
        a, b = cores[i], cores[i + 1]
        lo, hi = sorted((int(round(a["z"])), int(round(b["z"]))))
        mid = (lo + hi) // 2
        h = counts[lo:hi + 1]
        below = h[: mid - lo + 1].sum(axis=0) if g.fr["z_up"] else h[mid - lo:].sum(axis=0)
        above = h[mid - lo:].sum(axis=0) if g.fr["z_up"] else h[: mid - lo + 1].sum(axis=0)
        pur_lo = below[names[i]] / max(below[1:].sum(), 1)
        pur_hi = above[names[i + 1]] / max(above[1:].sum(), 1)
        o = off.get((NAMES[names[i]], NAMES[names[i + 1]]))
        if pur_lo < CFG["purity_min"] or pur_hi < CFG["purity_min"] or (o is not None and o > CFG["border_tol_mm"]):
            todo.append({"i": i, "pair": f"{NAMES[names[i]]}/{NAMES[names[i + 1]]}",
                         "purity_below": round(float(pur_lo), 3),
                         "purity_above": round(float(pur_hi), 3), "border_offset_mm": o})
    groups = []
    for t in todo:
        if groups and t["i"] == groups[-1][-1]["i"] + 1:
            groups[-1].append(t)
        else:
            groups.append([t])
    grecs = []
    for grp in groups:
        i0, i1 = grp[0]["i"], grp[-1]["i"] + 1
        cand = out
        for t in grp:
            i = t["i"]
            cand = recut_pair(cand, ct, g, cores[i]["z"], cores[i + 1]["z"], names[i], names[i + 1], dips)
        zc = [cores[k]["z"] for k in range(i0, i1 + 1)]
        pitch = (max(zc) - min(zc)) / max(i1 - i0, 1)
        z0 = max(int(min(zc) - 0.6 * pitch), 0)
        z1 = min(int(max(zc) + 0.6 * pitch), seg.shape[2] - 1)
        labels = {names[k] for k in range(i0, i1 + 1)} | {cores[k]["ai"] for k in range(i0, i1 + 1)}
        cand = relabel_arches(cand, g, z0, z1, labels)
        cand = absorb_orphans(cand, sorted(labels))
        m0, m1 = quick_meter(out, vox), quick_meter(cand, vox)
        ok = m1["score"] <= m0["score"]
        grecs.append({"pairs": [t["pair"] for t in grp], "details": grp,
                      "problems_before": m0, "problems_after": m1,
                      "result": "ACCEPTED" if ok else "REVERTED"})
        if ok:
            out = cand
    rep["stage6_rename_recut"] = {"colour_renames": mapping, "groups": grecs}
    return out


# ========================================================== 7 SPINOUS ====
def _corridor(seg, g, z):
    """Midline strip behind the spinal canal on slice z (spinous processes)."""
    dx, dy = g.zooms[0], g.zooms[1]
    ii = np.arange(seg.shape[0])[:, None]
    jj = np.arange(seg.shape[1])[None, :]
    behind = g.post * (jj - g.back[z]) * dy
    return (np.abs(ii - g.cx[z]) * dx <= CFG["corridor_half_mm"]) & (behind >= CFG["canal_mm"]), behind


def body_bands(seg, g):
    reg = body_region(seg, g, 0, seg.shape[2] - 1)
    bands = {}
    zidx = np.arange(seg.shape[2])
    for l in range(1, 25):
        c = (reg & (seg == l)).sum(axis=(0, 1)).astype(float)
        if c.sum() < 50:
            continue
        cw = np.cumsum(c) / c.sum()
        a, b = zidx[np.searchsorted(cw, 0.04)], zidx[min(np.searchsorted(cw, 0.96), len(zidx) - 1)]
        bands[l] = (min(a, b), max(a, b))
    return bands


def upward_violation(seg, ct, g, bands):
    """Corridor bone carrying a label ABOVE that label's own body (+ tolerance):
    impossible, spinous processes only hang downward. cm3."""
    tol = CFG["upv_tol_mm"]
    tops = {l: float(max(up(b[0], g), up(b[1], g))) for l, b in bands.items()}
    n = 0
    for z in range(g.zmin, g.zmax + 1):
        corr, _ = _corridor(seg, g, z)
        v = seg[:, :, z][corr & (ct[:, :, z] >= CFG["bone_hu"])]
        h = float(up(z, g))
        n += sum(int((v == l).sum()) for l in np.unique(v) if l and l in tops and h > tops[l] + tol)
    return n * float(np.prod(g.zooms)) / 1e3


def stage7_spinous(seg, ct, g, rep):
    """Top-to-bottom sweep of the midline posterior strip. On each slice the strip
    nearest the canal (the spinous root) takes the colour of the BODY at that
    level; further back, bone takes the colour it had one slice higher (so colour
    only flows down the hanging process); anything left takes the nearest assigned
    colour in-plane. Kept only if the upward-violation meter drops and no label
    changes by more than spinous_max_change."""
    bands = body_bands(seg, g)
    before = upward_violation(seg, ct, g, bands)
    out = seg.copy()
    reg_lab = {}
    for z in range(g.zmin, g.zmax + 1):
        reg = body_region(seg, g, z, z)[:, :, 0]
        v = seg[:, :, z][reg]
        v = v[v > 0]
        if v.size > 20:
            reg_lab[z] = int(np.bincount(v).argmax())
    zs = range(g.zmax, g.zmin - 1, -1) if g.fr["z_up"] else range(g.zmin, g.zmax + 1)
    prev = None
    for z in zs:
        corr, behind = _corridor(seg, g, z)
        C = corr & (seg[:, :, z] > 0) & (ct[:, :, z] >= CFG["bone_hu"])
        if not C.any():
            prev = None
            continue
        new = np.zeros(C.shape, np.uint8)
        if z in reg_lab:
            strip = C & (behind <= CFG["canal_mm"] + CFG["strip_mm"])
            new[strip] = reg_lab[z]
        if prev is not None:
            flow = C & (new == 0) & (prev > 0)
            new[flow] = prev[flow]
        rest = C & (new == 0)
        if rest.any() and (new > 0).any():
            _, (ia, ja) = ndimage.distance_transform_edt(new == 0, return_indices=True)
            new[rest] = new[ia[rest], ja[rest]]
        elif rest.any():
            new[rest] = seg[:, :, z][rest]
        out[:, :, z][C] = new[C]
        prev = np.where(C, new, 0)
    after = upward_violation(out, ct, g, bands)
    vox = float(np.prod(g.zooms))
    ch = {}
    for l in range(1, 25):
        v0 = int((seg == l).sum())
        if v0:
            ch[NAMES[l]] = round(abs(int((out == l).sum()) - v0) * vox / 1e3, 2)
    worst = max(ch.values()) if ch else 0.0
    ok = after < before and worst <= CFG["spinous_max_change_cm3"]
    rep["stage7_spinous"] = {"upward_violation_cm3": [round(before, 2), round(after, 2)],
                             "max_label_change_cm3": worst,
                             "result": "ACCEPTED" if ok else "REVERTED"}
    return out if ok else seg


# ============================================================ 8 AUDIT ====
def audit(seg, g, dips, ct):
    vox = float(np.prod(g.zooms))
    rows = {}
    objs = ndimage.find_objects(seg)
    for l in range(1, 25):
        if l > len(objs) or objs[l - 1] is None:
            continue
        sl = objs[l - 1]
        m = seg[sl] == l
        _, sizes = components(m)
        idx = np.argwhere(m)
        rows[l] = {"cm3": round(m.sum() * vox / 1e3, 1),
                   "pieces": int((sizes * vox > 50).sum()),
                   "h": float(up(idx[:, 2].mean() + sl[2].start, g)),
                   "edge": any(s.start == 0 or s.stop == n for s, n in zip(sl, seg.shape))}
    ls = sorted(rows)
    order = sum(1 for a, b in zip(ls[:-1], ls[1:]) if rows[b]["h"] <= rows[a]["h"])
    size_flags = []
    for i, l in enumerate(ls):
        if l >= 23 or rows[l]["edge"]:
            continue
        nb = [rows[ls[j]]["cm3"] for j in range(max(0, i - 2), min(len(ls), i + 3))
              if j != i and ls[j] < 23 and not rows[ls[j]]["edge"]]
        if nb:
            r = rows[l]["cm3"] / float(np.median(nb))
            if r < 0.6 or r > 1.6:
                size_flags.append(f"{NAMES[l]} {r:.2f}x neighbours")
    pairs, _ = border_meter(seg, g, dips)
    off = [f"{p['lower']}/{p['upper']} {p['border_offset_mm']} mm" for p in pairs
           if p["border_offset_mm"] is not None and p["border_offset_mm"] > CFG["border_tol_mm"]]
    upv = upward_violation(seg, ct, g, body_bands(seg, g))
    frag = [NAMES[l] for l in ls if rows[l]["pieces"] > 1]
    missing = [NAMES[l] for l in range(min(ls), max(ls) + 1) if l not in rows and l < 23] if ls else []
    return {"fragmented": frag, "missing_names": missing, "order_violations": order,
            "size_flags": size_flags,
            "borders_off_disc": off, "spinous_upward_cm3": round(upv, 2),
            "volumes_cm3": {NAMES[l]: rows[l]["cm3"] for l in ls},
            "bad": len(frag) + 3 * len(missing) + order + len(size_flags) + len(off)}


# ============================================================== DRIVER ====
def refine(seg, ct, affine, rep):
    """All stages on one (cropped) case. Returns the refined labels + landmark mask."""
    zooms = tuple(float(z) for z in nib.affines.voxel_sizes(affine))
    fr = axis_frame(affine)
    t0 = time.time()
    s1 = stage1_cleanup(seg, ct, zooms, rep)
    g = build_geometry(s1, ct, zooms, fr)
    dips = find_dips(g)
    rep["audit_input"] = audit(seg, g, dips, ct)
    rep["audit_after_cleanup"] = audit(s1, g, dips, ct)
    cores, radius = find_cores(s1, ct, g)
    rep["bodies"] = {"shrink_radius_mm": radius, "n": len(cores),
                     "ai_colour_bottom_up": [NAMES[c["ai"]] for c in cores],
                     "purity": [c["purity"] for c in cores]}
    lm = np.zeros(seg.shape, np.uint8)
    if len(cores) < 3:
        rep["note"] = "fewer than 3 bodies found; only cleanup applied"
        return s1, lm
    ribs, rmask, rib_diag = find_ribs(s1, ct, g, cores)
    sac, smask = find_sacrum(s1, ct, g, cores)
    names, anchors, decisions = name_bodies(cores, ribs, sac, g)
    rep["landmarks"] = {"rib_bodies_bottom_up_index": sorted({r["body"] for r in ribs}),
                        "ribs": ribs, "rib_search": rib_diag, "sacrum": sac,
                        "anchors": [dict(a, name=NAMES[a["name"]]) for a in anchors]}
    rep["naming"] = {"ai_bottom_up": [NAMES[c["ai"]] for c in cores],
                     "final_bottom_up": [NAMES[n] for n in names], "runs": decisions}
    for k in range(len(cores)):
        lm[rmask == 100 + k] = names[k]
    if smask is not None:
        lm = np.maximum(lm, smask)

    s6 = stage6_rename_recut(s1, ct, g, cores, names, dips, rep)
    a1, a6 = rep["audit_after_cleanup"], audit(s6, g, dips, ct)
    ok6 = a6["bad"] <= a1["bad"] and not a6["missing_names"]
    rep["stage6_rename_recut"]["result"] = "ACCEPTED" if ok6 else "REVERTED"
    rep["stage6_rename_recut"]["bad_before_after"] = [a1["bad"], a6["bad"]]
    cur = s6 if ok6 else s1
    s7 = stage7_spinous(cur, ct, g, rep)
    rep["audit_output"] = audit(s7, g, dips, ct)
    rep["seconds"] = round(time.time() - t0, 1)
    return s7, lm


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)




# ==========================================================================
# PART 1  SHAPEKIT-PRO
#
#  ShapeKit-Pro: evidence-edited, anatomy-audited post-processing for
#  vertebra segmentations.
#
#  ABSTRACT. Segmentation models label vertebrae well until the anatomy gets
#  hard: scoliosis, collapsed discs, and ankylosis make them fragment labels,
#  misplace whole levels, and paint one vertebra's bone with a neighbor's
#  color, while the standing delete-only cleanup tools remove real bone and can
#  never return a mislabeled process to its true owner. This file solves the
#  problem by RECOLORING INSIDE AN ENVELOPE instead of deleting: the raw
#  prediction is treated as the outer boundary and is never grown or erased,
#  the CT image is the only editor (HU valleys, bone corridors, and thickness
#  waists justify every change), and each repair stage carries its own defect
#  meter and reverts itself in full whenever it cannot prove improvement. Nine
#  gated stages repair fragmentation, disc-plane assignment, posterior-arch
#  ownership, and the one-level-down spinous chains that defeat every distance
#  and thickness race. With one parameter set on both AbdomenAtlasDemo cases
#  the output reaches zero structural audit flags, restores L1 from 23.3 to
#  62.9 cm3 where the delete-only baseline shrinks it to 11.6, and re-attaches
#  every spinous process to its own vertebra (upward violation 3.15 -> 0.21
#  cm3), all verified at native resolution.
#
#  Author:  Abhijit Das  (abhijit.das@mbzuai.ac.ae / aj.das.research@gmail.com)
#  Project: https://github.com/aj-das-research/jhu-bodymaps-warmup
#           The repository carries the full documentation for this file: an
#           illustrated README (animated raw-vs-refined comparisons, results
#           tables, metric plots), two method studies with the measured
#           failures of every rejected algorithm (documentation_v8/,
#           documentation_v9/), per-case machine-readable QA reports
#           (reports/), full-resolution diagnostic tooling and renders
#           (scripts/, visualizations/), and a synthetic ground-truth unit
#           test (tests/test_arch_phantom.py). Every number claimed in the
#           docstrings below is backed by a QA file in that repository.
#
#  Refines SuPreM / TotalSegmentator-style vertebra predictions (C1..L5) on the
#  AbdomenAtlasDemo scans. Design principle: fix MODEL errors, never "fix"
#  PATIENT reality - the CT image is the only editor; anatomical priors only
#  flag suspects and freeze what is already consistent.
#
#  Pipeline (all actions QA-logged per case):
#
#    Stage 1  evidence triage of disconnected components, per vertebra:
#               speck (<30 mm3)            -> removed
#               soft tissue (bone frac<=.5)-> removed (hallucination)
#               bone near main (<=5 mm)    -> re-attached through a CT-bone
#                                             corridor at the geometric neck of
#                                             the gap (unclaimed voxels only),
#                                             else kept + POSSIBLE_FRAGMENT
#               bone far from main         -> removed to the unclaimed pool,
#                                             logged REASSIGN_CANDIDATE
#    Stage 2a island guard: a component enclosed >=80% by one other label and
#             small relative to it is reassigned to that label (vertebrae do
#             not interpenetrate).
#    Stage 2b disc-aware boundary re-arbitration of SUSPECT bands only:
#             - suspicion: audit flags (FRAGMENTED / SIZE / ORDER), inflated or
#               starved size vs neighbors, robust log-volume trend residual,
#               or >=20% of the raw label lost to the pool; contiguous closure
#               forms the band, everything else is frozen.
#             - the CT's own density segments the column: mean HU sampled on
#               disks PERPENDICULAR to the body centerline dips at every
#               intervertebral disc. A spacing-regularized DP picks exactly the
#               expected number of minima between the frozen anchor levels
#               (vertebral heights vary smoothly - a periodicity prior), so the
#               image, not the labels, decides where bodies begin and end.
#             - each level seeds inside its disc-bounded segment; a uniform-
#               speed geodesic competition (synchronized layer growth through
#               the thresholded bone domain) floods the band; fronts meet at
#               low-HU clefts, which is where anatomy separates. Fragments and
#               pooled mass are reclaimed by whichever level reaches them
#               through bone first.
#             - cut surfaces tilt with the column: segment membership uses the
#               first-order arc-length projection onto the centerline tangent,
#               clipped to +/-6 mm so it stays monotone off-axis.
#             - arch phase: posterior elements are rebuilt from PEDICLE ROOTS.
#               Every posterior element attaches to its own level through the
#               two pedicles - the only bone corridors crossing the body/arch
#               frontier - so per level the roots are the arch bone both
#               bone-geodesically and Euclidean-within ~3 mm of that level's
#               phase-1 body, and a waist-severed two-tier race regrows the
#               arch from these roots: identity is decided on the supra-neck
#               core (facet bridges are thinner than any true posterior
#               element, so severing sub-1.5 mm bone means no level can leak
#               across a joint), then the shell rejoins locally and opposing
#               fronts meet mid-bridge at the facet waists.
#             - guards: C1/C2 skipped (no disc plane at the atlas/axis), band
#               skipped when the expected disc count cannot be resolved, and
#               the whole band reverts unless the audit strictly improves.
#    Stage 2c interface polish: for every adjacent pair whose interface is
#             guillotine-flat (planarity RMS below trigger - anatomy
#             interdigitates, so near-planar means geometric arbitration, not
#             joint anatomy), the boundary is re-solved ONLY inside a small
#             collar: seeds just outside the collar, HU-valley watershed so
#             the line settles across the darkest surface of the cleft (each
#             endplate stays with its own vertebra); accepted per pair only if
#             planarity strictly improves with bounded swap and no new
#             fragments, else reverted.
#    Stage 2d pool reclamation (runs LAST) - envelope rule: the raw prediction
#             is the outer boundary; real BONE inside it gets its class fixed,
#             it is never deleted. Every dropped raw-labeled bone component is
#             re-owned by the AXIAL-RING vote (in the axial projection a
#             vertebra is one closed ring; a true process fragment is
#             2D-bone-connected to its own ring in its own slices, while ribs
#             cross slices as disconnected islands), cross-checked by 3D
#             bone-geodesic linkage, then re-attached through a CT-bone
#             corridor. Fragments linked to no vertebra through bone stay out,
#             flagged (they are not vertebra by the label definition).
#    Stage 2e multiview waist-severed recoloring (after 2d): a plate-on-plate
#             contact (spinous blades at the interspinous space) is not a thin
#             waist in 3D, but in the 2D cross-section of the view containing
#             the structure's long axis it is a thin LINE. Each voxel is
#             judged in all three orthogonal views: eroded 2D bone cores are
#             identified by the level ANCHOR they contain (mass within ~28 mm
#             of a body through its own label - tips can never anchor their
#             captor), single-anchor components vote their level in-plane,
#             ambiguous components abstain as blockers, and only unanimous
#             cross-view votes recolor a voxel. Bounded per pair, loser must
#             not fragment, stage reverts if the audit degrades.
#    Stage 2f core-integrity surgery (after 2e): a mixed in-plane supra-neck
#             piece is the defect by definition (one rigid cross-section, two
#             labels): unify to majority when the internal boundary crosses
#             the piece's thick interior and the minority is small, else
#             relocate the boundary to the in-plane thickness valley. Gated on
#             the mixed-piece meter, audit, bounded per-level shifts.
#    Stage 2g imbrication repair (last relabeling stage): thoracolumbar
#             spinous blades IMBRICATE - root inside its own disc band, blade
#             drooping caudally past the next level - so every distance race
#             hands the drooping half one level down, and on DISH the ossified
#             interspinous sheet defeats every thickness signal too. The one
#             surviving invariant: NOTHING in the midline posterior corridor
#             grows toward the head. The corridor is re-derived in a single
#             top-to-bottom consecutive-slice sweep: band-consistent ring-strip
#             labels seed, the z+1 assignment flows down wherever bone
#             continues, a uniform 2D watershed extends in-plane - so a label
#             can only enter at its junction and flow caudally to the tip.
#             Gated on the upward-violation meter (label mass above its own
#             band top: anatomically impossible), mixed-piece meter, audit,
#             bounded per-level shifts.
#    Stage 3  interface regularization (iterated 26-neighborhood majority vote
#             on label-label interfaces), orphan-component absorption, enclosed-
#             hole + directional pit filling, and bounded volume-preserving smoothing
#             (Gaussian-smoothed SDT, volume-preserving iso-level by bisection,
#             changes confined to +/-1.5 mm, additions bone-gated; a label
#             reverts if Dice vs its pre-smooth mask drops below 0.97 or its
#             component count increases).
#    Stage 4  audit of the result (fragmentation, size-smoothness, ordering)
#             written to the QA report; flags never edit voxels.
#
#  Verified on the two AbdomenAtlasDemo cases (identical parameters):
#    BDMAP_00000006 (clean): 10 fragmented + 1 ORDER -> all clean; C1 caudal
#      articular tips on CT-certified bone are preserved (the delete-only
#      baseline removes them).
#    BDMAP_00000031 (sick):  21 fragmented, mass misassignment T8..L2 with
#      L1 at 23.3 cm3 vs ~60 cm3 neighbors -> all clean; L1 restored to
#      ~65 cm3 by re-arbitration at the detected disc planes; the ShapeKit
#      baseline instead deepens the error (L1 11.6 cm3, SIZE 0.20). The
#      imbricated spinous chain (raw painted every T7..L1 blade with mixed /
#      one-off labels) reads clean after stage 2g: upward-violation
#      3.15 -> 0.21 cm3, every blade root-band consistent.
#
#  Dependencies: numpy, scipy, nibabel, scikit-image, cc3d (connected-components-3d).
#
# ==========================================================================
# SuPreM combined_labels ids: 1=L5 ... 24=C1 (bottom-up)
NAMES_BOTTOM_UP = (["L5", "L4", "L3", "L2", "L1"]
                   + [f"T{i}" for i in range(12, 0, -1)]
                   + [f"C{i}" for i in range(7, 0, -1)])
ID_TO_NAME = {i + 1: n for i, n in enumerate(NAMES_BOTTOM_UP)}
NAME_TO_ID = {n: i for i, n in ID_TO_NAME.items()}
TOP_DOWN = list(reversed(NAMES_BOTTOM_UP))
STRUCT6 = ndimage.generate_binary_structure(3, 1)
STRUCT26 = np.ones((3, 3, 3), dtype=bool)

P = {
    # evidence gates
    "bone_hu": 150.0, "speck_mm3": 30.0, "soft_bone_fraction": 0.50,
    "near_mm": 5.0, "corridor_pad_mm": 6.0, "bridge_cap_mm3": 500.0,
    # islands
    "enclosure": 0.80, "max_rel_to_host": 0.5, "min_labeled_shell": 0.5,
    # suspicion / bands
    "suspect_size_lo": 0.60, "suspect_size_hi": 1.80, "suspect_logres": 0.30,
    "suspect_pool_frac": 0.20, "band_gap_close": 3,
    # disc profile / DP
    "profile_sigma_mm": 2.5, "profile_radius_mm": 10.0, "min_body_height_mm": 12.0,
    "max_body_height_mm": 45.0, "dp_beta": 2.0, "pin_lo_mm": 8.0, "pin_hi_mm": 35.0,
    # arbitration domain
    "seed_margin_mm": 3.0, "seed_radius_mm": 14.0, "halo_mm": 2.5, "band_pad_mm": 4.0,
    "arch_root_mm": 3.0, "arch_neck_mm": 1.5, "arch_cost": "core",
    "arch_overhang_mm": 16.0,
    # imbrication repair (stage 2g)
    "imb_half_mm": 14.0, "imb_strip_mm": 6.0, "imb_canal_mm": 12.0,
    "imb_max_shift_cm3": 6.0,
    # interface polish (stage 2c)
    "polish_trigger_mm": 3.5, "polish_min_contact_mm2": 80.0, "polish_collar_mm": 6.0,
    "polish_seed_shell_mm": 2.5, "polish_max_shift_frac": 0.15,
    "polish_min_gain_mm": 0.3, "polish_priority": "hu",
    # pool reclamation (stage 2d, runs last)
    "reclaim_neck_mm": 10.0, "reclaim_link_mm": 10.0,
    # multiview recoloring (stage 2e)
    "mv_neck_mm": 1.2, "mv_anchor_mm": 40.0, "mv_max_shift_frac": 0.25,
    "mv_elong": 2.5,
    # skeleton-component relabeling (stage 2f)
    "skel_cut_mm": 1.8,
    # smoothing
    "sigma_mm": 1.2, "max_dev_mm": 1.5, "vol_tol": 0.01, "min_dice": 0.97,
    "smooth_pad_mm": 5.0,
    # io
    "crop_margin_mm": 25.0,
}


# ------------------------------------------------------------------ utils --
def _bbox_pad(sl, pad_vox, shape):
    """Takes: sl (tuple of slices from find_objects), pad_vox (per-axis voxel
        padding), shape (array shape).
    Does: grows a bounding box by the padding, clamped to the array bounds.
    Returns: the padded tuple of slices.
    """
    return tuple(slice(max(s.start - int(p), 0), min(s.stop + int(p), n))
                 for s, p, n in zip(sl, pad_vox, shape))


def _components(mask):
    """Takes: mask (3D bool).
    Does: 26-connected component labeling (cc3d).
    Returns: (labels volume, per-component voxel counts; counts[0] is zeroed).
    """
    cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    counts = np.bincount(cc.ravel())
    counts[0] = 0
    return cc, counts


def pro_audit(seg, affine, vox_mm3):
    """Repo-convention audit: FRAGMENTED / SIZE(<0.6) / ORDER / EMPTY flags.

    Takes: seg (uint8 labels), affine (voxel-to-world, orients the S-I axis),
        vox_mm3.
    Does: the independent structural audit: per level it measures components,
        volume, size ratio against neighbors, and centroid ordering, and raises
        FRAGMENTED / EMPTY / SIZE / ORDER flags. Flags never edit voxels; every
        stage gate consumes this.
    Returns: (per-level row dicts, summary dict of flag counts).
    """
    rows = []
    objs = ndimage.find_objects(seg)
    for name in TOP_DOWN:
        lid = NAME_TO_ID[name]
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


# ---------------------------------------------------------------- stage 1 --
def stage1_triage(seg, ct, zooms, vox_mm3, records):
    """Takes: seg (raw uint8 labels), ct (HU volume, same grid), zooms (mm),
        vox_mm3, records (QA list, appended per decision).
    Does: evidence triage of every disconnected component: sub-speck and
        mostly-soft-tissue components are removed as noise, bone near its main
        body re-attaches through a CT-certified corridor, and far bone is moved
        to the unclaimed pool for stage 2d instead of being deleted.
    Returns: the triaged label volume.
    """
    out = seg.copy()
    bone = ct >= P["bone_hu"]
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        main = cc == main_id
        union_sl = ndimage.find_objects((cc > 0).astype(np.uint8))[0]
        pad_vox = np.ceil(P["near_mm"] / np.asarray(zooms)) + 2
        union_sl = _bbox_pad(union_sl, pad_vox, seg.shape)
        dist_main = ndimage.distance_transform_edt(~main[union_sl], sampling=zooms)
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id:
                continue
            comp = cc == comp_id
            vol = float(counts[comp_id]) * vox_mm3
            hu = ct[comp]
            rec = {"stage": 1, "label": ID_TO_NAME[lid], "vol_mm3": round(vol, 1),
                   "hu_mean": round(float(hu.mean()), 1),
                   "bone_fraction": round(float((hu >= P["bone_hu"]).mean()), 3)}
            if vol < P["speck_mm3"]:
                out[comp] = 0
                rec["action"] = "REMOVED_SPECK"
            elif rec["bone_fraction"] <= P["soft_bone_fraction"]:
                out[comp] = 0
                rec["action"] = "REMOVED_SOFT"
            else:
                d = float(dist_main[comp[union_sl]].min())
                rec["dist_to_main_mm"] = round(d, 2)
                if d <= P["near_mm"]:
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


def _bridge(seg, bone, main, comp, lid, zooms, neck_mm=None):
    """Takes: seg (labels, painted in place), bone (CT bone mask), main / comp
        (bool masks of the label's main body and the candidate component), lid
        (label id), zooms, neck_mm (optional corridor width cap).
    Does: searches a padded box for a CT-bone corridor connecting comp to main
        and, when one exists within the volume cap, paints the corridor's
        unclaimed bone with lid, so re-attachment adds only image-certified
        bone.
    Returns: the painted corridor mask, or None when no admissible corridor
        exists.
    """
    neck_mm = P["near_mm"] if neck_mm is None else neck_mm
    nz_c = ndimage.find_objects(comp.astype(np.uint8))[0]
    nz_m = ndimage.find_objects(main.astype(np.uint8))[0]
    joint = tuple(slice(min(a.start, b.start), max(a.stop, b.stop))
                  for a, b in zip(nz_c, nz_m))
    pad_vox = np.ceil(P["corridor_pad_mm"] / np.asarray(zooms)) + 1
    joint = _bbox_pad(joint, pad_vox, seg.shape)
    seg_j, bone_j = seg[joint], bone[joint]
    comp_j, main_j = comp[joint], main[joint]
    d_comp = ndimage.distance_transform_edt(~comp_j, sampling=zooms)
    d_main = ndimage.distance_transform_edt(~main_j, sampling=zooms)
    neck = (d_comp <= neck_mm) & (d_main <= neck_mm)
    domain = (bone_j & ((seg_j == 0) | (seg_j == lid)) & neck) | comp_j | main_j
    reach = ndimage.binary_propagation(main_j, mask=domain, structure=STRUCT26)
    if not (reach & comp_j).any():
        return None
    fill = reach & (seg_j == 0) & bone_j & neck
    if float(fill.sum()) * abs(np.prod(zooms)) > P["bridge_cap_mm3"]:
        return None
    corridor = np.zeros_like(comp)
    corridor[joint] = fill
    return corridor


# --------------------------------------------------------------- stage 2a --
def stage2a_islands(seg, vox_mm3, records):
    """Takes: seg, vox_mm3, records (QA list).
    Does: island guard: a component enclosed at least 80 percent by one other
        label and small relative to that host is absorbed by it, since
        vertebrae do not interpenetrate.
    Returns: the label volume with enclosed islands absorbed.
    """
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
            shell = ndimage.binary_dilation(comp_l, structure=STRUCT26) & ~comp_l
            neigh = out[sl][shell]
            labeled = neigh[neigh != 0]
            if neigh.size == 0 or labeled.size / neigh.size < P["min_labeled_shell"]:
                continue
            ids, cnt = np.unique(labeled, return_counts=True)
            host = int(ids[cnt.argmax()])
            frac = float(cnt.max()) / labeled.size
            rec = {"stage": "2a", "label": ID_TO_NAME[lid], "host": ID_TO_NAME[host],
                   "vol_mm3": round(float(counts[comp_id]) * vox_mm3, 1),
                   "enclosure": round(frac, 3)}
            if (host != lid and frac >= P["enclosure"]
                    and counts[comp_id] <= P["max_rel_to_host"] * vol.get(host, 0)):
                out[comp] = host
                rec["action"] = "ISLAND_REASSIGNED"
            else:
                rec["action"] = "ISLAND_FLAG_ONLY"
            records.append(rec)
    return out


# --------------------------------------------------------------- stage 2b --
def find_suspects(seg, raw, affine, vox_mm3):
    """Takes: seg, raw (original prediction, for the unclaimed pool), affine,
        vox_mm3.
    Does: runs the audit, marks suspect levels (size, fragmentation, or
        ordering anomalies, or a large nearby pool), and closes small gaps so
        contiguous suspects and their audit-clean neighbors form re-arbitration
        bands.
    Returns: (list of level-name bands, suspect names, top-down list of present
        levels).
    """
    rows, _ = pro_audit(seg, affine, vox_mm3)
    info = {r["name"].replace("vertebrae_", ""): r for r in rows if r["voxels"] > 0}
    order = [n for n in TOP_DOWN if n in info]
    suspects = set()
    logv = {n: np.log(max(info[n]["voxels"], 1)) for n in order}
    for i, n in enumerate(order):
        w = [logv[order[j]] for j in range(max(0, i - 2), min(len(order), i + 3)) if j != i]
        if w and abs(logv[n] - float(np.median(w))) > P["suspect_logres"]:
            suspects.add(n)
    for n, r in info.items():
        ratio = r.get("size_ratio_vs_neighbors")
        if ratio is not None and (ratio < P["suspect_size_lo"] or ratio > P["suspect_size_hi"]):
            suspects.add(n)
        if r["components"] > 1 or "ORDER" in r["flags"]:
            suspects.add(n)
    for lid in np.unique(raw):
        if lid == 0:
            continue
        raw_n = int((raw == lid).sum())
        pool_n = int(((raw == lid) & (seg == 0)).sum())
        if raw_n and pool_n / raw_n >= P["suspect_pool_frac"]:
            suspects.add(ID_TO_NAME[int(lid)])
    idx = {n: i for i, n in enumerate(order)}
    s_idx = sorted(idx[n] for n in suspects if n in idx)
    bands = []
    for i in s_idx:
        if bands and i - bands[-1][-1] <= P["band_gap_close"] + 1:
            bands[-1] = list(range(bands[-1][0], i + 1))
        else:
            bands.append([i])
    return [[order[j] for j in range(b[0], b[-1] + 1)] for b in bands], sorted(suspects), order


def column_profile(seg, raw, ct, zooms):
    """Body centerline (largest in-plane bone component) + perpendicular
    mean-HU profile: bodies read high, discs dip, even under kyphosis.

    Takes: seg, raw, ct, zooms.
    Does: traces the per-slice column centroid and samples mean HU inside
        centerline disks, producing the density profile whose minima are
        intervertebral discs.
    Returns: (cx, cy per-slice centroid arrays, smoothed HU profile; the
        profile is zeros when the column is too short).
    """
    bone = ct >= P["bone_hu"]
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
    r = P["profile_radius_mm"]
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
                                      sigma=max(P["profile_sigma_mm"] / zooms[2], 0.8))
    return cxs, cys, rho_s


def disc_minima(rho_s, z_lo, z_hi, n_cuts, zooms, pin_lo, pin_hi):
    """Exactly n_cuts minima chosen by spacing-regularized DP with anchor pins.

    Takes: rho_s (HU profile), z_lo / z_hi (search window), n_cuts (expected
        disc count), zooms, pin_lo / pin_hi (whether the window ends anchor on
        clean neighbors).
    Does: selects exactly n_cuts profile minima by spacing-regularized dynamic
        programming under body-height priors, so shallow or fused discs cannot
        collapse the count.
    Returns: sorted z indices of the chosen cuts, or None when the expected
        count cannot be resolved (the band is then skipped, flagged).
    """
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
    min_h, max_h = P["min_body_height_mm"], P["max_body_height_mm"]
    span = (z_hi - z_lo) * dz
    NEG = -1e18
    first_ok = (z >= P["pin_lo_mm"]) & (z <= P["pin_hi_mm"]) if pin_lo else np.ones(m, bool)
    last_ok = ((z >= span - P["pin_hi_mm"]) & (z <= span - P["pin_lo_mm"])
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
                        s = dp[k - 1, i, i2] - P["dp_beta"] * np.log(h / h_prev) ** 2
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


def arch_repartition(labels_ws, body_zone, A, band_ids, zooms, vox_mm3):
    """Rebuild the posterior arch from PEDICLE ROOTS (pure function of the
    phase-1 flood; bodies are never edited).

    Anatomy: every posterior element (pedicles, laminae, spinous, transverse
    and articular processes) attaches to its own vertebra exclusively through
    the two pedicles - the only bone corridors crossing the body/arch
    frontier - while adjacent levels touch posteriorly only at thin
    facet-joint bridges. Two consequences drive the algorithm:

      1. SEEDS: per level, the pedicle roots are the arch-domain bone that is
         BONE-GEODESICALLY within arch_root_mm of that level's phase-1 body
         (masked dilation through own body + arch only, so Euclidean
         proximity across the foramen or a collapsed disc cannot mint a
         root) AND Euclidean-within the same radius (so voxel-step overshoot
         on anisotropic grids cannot either). Voxels reached by two bodies
         are contested and stay unseeded.
      2. COMPETITION (arch_cost="core"): a waist-severed two-tier race.
         Tier 1 decides IDENTITY on the supra-neck core (arch voxels with
         local thickness >= arch_neck_mm, plus the roots): facet bridges are
         thinner than any true posterior element, so severing them makes
         each level's core reachable only from its own pedicle roots and the
         identity race cannot leak across a joint. Tier 2 rejoins the shell
         (sub-neck voxels, including the bridges themselves) by a local
         uniform BFS from the tier-1 labels, so opposing shells meet
         mid-bridge. Phantom-measured ablations: "edt" (-EDT priority) has
         no post-neck resistance and splits hanging processes by depth
         order; "uniform" lets a root near a joint out-run a far pedicle.

    Takes: labels_ws (band race result), body_zone (bool body domain), A (bool
        arch domain), band_ids, zooms, vox_mm3.
    Does: rebuilds posterior-arch ownership from pedicle-root seeds with the
        waist-severed two-tier race (arch_cost modes: core = release, hier /
        edt / uniform = measured ablations); bodies are never edited.
    Returns: (labels with the arch re-owned, QA record dict).
    """
    body_lab = np.where(body_zone, labels_ws, 0).astype(np.int32)
    roots = np.zeros(A.shape, dtype=np.int32)
    contested = np.zeros(A.shape, dtype=bool)
    pad_vox = np.ceil((P["arch_root_mm"] + 2.0) / np.asarray(zooms)).astype(int) + 1
    root_it = max(int(np.ceil(P["arch_root_mm"] / min(zooms))), 1)
    rootless = []
    for lid in band_ids:
        body = body_lab == lid
        if not body.any():
            rootless.append(ID_TO_NAME[lid])
            continue
        sl = _bbox_pad(ndimage.find_objects(body.astype(np.uint8))[0],
                       pad_vox, A.shape)
        reach = ndimage.binary_dilation(body[sl], structure=STRUCT6,
                                        iterations=root_it,
                                        mask=A[sl] | body[sl])
        d_eu = ndimage.distance_transform_edt(~body[sl], sampling=zooms)
        r = reach & A[sl] & (d_eu <= P["arch_root_mm"] + max(zooms))
        if not r.any():
            rootless.append(ID_TO_NAME[lid])
            continue
        sub = roots[sl]
        contested[sl] |= (sub > 0) & (sub != lid) & r
        sub[r & (sub == 0)] = lid
    roots[contested] = 0
    rec = {"arch_mode": f"pedicle_roots_{P['arch_cost']}",
           "arch_roots_cm3": {ID_TO_NAME[l]: round(float((roots == l).sum())
                                                   * vox_mm3 / 1000.0, 2)
                              for l in band_ids if (roots == l).any()},
           "arch_contested_mm3": round(float(contested.sum()) * vox_mm3, 1)}
    if rootless:
        rec["arch_rootless"] = rootless
    if not (roots > 0).any():
        rec["arch_skipped"] = "no pedicle roots found"
        return labels_ws, rec
    if P["arch_cost"] == "hier":
        # MULTI-SCALE PEEL RACE (ablation - MEASURED WORSE on-case, kept for
        # the record). Idea: no single neck scale exists on DISH (facet
        # necks 1-2 mm, fusions 2-5 mm, blades 4-8 mm), so claim thick bone
        # first and let fronts cross thinner bone only later; thickness
        # measured in-plane per parasagittal slice. MEASURED OUTCOME
        # (BDMAP_00000031, T5..L2 band): upward-violation 0.33 -> 7.68 cm3
        # vs 3.16 for "core" - REVERTED by the imbrication gate. WHY IT
        # FAILS: the ossified interspinous bridge is a MIDLINE SAGITTAL
        # SHEET, continuous with both blades; its sagittal in-plane
        # cross-section is a broad blob, so in-plane thickness is LARGE at
        # exactly the bridge to be severed - the peel welds blade to blade
        # (and 3D EDT agrees). No geometric thickness signal separates a
        # blade from a fusion sheet in its own plane; identity there must
        # come from CAUDAL MONOTONICITY instead (stage 2g).
        bb = _bbox_pad(ndimage.find_objects(A.astype(np.uint8))[0],
                       np.array([2, 2, 2]), A.shape)
        Ab = A[bb]
        solid = Ab | body_zone[bb]      # anatomical thickness, not domain-cut
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
    elif P["arch_cost"] == "core":
        edtA = ndimage.distance_transform_edt(A, sampling=zooms).astype(np.float32)
        core = A & ((edtA >= P["arch_neck_mm"]) | (roots > 0))
        lab1 = watershed(np.zeros(A.shape, dtype=np.uint8), markers=roots,
                         mask=core)
        labels_arch = watershed(np.zeros(A.shape, dtype=np.uint8), markers=lab1,
                                mask=A)
    elif P["arch_cost"] == "edt":
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


def arbitrate_band(seg, raw, ct, band, neighbors, cxs, cys, rho_s, zooms,
                   vox_mm3, qa):
    """Re-arbitrate one contiguous suspect band (v2 geometry).

    v2 upgrades over the flat-z version:
      - ARC-LENGTH OBLIQUE CUTS: segment membership uses s = first-order
        projection onto the column centerline tangent, so cut planes tilt with
        kyphosis instead of clipping tilted bodies with axial guillotines.
      - AGREEMENT-EXTENDED SEEDS: body-core seeds grow by geodesic propagation
        through voxels whose CURRENT label already matches the segment (and
        whose s lies in-segment), giving each level arch seeds wherever the
        model already agreed; only genuinely contested bone is left to race.
      - EDT-PRIORITY WATERSHED: flood priority = -distance-to-background, the
        classic touching-object splitter: thick masses (bodies, laminae) are
        claimed first and fronts meet at thin waists, which anatomically are
        the facet joints and the pars, not arbitrary mid-arch loci.

    Takes: seg, raw, ct, band (level names), neighbors (clean levels below /
        above), cxs / cys / rho_s (column geometry from column_profile), zooms,
        vox_mm3, qa.
    Does: re-arbitrates one suspect band end to end: disc cuts by DP, oblique
        arc-length segment membership, agreement-extended body seeds, uniform
        geodesic race, then the pedicle-root arch rebuild.
    Returns: the re-arbitrated label volume, or None when disc counting is
        unresolved.
    """
    bone = ct >= P["bone_hu"]
    band_ids = [NAME_TO_ID[n] for n in band]
    band_mask = np.isin(seg, band_ids)
    pool = (np.isin(raw, band_ids)) & (seg == 0) & bone
    zsel = np.nonzero((band_mask | pool).any(axis=(0, 1)))[0]
    pad = int(P["band_pad_mm"] / zooms[2]) + 1
    z0 = max(int(zsel[0]) - pad, 0)
    z1 = min(int(zsel[-1]) + pad + 1, seg.shape[2])
    rec = {"levels": band, "z_range_crop": [int(z0), int(z1)]}
    qa["bands"].append(rec)

    def centroid_z(name):
        zs = np.nonzero((seg == NAME_TO_ID[name]).any(axis=(0, 1)))[0]
        return float(zs.mean()) if zs.size else None

    ups = sorted((NAME_TO_ID[n], centroid_z(n)) for n in band if centroid_z(n) is not None)
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

    # ---- arc-length field ----------------------------------------------
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
        # oblique correction, clipped to +/-6 mm: full first-order tilt at the
        # body radius (where cuts matter), saturated at rib/process radius
        # where the per-z linearization folds (validated visually)
        corr = np.clip(dx * np.float32(T[z, 0]) + dy * np.float32(T[z, 1]), -6.0, 6.0)
        s_f[:, :, z - z0e] = np.float32(s_axis[z]) + corr

    ids_by_z = sorted(band_ids) if z_up else sorted(band_ids, reverse=True)
    seg_ranges = list(zip(s_cuts[:-1], s_cuts[1:]))
    assign = dict(zip(ids_by_z, seg_ranges))
    s_lo_all, s_hi_all = s_cuts[0], s_cuts[-1]

    # ---- seeds: body core + agreement extension ------------------------
    markers = np.zeros(seg_s.shape, dtype=np.int32)
    r2 = P["seed_radius_mm"] ** 2
    margin = P["seed_margin_mm"]
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
            er = ndimage.binary_erosion(seg_s == lid, structure=STRUCT6, iterations=2)
            seeds = er if er.any() else (seg_s == lid)
            qa["flags"].append(f"SEED_FALLBACK_{ID_TO_NAME[lid]}")
        markers[seeds] = lid
    rec["seed_cm3"] = {ID_TO_NAME[l]: round(float((markers == l).sum()) * vox_mm3 / 1000.0, 1)
                       for l in assign}

    # ---- domain + EDT-priority competition ------------------------------
    st = ndimage.generate_binary_structure(3, 1)
    halo_it = int(max(np.round(P["halo_mm"] / np.asarray(zooms))))
    grown = ndimage.binary_dilation(band_s | pool_s, structure=st, iterations=max(halo_it, 1))
    halo = grown & (seg_s == 0) & bone_s & ~pool_s
    # bodies are gated strictly at the outer disc cuts, but the ARCH domain
    # may OVERHANG them: superior/inferior articular processes and steep
    # thoracic spinous processes legitimately interdigitate 8-15 mm past the
    # disc plane, and hard s-gating them amputated real bone (measured as
    # cleared_out_of_band). The pedicle-root race decides who owns the
    # overhang; frozen neighbors' voxels are never in the domain either way.
    rb2 = (P["seed_radius_mm"] + 4.0) ** 2
    body_zone = d2_slab <= rb2
    in_s = (s_f >= s_lo_all) & (s_f <= s_hi_all)
    in_s_arch = ((s_f >= s_lo_all - P["arch_overhang_mm"])
                 & (s_f <= s_hi_all + P["arch_overhang_mm"]))
    D = (band_s | pool_s | halo) & np.where(body_zone, in_s, in_s_arch)
    markers[~D] = 0
    # uniform-speed geodesic competition: distance is the cost. (A depth
    # priority was tried and convicted: laminae form one connected deep river,
    # so the first label entering ran the whole posterior column.)
    labels_ws = watershed(np.zeros(D.shape, dtype=np.uint8), markers=markers, mask=D)

    # ---- phase 2: arch rebuild from pedicle roots ------------------------
    # The flood race misassigns posterior elements (spinous/laminae) because
    # facet joints are bone-continuous. Anatomy fix: regrow the arch from
    # each level's PEDICLE ROOTS - the only bone corridors crossing the
    # body/arch frontier - with a waist-severed identity race that cannot
    # leak across thin facet necks (see arch_repartition).
    A = D & ~body_zone
    if A.any():
        labels_ws, arch_rec = arch_repartition(labels_ws, body_zone & D, A,
                                               list(assign), zooms, vox_mm3)
        rec.update(arch_rec)
        for nm in arch_rec.get("arch_rootless", []):
            qa["flags"].append(f"ARCH_NO_ROOT_{nm}")

    out = seg.copy()
    out[band_mask] = 0  # includes out-of-band overreach, which is cleared
    out_s = out[:, :, slab]
    sel = labels_ws > 0
    out_s[sel] = labels_ws[sel].astype(seg.dtype)
    rec["unreached_pool_mm3"] = round(float((D & ~sel & (seg_s == 0)).sum()) * vox_mm3, 1)
    cleared = band_mask & (out == 0)
    rec["cleared_out_of_band_mm3"] = round(float(cleared.sum()) * vox_mm3, 1)
    rec["cleared_arch_mm3"] = round(float((cleared[:, :, slab] & ~body_zone).sum())
                                    * vox_mm3, 1)
    return out



def _spinous_frame(seg, ct, zooms):
    """Shared geometry for the spinous meter and repair (stage 2g): column
    centerline, posterior side sign, per-z corridor front = body posterior
    edge + canal depth (DISH cannot fuse the body edge away), and per-level
    body z-bands from the 18 mm centerline cylinder - bodies are
    disc-cut-validated, the trusted identity source.

    Takes: seg, ct, zooms.
    Does: computes the shared spinous-corridor geometry used by the meter and
        stage 2g.
    Returns: dict with centerline (cx, cy), posterior sign, per-slice corridor
        front y0, and per-level body bands; None when the column is too short.
    """
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
    off = P["imb_canal_mm"] / zooms[1]
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


def _imbrication_cm3(seg, ct, zooms, vox_mm3, lids=None, tol_mm=5.0,
                     frame=None):
    """Spinous UPWARD-VIOLATION meter (one-sided tripwire for the
    imbrication error class). In the MIDLINE posterior corridor (|x-cx| <=
    7 mm, posterior of body edge + canal) the only structures are spinous
    blades and interspinous bone, and blades angle strictly CAUDALLY: a
    blade droops BELOW its vertebra, never reaches above it. Corridor volume
    carrying a label whose own body band lies below (z > band_top + tol) is
    therefore anatomically impossible - it is a blade wearing the label of
    the level below (measured failure mode: the T9..L2 one-down chain).
    The drooped part of a stolen blade z-overlaps the thief's own band and
    is invisible to any z-test - ONLY root-attachment can decide it - so
    this meter is a tripwire, not a complete count.
    Returns (total_cm3, per_level_dict). Pass frame= to reuse (and to
    measure before/after with IDENTICAL geometry).

    Takes: seg, ct, zooms, vox_mm3, lids (levels to score, default all
        present), tol_mm, frame (optional precomputed _spinous_frame so before
        / after are measured in identical geometry).
    Does: scores the spinous upward-violation meter defined above.
    Returns: (total violating volume in cm3, per-level dict of offenders).
    """
    fr = frame if frame is not None else _spinous_frame(seg, ct, zooms)
    if fr is None:
        return 0.0, {}
    if lids is None:
        lids = [int(v) for v in np.unique(seg) if v != 0]
    bone = ct >= P["bone_hu"]
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
    per = {ID_TO_NAME[l]: round(n * vox_mm3 / 1e3, 2)
           for l, n in upv.items() if n * vox_mm3 > 20.0}
    total = round(float(sum(n for n in upv.values())) * vox_mm3 / 1e3, 2)
    return total, per


def stage2g_imbrication(seg, ct, affine, zooms, vox_mm3, qa):
    """CAUDAL-FLOW spinous repair: blade identity = ROOT ATTACHMENT,
    propagated strictly downward through consecutive slices.

    THE DEFECT CLASS (measured in the v8 full-resolution review):
    thoracolumbar spinous blades IMBRICATE - the root joins the ring inside
    its own disc band and the blade droops caudally past the next level, so
    the tip lies Euclidean-nearest the vertebra BELOW. Every distance race
    therefore hands the drooping half one level down (the T9..L2 one-down
    chain), and on DISH no thickness signal can veto it: the ossified
    interspinous bridge is a midline sagittal SHEET, bone-continuous with
    both blades and broad in its own cross-section (the "hier" multi-scale
    peel ablation measured upward-violation 0.33 -> 7.68 cm3 - worse than
    the race it tried to fix). One invariant survives all of it: NOTHING in
    the midline posterior corridor grows toward the head.

    Mechanism: the corridor's anterior STRIP (imb_strip_mm at the ring,
    where the junctions live) keeps its labels - junction identity is
    race-derived from pedicle roots and audit-verified. The corridor mass
    posterior of the strip is re-derived in ONE top-to-bottom sweep over
    consecutive axial slices: markers at slice z = strip labels at z, plus
    the z+1 assignment wherever its (x,y) footprint continues into bone at
    z, extended in-plane by a uniform 2D watershed. A label can only ENTER
    the corridor at its junction and FLOW DOWN, so a lower level can never
    claim bone above its own band, while a drooping blade carries its
    root's label to the tip; descending fronts of consecutive levels meet
    inside the interspinous sheet. (This is the consecutive-slice
    "temporal outline propagation" idea, made safe by the anatomical
    direction constraint - undirected chaining was tried and reverted, see
    stage 2f docstring.)

    Recolor-only: the raw envelope is preserved, unreached labeled voxels
    keep their labels, nothing is deleted. Gates: upward-violation must
    improve (or stay ~0 with negligible churn), the mixed-piece meter must
    not increase, the audit must not degrade, per-level net shift <=
    imb_max_shift_cm3 - else full revert.

    Takes: seg, ct, affine, zooms, vox_mm3, qa (record written under key
        'imbrication').
    Does: the caudal-flow spinous repair described above.
    Returns: the repaired label volume, or the input unchanged after a gated
        full revert.
    """
    fr = _spinous_frame(seg, ct, zooms)
    if fr is None:
        qa["imbrication"] = {"skipped": "no frame"}
        return seg
    cx, psign, y0 = fr["cx"], fr["psign"], fr["y0"]
    bone = ct >= P["bone_hu"]
    u0, up0 = _imbrication_cm3(seg, ct, zooms, vox_mm3, frame=fr)
    hw = max(int(round(P["imb_half_mm"] / zooms[0])), 2)
    sw = int(round(P["imb_strip_mm"] / zooms[1]))
    zs = np.nonzero(seg.any(axis=(0, 1)))[0]
    if zs.size == 0:
        qa["imbrication"] = {"skipped": "empty"}
        return seg
    # the slab must FOLLOW THE BLADES, not the body centroid: vertebral
    # rotation (scoliosis) swings the spinous line laterally, and a slab
    # centered on the body clips the blade - the clipped part keeps its
    # stolen label and a fresh mixed piece is minted at the slab edge
    # (measured: badcut 35 -> 36, upv only halved, on the body-centered
    # first attempt)
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
        # a strip voxel SEEDS the flow only when its label is BAND-
        # CONSISTENT (junctions live at their own band by definition);
        # a mislabeled strip patch - the near-root end of a stolen blade -
        # is flow territory and gets re-derived like the rest (keeping it
        # left residual violations in the root belt and minted a mixed
        # piece per blade: badcut 35 -> 36 on the first attempt)
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
    # CONVERGENCE (same treatment as stage 2f): per-slice reassignment
    # leaves ragged seams and strands old-label slivers at the slab and
    # strip boundaries - unconverged, five levels fragmented into dozens
    # of components (measured: T10 -> 44). Volume-preserving interface
    # majority vote, then fuse detached slivers into their surrounding
    # label (nothing deleted), inside the changed-region bbox.
    diff = out != seg
    if diff.any():
        padv = np.ceil(10.0 / np.asarray(zooms)).astype(int)
        bb = _bbox_pad(ndimage.find_objects(diff.astype(np.uint8))[0],
                       padv, seg.shape)
        sub = majority_filter(out[bb].copy(), iters=1)
        sub = pro_absorb_orphans(sub, vox_mm3, max_mm3=1500.0, delete_below=0.0)
        keep = out[bb] > 0
        out[bb] = np.where(keep, np.where(sub > 0, sub, out[bb]), 0)
    u1, up1 = _imbrication_cm3(out, ct, zooms, vox_mm3, frame=fr)
    cx_med = int(round(np.median(cx)))
    mix_before = _badcut_pieces(seg, bone, zooms, cx_med)
    mix_after = _badcut_pieces(out, bone, zooms, cx_med)
    changed = int((out != seg).sum())
    deltas = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        d = (int((out == lid).sum()) - int((seg == lid).sum())) * vox_mm3 / 1e3
        if abs(d) > 0.05:
            deltas[ID_TO_NAME[lid]] = round(d, 2)
    _, s_before = pro_audit(seg, affine, vox_mm3)
    _, s_after = pro_audit(out, affine, vox_mm3)
    bad = lambda s: (s["n_fragmented"] + s["n_empty"] + s["n_size"]
                     + s["n_order"])
    rec = {"changed_cm3": round(changed * vox_mm3 / 1e3, 2),
           "unreached_mm3": round(unreached * vox_mm3, 1),
           "upv_cm3_before": u0, "upv_cm3_after": u1,
           "upv_per_level_before": up0, "upv_per_level_after": up1,
           "badcut_pieces_before": mix_before,
           "badcut_pieces_after": mix_after,
           "volume_deltas_cm3": deltas}
    qa["imbrication"] = rec
    over = [n for n, d in deltas.items()
            if abs(d) > P["imb_max_shift_cm3"]]
    ok_meter = (u1 < u0 - 0.2) or (u0 <= 0.2 and u1 <= u0 + 0.05
                                   and changed * vox_mm3 / 1e3 <= 2.0)
    rec["gate"] = {"ok_meter": bool(ok_meter),
                   "audit_bad_before_after": [bad(s_before), bad(s_after)]}
    if (not ok_meter or mix_after > mix_before
            or bad(s_after) > bad(s_before) or over):
        rec["reverted_all"] = True
        if over:
            rec["over_shift"] = over
        qa["flags"].append("IMBRICATION_REVERTED_ALL")
        return seg
    return out


def stage2b_arbitrate(seg, raw, ct, affine, zooms, vox_mm3, qa):
    """Takes: seg, raw, ct, affine, zooms, vox_mm3, qa.
    Does: finds suspect bands and runs arbitrate_band on each; a band edit is
        kept only when its audit badness strictly improves, and the band's
        imbrication meter is recorded before and after (report-only here; stage
        2g owns the repair).
    Returns: the label volume with all accepted band edits applied.
    """
    bands, suspects, present = find_suspects(seg, raw, affine, vox_mm3)
    qa["suspects"] = suspects
    if not bands:
        return seg
    cxs, cys, rho_s = column_profile(seg, raw, ct, zooms)

    def badness(s, names):
        rows_, _ = pro_audit(s, affine, vox_mm3)
        bad = 0
        for r in rows_:
            nm = r["name"].replace("vertebrae_", "")
            if nm in names:
                bad += sum(1 for f in r["flags"]
                           if f.startswith(("SIZE", "FRAG", "EMPTY", "ORDER")))
                ratio = r.get("size_ratio_vs_neighbors")
                if ratio is not None and ratio > P["suspect_size_hi"]:
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
        band_lids = [NAME_TO_ID[n] for n in band]
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
            # imbrication UPV is recorded but NOT gated here: the band edit
            # owns body identity (its win), while blade imbrication is owned
            # and repaired downstream by stage 2g - vetoing the whole band
            # for a 2-3 cm3 blade steal would throw away 40+ cm3 of body
            # fixes (measured: this exact veto reverted the T5..L2 band).
            out = cand
    return out


# --------------------------------------------------------------- stage 2c --
def _adjacent_interfaces(seg):
    """One sweep over 3 axes -> {(a,b): Nx3 voxel points}, a<b.

    Takes: seg.
    Does: collects the voxel coordinates of every label-label interface by face
        adjacency.
    Returns: dict mapping ordered label pairs (a, b) to N x 3 coordinate
        arrays.
    """
    out = {}
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
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


def _planarity_mm(pts, zooms):
    """Takes: pts (N x 3 interface voxel coordinates), zooms.
    Does: fits a plane by SVD in millimeter space and measures the RMS out-of-
        plane residual, the flatness meter behind stage 2c (real joints
        interdigitate; near-planar means geometric arbitration).
    Returns: RMS distance in mm, or None for degenerate point sets.
    """
    if pts is None or len(pts) < 50:
        return None
    Q = pts * np.asarray(zooms)
    Q = Q - Q.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return float(np.sqrt(((Q @ Vt[2]) ** 2).mean()))


def _pair_pts(seg, a, b):
    """Takes: seg, a, b (label ids).
    Does: collects voxel coordinates on both sides of the a | b interface (face
        adjacency).
    Returns: N x 3 int array, or None when the pair does not touch.
    """
    pts = []
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        m = ((x == a) & (y == b)) | ((x == b) & (y == a))
        if m.any():
            i = np.nonzero(m)
            pp = np.c_[i[0], i[1], i[2]]
            pp[:, ax] += 1
            pts.append(pp)
    return np.vstack(pts) if pts else None


def stage2c_interface_polish(seg, ct, zooms, vox_mm3, qa):
    """Evidence-local re-arbitration of guillotine-flat adjacent interfaces
    (v3-plan P1b). Anatomy interdigitates, so a near-planar label interface
    is a fingerprint of geometric arbitration, not of the joint. For every
    adjacent pair whose interface planarity RMS is below the trigger:
    re-solve ONLY a small collar around the interface - domain is the two
    labels inside the collar, seeds are their voxels just outside, priority
    is the (negative) smoothed HU so the boundary settles across the darkest
    surface of the cleft (cortical endplate - disc - endplate reads
    bright-dark-bright, and the valley belongs to neither side). Each pair
    is accepted only if planarity strictly improves, the swap stays bounded,
    and neither label fragments; otherwise it reverts. Flags never edit.

    Takes: seg, ct, zooms, vox_mm3, qa (per-pair records appended under
        'polish').
    Does: the HU-valley interface re-solve described above.
    Returns: the label volume with accepted per-pair boundary re-solves
        applied.
    """
    out = seg.copy()
    pairs = _adjacent_interfaces(out)
    adj = {tuple(sorted((NAME_TO_ID[TOP_DOWN[k]], NAME_TO_ID[TOP_DOWN[k + 1]])))
           for k in range(len(TOP_DOWN) - 1)}
    for (a, b) in sorted(pairs):
        if (a, b) not in adj:
            continue
        pts = pairs[(a, b)]
        p0 = _planarity_mm(pts, zooms)
        area = len(pts) * (vox_mm3 ** (2 / 3))
        rec = {"pair": f"{ID_TO_NAME[b]}|{ID_TO_NAME[a]}",
               "contact_mm2": round(area, 0),
               "planarity_before": None if p0 is None else round(p0, 2)}
        qa["polish"].append(rec)
        if (p0 is None or area < P["polish_min_contact_mm2"]
                or p0 >= P["polish_trigger_mm"]):
            rec["action"] = "skip"
            continue
        lo = np.maximum(pts.min(0) - 2, 0)
        hi = np.minimum(pts.max(0) + 3, np.asarray(out.shape))
        pad = np.ceil((P["polish_collar_mm"] + P["polish_seed_shell_mm"] + 2)
                      / np.asarray(zooms)).astype(int)
        sl = tuple(slice(max(int(l - p), 0), min(int(h + p), n))
                   for l, h, p, n in zip(lo, hi, pad, out.shape))
        sub = out[sl]
        iface = np.zeros(sub.shape, dtype=bool)
        iface[tuple((pts - [s.start for s in sl]).T)] = True
        d_if = ndimage.distance_transform_edt(~iface, sampling=zooms)
        collar = d_if <= P["polish_collar_mm"]
        shell = (d_if > P["polish_collar_mm"]) & \
                (d_if <= P["polish_collar_mm"] + P["polish_seed_shell_mm"])
        ab = (sub == a) | (sub == b)
        R = collar & ab
        seeds = np.where(shell & ab, sub, 0).astype(np.int32)
        if not (seeds == a).any() or not (seeds == b).any() or not R.any():
            rec["action"] = "skip_no_seeds"
            continue
        if P["polish_priority"] == "hu":
            # boundary settles across the darkest surface of the cleft
            # (joint space / disc): fronts traverse bone freely, the HU
            # valley is claimed last, and each endplate stays with its own
            # vertebra. (|grad HU| puts the line on an endplate ridge
            # instead, which annexes thick discs to one side.)
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
        # (A random-walker fallback - Grady 2006, the v3-plan P1b candidate -
        # was evaluated here and dropped: scikit-image's solver returns
        # degenerate probabilities on these anisotropic thin-cleft ROIs, so
        # it never produced an acceptable candidate. Pairs whose watershed
        # candidate fails the gate simply revert.)
    return out


def _polish_gate(rec, sub, cand, a, b, zooms, tag, p0):
    """Accept iff planarity strictly improves, swap bounded, no new fragments.

    Takes: rec (QA record, mutated), sub / cand (labels in the collar box
        before and after), a, b (the pair), zooms, tag, p0 (planarity before).
    Does: the per-pair acceptance test of stage 2c: planarity must strictly
        improve by polish_min_gain_mm, the swap stays under
        polish_max_shift_frac of the smaller label, and neither label may
        fragment.
    Returns: True when the candidate is accepted.
    """
    m = (cand == a) | (cand == b)
    swapped = int(((cand != sub) & m).sum())
    n_min = min(int((sub == a).sum()), int((sub == b).sum()))
    p1 = _planarity_mm(_pair_pts(cand, a, b), zooms)
    rec[f"planarity_after_{tag}"] = None if p1 is None else round(p1, 2)
    rec[f"swapped_vox_{tag}"] = swapped
    ok = (p1 is not None and swapped > 0
          and swapped <= P["polish_max_shift_frac"] * max(n_min, 1)
          and p1 > p0 + P["polish_min_gain_mm"])
    if ok:
        for lab in (a, b):
            _, c_new = _components(cand == lab)
            _, c_old = _components(sub == lab)
            if int((c_new > 0).sum()) > int((c_old > 0).sum()):
                ok = False
                rec["fragment_guard"] = ID_TO_NAME[lab]
                break
    rec["action"] = tag if ok else f"revert_{tag}"
    return bool(ok)


# --------------------------------------------------------------- stage 2e --
def _centerline_xy(seg):
    """Takes: seg.
    Does: per-slice centroid of all labeled voxels, interpolated across empty
        slices and smoothed: the column centerline used by every sheared-frame
        stage.
    Returns: (cx, cy) float arrays over z, or (None, None) when too few labeled
        slices exist.
    """
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


def _shift2d_stack(vol, sx, sy):
    """Integer per-slice (x, y) shifts with zero fill - exactly invertible
    with (-sx, -sy). Shear-straightens a curved column so fixed-index
    sagittal/coronal planes follow the anatomy along the whole spine.

    Takes: vol, sx / sy (integer per-slice shifts).
    Does: shifts every z slice in-plane with zero fill; integer shifts make the
        shear-straightening exactly invertible with negated shifts.
    Returns: the shifted volume.
    """
    out = np.zeros_like(vol)
    nx, ny, nz = vol.shape
    for z in range(nz):
        dx, dy = int(sx[z]), int(sy[z])
        xs0, xs1 = max(0, -dx), min(nx, nx - dx)
        ys0, ys1 = max(0, -dy), min(ny, ny - dy)
        out[xs0:xs1, ys0:ys1, z] = vol[xs0 + dx:xs1 + dx, ys0 + dy:ys1 + dy, z]
    return out


def _mv_anchors(seg, ct, zooms, cx, cy):
    """Per-level ANCHOR mass: voxels reachable from the level's BODY CORE
    through its own label within mv_anchor_mm. Bodies are the most reliable
    identity post-refinement (disc cuts), and a mislabeled process TIP is
    45 mm+ from the wrong level's body THROUGH that level's label, so tips
    can never anchor their captor.

    Takes: seg, ct, zooms, cx / cy (column centerline).
    Does: builds the per-level ANCHOR mass described above (reach from the body
        core through the level's own label, bounded by mv_anchor_mm).
    Returns: uint8 volume of anchor labels.
    """
    rb = (P["seed_radius_mm"] + 4.0)
    anchors = np.zeros_like(seg)
    it = max(int(np.ceil(P["mv_anchor_mm"] / min(zooms))), 1)
    pad = np.ceil((P["mv_anchor_mm"] + 4.0) / np.asarray(zooms)).astype(int)
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
            core = ndimage.binary_erosion(lab, structure=STRUCT6, iterations=2)
        if not core.any():
            continue
        reach = ndimage.binary_dilation(core, structure=STRUCT6, iterations=it,
                                        mask=lab)
        anchors[sl][reach] = lid
    return anchors


def stage2e_multiview_recolor(seg, ct, affine, zooms, vox_mm3, qa):
    """Multiview waist-severed recoloring (recolor-only; envelope-safe).

    Physics: a plate-on-plate contact (two spinous blades touching flat at
    the interspinous space, a process resting on a lamina) is NOT a thin
    waist in 3D - the EDT of a broad surface contact is large, so the 3D
    race cannot see the joint. But in the 2D CROSS-SECTION of the view
    whose plane contains the structure's long axis, the same contact is a
    thin LINE: one erosion severs it, while the structure's own in-plane
    core stays connected end to end. Hard from the axial view, easy from
    the sagittal - so every voxel is judged in all three orthogonal views:

      per view, per slice: erode the 2D bone cross-section by mv_neck_mm
      (plate kisses vanish); each surviving 2D core component is identified
      by which levels' ANCHOR mass it contains (see _mv_anchors); a
      component with exactly ONE anchor level votes that level onto every
      pixel it reconstructs in-plane (uniform 2D watershed); components
      with zero or multiple anchors become BLOCKERS - they claim their own
      shells and abstain, so a vote can never leak back across a severed
      joint.

    A component may only VOTE when the view sees the structure lengthwise:
    its in-plane PCA elongation must exceed mv_elong (blades vote from the
    sagittal view, transverse wings from the coronal; an axial cross-cut
    of a fused tip is compact and abstains). All slicing runs in a
    SHEAR-STRAIGHTENED frame (integer per-slice centerline shifts, exactly
    invertible): on a scoliotic column a fixed plane drifts off-midline
    and chains everything into one component, while the straightened plane
    follows the anatomy end to end. A voxel is recolored only when its
    votes are unanimous for one level different from its current label
    (any conflicting view vetoes). Gates: per adjacent pair the swap is
    bounded by mv_max_shift_frac of the smaller label, the losing label
    must not fragment, and the stage reverts globally unless the audit
    stays at least as clean.

    Takes: seg, ct, affine, zooms, vox_mm3, qa (record under 'multiview').
    Does: the unanimous three-view waist-severed recoloring described above.
    Returns: the recolored volume, or the input after a gated full revert.
    """
    cx, cy = _centerline_xy(seg)
    if cx is None:
        qa["multiview"] = {"flipped_mm3": 0.0}
        return seg
    bone = ct >= P["bone_hu"]
    anchors = _mv_anchors(seg, ct, zooms, cx, cy)
    sx = np.round(cx - np.median(cx)).astype(int)
    sy = np.round(cy - np.median(cy)).astype(int)
    bone_s = _shift2d_stack(bone.astype(np.uint8), sx, sy).astype(bool)
    anch_s = _shift2d_stack(anchors, sx, sy)
    votes = np.zeros(seg.shape, dtype=np.uint8)      # sheared frame
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
            core2 = edt2 >= P["mv_neck_mm"]
            if not core2.any():
                continue
            lab2, n2 = ndimage.label(core2)
            if n2 == 0:
                continue
            a2 = anch_s[idx]
            sel = (a2 > 0) & (lab2 > 0)
            comp_map = np.full(n2 + 1, SENT, dtype=np.int32)  # default: blocker
            if sel.any():
                pairs = np.unique(np.stack([lab2[sel], a2[sel]]).astype(np.int64),
                                  axis=1)
                comps, first = np.unique(pairs[0], return_index=True)
                ncnt = np.bincount(pairs[0], minlength=n2 + 1)
                # elongation gate: only lengthwise-seen structures may vote
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
                    if w[1] >= P["mv_elong"] ** 2 * max(w[0], 1e-6):
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
    # back to the original frame (exact inverse shifts)
    votes = _shift2d_stack(votes, -sx, -sy)
    conflict = _shift2d_stack(conflict.astype(np.uint8), -sx, -sy).astype(bool)
    cand = (seg > 0) & (votes > 0) & ~conflict & (votes != seg)
    if not cand.any():
        qa["multiview"] = {"flipped_mm3": 0.0}
        return seg
    out = seg.copy()
    out[cand] = votes[cand]
    # gates: per-pair bound + loser-fragmentation guard
    flips = {}
    pts_from = seg[cand]
    pts_to = votes[cand]
    for a, b in zip(pts_from, pts_to):
        flips[(int(a), int(b))] = flips.get((int(a), int(b)), 0) + 1
    reverted_pairs = []
    for (a, b), n in sorted(flips.items()):
        n_min = min(int((seg == a).sum()), int((seg == b).sum()))
        undo = False
        if n > P["mv_max_shift_frac"] * max(n_min, 1):
            undo = True
        else:
            _, c_new = _components(out == a)
            _, c_old = _components(seg == a)
            if int((c_new > 0).sum()) > int((c_old > 0).sum()):
                undo = True
        if undo:
            m = cand & (seg == a) & (votes == b)
            out[m] = a
            reverted_pairs.append(f"{ID_TO_NAME[a]}->{ID_TO_NAME[b]}")
    qa["multiview"] = {
        "flipped_mm3": round(float((out != seg).sum()) * vox_mm3, 1),
        "pair_flips_mm3": {f"{ID_TO_NAME[a]}->{ID_TO_NAME[b]}":
                           round(n * vox_mm3, 1) for (a, b), n in flips.items()},
        "reverted_pairs": reverted_pairs}
    _, s_before = pro_audit(seg, affine, vox_mm3)
    _, s_after = pro_audit(out, affine, vox_mm3)
    bad = lambda s: s["n_fragmented"] + s["n_empty"] + s["n_size"] + s["n_order"]
    if bad(s_after) > bad(s_before):
        qa["multiview"]["reverted_all"] = True
        qa["flags"].append("MULTIVIEW_REVERTED_ALL")
        return seg
    return out


# --------------------------------------------------------------- stage 2f --
def _masscut_mm2(seg, edtb, vox_mm3, thresh=2.5):
    """Area of label-label boundary crossing bone thicker than thresh -
    the violation meter for the core-integrity invariant.

    Takes: seg, edtb (3D bone EDT in mm), vox_mm3, thresh.
    Does: measures the mass-cut meter described above.
    Returns: label-boundary area crossing bone thicker than thresh, in mm2.
    """
    m = np.zeros(seg.shape, dtype=bool)
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        b = (x > 0) & (y > 0) & (x != y)
        m[tuple(s_hi)] |= b
        m[tuple(s_lo)] |= b
    return float((m & (edtb >= thresh)).sum()) * (vox_mm3 ** (2 / 3))


def _boundary_edt2(mm, edt2):
    """Mean in-plane thickness along the internal label boundary of a
    labeled 2D patch (0 outside). Returns 0.0 when no internal boundary.

    Takes: mm (2D labels of one piece, zero elsewhere), edt2 (in-plane
        thickness map).
    Does: measures how thick the bone is along the piece's INTERNAL label
        boundary, the physics test of stages 2f and 2g.
    Returns: mean in-plane thickness along that boundary in mm (0 when there is
        none).
    """
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


def _badcut_pieces(seg, bone, zooms, cx_med, span=10, thresh=2.6):
    """Direct defect meter: number of supra-neck 2D core pieces (sagittal,
    within +/-span of midline) whose INTERNAL label boundary crosses bone
    thicker than thresh - a boundary through the middle of a rigid piece.
    Mixed pieces whose boundary sits at a thin waist (a legitimate joint
    through a PV-fused pair) do NOT count.

    Takes: seg, bone, zooms, cx_med (median centerline x), span (parasagittal
        half-width in slices), thresh.
    Does: counts the badcut meter described above: mixed supra-neck 2D pieces
        whose internal boundary crosses thick bone; boundaries at thin in-plane
        waists (legitimate fused joints) are excluded.
    Returns: the number of violating pieces.
    """
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


def stage2f_skeleton_relabel(seg, ct, affine, zooms, vox_mm3, qa):
    """CORE-INTEGRITY relabeling - the invariant, enforced in 3D:
    A LABEL BOUNDARY MAY ONLY PASS THROUGH THIN BONE (joint clefts, PV
    plates, necks) - NEVER THROUGH THE INTERIOR OF A THICK MASS. A facet
    knob, a blade, a body is one rigid piece and carries one label.

    Mechanism - PER-PIECE UNIFICATION (surgical, local, no seeds, no
    propagation; the global re-derivations tried first kept leaking
    through this case's pathological fusions and were auto-reverted):

    A mixed in-plane core piece IS the defect by definition: one rigid 2D
    cross-section piece (blade, knob), in-plane separated from its
    neighbors, carrying two labels. Rule: such a piece is unified to its
    majority label IFF physics says the internal boundary is impossible -
    it crosses the piece's THICK interior (in-plane EDT at the internal
    boundary >= skel_cut_mm). If the boundary sits at the piece's thin
    waist it is a legitimate joint through a PV-fused pair - skipped. The
    minority share must be <= 40% (a 50/50 split could be two whole fused
    structures). Runs over all three orthogonal views in the
    shear-straightened frame; recolor-only. Gates: the mixed-piece meter
    must not increase, audit must not degrade, per-level shift <= 4 cm3,
    else full revert.

    Takes: seg, ct, affine, zooms, vox_mm3, qa (record under 'skeleton').
    Does: the per-piece core-integrity surgery described above.
    Returns: the repaired volume, or the input after a gated full revert.
    """
    import gc
    gc.collect()
    # the pipeline crop follows the RAW prediction bbox, which scattered
    # hallucinations inflate to most of the volume; by this stage seg is
    # spine-only, so work on its tight bbox (~5x fewer voxels)
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
    bone = ct >= P["bone_hu"]
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
            l2, n2 = ndimage.label((edt2 >= P["mv_neck_mm"]) & lab2d)
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
                # physics test: internal boundary thickness within piece
                mm = np.where(m, s2, 0)
                bedt = _boundary_edt2(mm, edt2)
                if bedt < P["skel_cut_mm"]:
                    continue        # boundary at a thin waist: legit joint
                if minority_frac <= 0.40:
                    # split single structure: unify to majority
                    s2[m & (mm != major) & (mm > 0)] = major
                    continue
                # PV-fused multi-structure piece: both sides are real -
                # RELOCATE the internal boundary to the in-plane thickness
                # valley (the kiss/joint plane) instead
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
    # per-slice decisions leave ragged seams: converge them with the
    # volume-preserving 26-neighborhood interface majority vote, then fuse
    # any disconnected slivers into their surrounding label (no deletion)
    out = majority_filter(out, iters=2)
    gc.collect()
    out = pro_absorb_orphans(out, vox_mm3, max_mm3=600.0, delete_below=0.0)
    # a recolored sliver left detached from its new label (no labeled
    # shell, so absorb cannot fuse it) reverts to its ORIGINAL label -
    # envelope kept, no new fragments minted
    for lid in [int(v) for v in np.unique(out) if v != 0]:
        cc_, counts_ = _components(out == lid)
        if (counts_ > 0).sum() <= 1:
            continue
        main_ = int(counts_.argmax())
        for comp_id in np.nonzero(counts_)[0]:
            if comp_id == main_ or counts_[comp_id] * vox_mm3 > 600.0:
                continue
            comp_ = cc_ == comp_id
            if counts_[comp_id] * vox_mm3 < P["speck_mm3"]:
                sh_ = ndimage.binary_dilation(comp_, structure=STRUCT26) & ~comp_
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
    cx_med = int(round(np.median(cx)))
    mix_before = _badcut_pieces(seg, bone, zooms, cx_med)
    mix_after = _badcut_pieces(out, bone, zooms, cx_med)
    changed = int((out != seg).sum())
    deltas = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        d = (int((out == lid).sum()) - int((seg == lid).sum())) * vox_mm3 / 1e3
        if abs(d) > 0.05:
            deltas[ID_TO_NAME[lid]] = round(d, 2)
    _, s_before = pro_audit(seg, affine, vox_mm3)
    _, s_after = pro_audit(out, affine, vox_mm3)
    bad = lambda s: s["n_fragmented"] + s["n_empty"] + s["n_size"] + s["n_order"]
    rec = {"changed_mm3": round(changed * vox_mm3, 1),
           "masscut_before_mm2": round(mc_before, 0),
           "masscut_after_mm2": round(mc_after, 0),
           "badcut_pieces_before": mix_before,
           "badcut_pieces_after": mix_after,
           "volume_deltas_cm3": deltas}
    qa["skeleton"] = rec
    # gate on the DIRECT defect meter (split pieces) + audit + bounded
    # per-level shifts; masscut is reported only - the correct boundary
    # through a plate-fused joint is legitimately thick, so thickness
    # alone would veto right fixes.
    over = [n for n, d in deltas.items() if abs(d) > 4.0]
    if mix_after > mix_before or bad(s_after) > bad(s_before) or over:
        rec["reverted_all"] = True
        if over:
            rec["over_shift"] = over
        qa["flags"].append("SKELETON_REVERTED_ALL")
        return seg_orig
    out_full = seg_orig.copy()
    out_full[tight] = out
    return out_full


# --------------------------------------------------------------- stage 2d --
def stage2d_reclaim_pool(seg, raw, ct, zooms, vox_mm3, qa):
    """Envelope rule: the raw prediction is the outer boundary - real BONE
    inside it gets its CLASS fixed, it is never deleted. Runs last, so it
    reclaims whatever the evidence stages removed and no band re-claimed
    (the model systematically labels a transverse process with the level
    above; stage 1 then removes it as far-from-its-label and, outside a
    suspect band, nothing brought it back). Owner of each dropped bone
    component (>= speck size) is decided by two independent physics:

      1. AXIAL-RING VOTE (primary): in the axial projection a vertebra is
         one closed ring, and a true process fragment is 2D-bone-connected
         to its own ring within its own slices, while ribs cross slices as
         disconnected in-plane islands. Vote = the label sharing the most
         in-slice 2D bone-component pixels with the fragment.
      2. 3D BONE-GEODESIC LINK: masked dilation through CT bone
         (reclaim_link_mm); vote = the label with the most reached voxels.

    The ring vote wins disagreements (both are logged). The fragment is
    re-attached through the stage-1 CT-bone corridor at a widened neck
    (reclaim_neck_mm) when possible. A fragment linked to NO vertebra
    through bone is, by the label definition, not vertebra (pelvic or rib
    hallucinations wearing a vertebra label) - it stays out and is flagged,
    never silently dropped. Sub-bone-HU raw mass stays removed (evidence).

    Takes: seg, raw, ct, zooms, vox_mm3, qa (record under
        'reclaim_cm3_by_level' plus flags).
    Does: the axial-ring envelope reclamation described above.
    Returns: the volume with re-owned raw-labeled fragments attached and fused.
    """
    bone = ct >= P["bone_hu"]
    pool = (raw > 0) & (seg == 0) & bone
    cc, counts = _components(pool)
    st2 = ndimage.generate_binary_structure(2, 1)
    summary = {}
    for comp_id in np.nonzero(counts)[0]:
        vol = float(counts[comp_id]) * vox_mm3
        if vol < P["speck_mm3"]:
            continue
        comp = cc == comp_id
        objs = ndimage.find_objects(comp.astype(np.uint8))[0]
        rl = raw[comp]
        ids_r, cnt_r = np.unique(rl[rl > 0], return_counts=True)
        rec = {"stage": "2d", "vol_mm3": round(vol, 1),
               "raw_label": ID_TO_NAME[int(ids_r[cnt_r.argmax()])]}
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
        pad_vox = np.ceil((P["reclaim_link_mm"] + 2.0) / np.asarray(zooms)).astype(int)
        slc = _bbox_pad(objs, pad_vox, seg.shape)
        it = max(int(np.ceil(P["reclaim_link_mm"] / min(zooms))), 1)
        reach = ndimage.binary_dilation(comp[slc], structure=STRUCT6,
                                        iterations=it,
                                        mask=bone[slc] | comp[slc])
        labs = seg[slc][reach]
        ids3, cnt3 = np.unique(labs[labs > 0], return_counts=True)
        geo = int(ids3[cnt3.argmax()]) if ids3.size else 0
        rec["ring_vote"] = ID_TO_NAME.get(ring, "-")
        rec["geodesic_vote"] = ID_TO_NAME.get(geo, "-")
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
                        neck_mm=P["reclaim_neck_mm"])
        seg[comp] = lid
        if added is not None:
            seg[added] = lid
            rec["action"] = "RECLAIMED_BRIDGED"
            rec["bridge_added_mm3"] = round(float(added.sum()) * vox_mm3, 1)
        else:
            rec["action"] = "RECLAIMED_KEPT_SEPARATE"
        nm = ID_TO_NAME[lid]
        summary[nm] = round(summary.get(nm, 0.0) + vol / 1000.0, 2)
        qa["records"].append(rec)
    qa["reclaim_cm3_by_level"] = summary
    # boundary slivers that could not bridge fuse into the label they touch
    # (envelope rule: nothing deleted - delete_below=0)
    return pro_absorb_orphans(seg, vox_mm3, max_mm3=300.0, delete_below=0.0)


def majority_filter(seg, iters=3):
    """Iterated 26-neighborhood majority vote on label-label interface voxels.

    Collapses voxel-scale interdigitation between adjacent labels (the source
    of spurious Euler handles). Label<->label swaps only; the mode must
    strictly beat the current label locally, so the filter converges and is
    approximately volume-preserving.

    Takes: seg, iters.
    Does: iterated 26-neighborhood majority voting restricted to label-label
        interfaces, the volume-preserving seam smoother shared by stages 2f,
        2g, and 3.
    Returns: the smoothed label volume.
    """
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    out = seg.copy()
    for _ in range(iters):
        diff = np.zeros(out.shape, dtype=bool)
        for ax in range(3):
            s_hi = [slice(None)] * 3; s_lo = [slice(None)] * 3
            s_hi[ax] = slice(1, None); s_lo[ax] = slice(None, -1)
            x, y = out[tuple(s_hi)], out[tuple(s_lo)]
            b = (x > 0) & (y > 0) & (x != y)
            diff[tuple(s_hi)] |= b
            diff[tuple(s_lo)] |= b
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


def pro_absorb_orphans(seg, vox_mm3, max_mm3=1200.0, delete_below=100.0):
    """Absorb non-largest components below max_mm3 into the label dominating
    their shell (remove if below delete_below and shell-free; delete_below=0
    disables removal - used after pool reclamation, where the envelope rule
    forbids deleting raw-labeled bone).

    Takes: seg, vox_mm3, max_mm3 (largest non-main component that may fuse),
        delete_below (specks under this with no labeled shell are dropped; 0
        disables all deletion for envelope-safe stages).
    Does: fuses every non-main component into the label surrounding it instead
        of deleting it.
    Returns: the volume with orphan components fused.
    """
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


# ---------------------------------------------------------------- stage 3 --
def stage3_smooth(seg, ct, zooms, vox_mm3, qa):
    """Takes: seg, ct, zooms, vox_mm3, qa (records under 'smooth').
    Does: regularization: interface majority vote, orphan absorption, enclosed-
        hole and directional pit filling, then bounded volume-preserving SDT
        smoothing (changes within max_dev_mm, additions bone-gated, per-label
        Dice and component-count guards with self-revert).
    Returns: the regularized label volume.
    """
    seg = majority_filter(seg, iters=3)
    seg = pro_absorb_orphans(seg, vox_mm3)
    bone = ct >= P["bone_hu"]
    pad_vox = np.ceil(P["smooth_pad_mm"] / np.asarray(zooms))
    objs = ndimage.find_objects(seg)
    per_label = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        sl = _bbox_pad(objs[lid - 1], pad_vox, seg.shape)
        m = seg[sl] == lid
        others = (seg[sl] > 0) & ~m
        m1 = ndimage.binary_fill_holes(m)
        # directional pit/window filling: 2D holes in sagittal and coronal
        # slices catch wall windows and tunnels that are not 3D-enclosed;
        # axial is EXCLUDED so the vertebral canal is never filled.
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
        s = ndimage.gaussian_filter(sdt, sigma=[P["sigma_mm"] / z for z in zooms])
        target = int(m_f.sum())
        lo, hi = -2.0, 2.0
        tau = 0.0
        for _ in range(40):
            tau = 0.5 * (lo + hi)
            n = int((s > tau).sum())
            if abs(n - target) <= P["vol_tol"] * target:
                break
            lo, hi = (tau, hi) if n > target else (lo, tau)
        per_label[lid] = (sl, m_f, sdt, s, float(tau))
        qa["smooth"].append({"label": ID_TO_NAME[lid],
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
            band = np.abs(sdt_l) <= P["max_dev_mm"]
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
        rec = next(r for r in qa["smooth"] if r["label"] == ID_TO_NAME[lid])
        rec["dice_vs_presmooth"] = round(dice, 4)
        if dice < P["min_dice"] or int((c_new > 0).sum()) > int((c_old > 0).sum()):
            out[sl][m_f] = lid
            out[sl][(out[sl] == lid) & ~m_f] = 0
            rec["reverted"] = True
    return pro_absorb_orphans(out, vox_mm3)


# ==========================================================================
# PART 4b  NAME CHECK (report only - it never changes the labels)
#
#  How it decides, in plain words:
#    1. One body per vertebra label (the part in front of the spinal canal), in order
#       bottom to top. Counting on the labels means a body that is hard to see in the
#       CT (thick slices, thin discs) can never shift the count.
#    2. Landmarks in the CT: the sacrum under the lowest vertebra (= L5); the ribs,
#       each given to the vertebra its inner end touches (in 3D, not by height - on
#       a curved spine the T12 rib head sits at the height of L1). Lowest vertebra
#       with ribs on both sides = T12; highest = T1, used only when 12 rib levels
#       were found (the first rib is often missed).
#    3. Count from those landmarks and compare with the names the labels have.
#    4. Boundary check, independent of the colours: shrink the mask until touching
#       bodies separate into cores (Hao's idea). A colour on two cores, or a core
#       holding two colours, is a boundary suspect - except a core on the arch (not
#       a body), or two neighbours that came out as one core because the disc between
#       them was not resolved (the second colour has no core of its own).
#         CONFIRMED            names match the landmarks, no boundary suspect
#         WRONG_NAMES          names are shifted against the landmarks
#         NEEDS_BOUNDARY_FIX   a colour covers two bodies, or a body holds two colours
#         UNSURE               the landmarks disagree with each other
#         UNCHECKED            no sacrum or rib landmark found
# ==========================================================================
def _label_bodies(seg, g, reg):
    """One body per vertebra label: the centre of its labelled voxels in front of the
    spinal canal (all its voxels if that part is tiny), bottom-up."""
    bodies = []
    objs = ndimage.find_objects(seg)
    for l in range(1, 25):
        if l > len(objs) or objs[l - 1] is None:
            continue
        sl = objs[l - 1]
        m = seg[sl] == l
        mb = m & reg[sl]
        c = (np.argwhere(mb if mb.sum() >= 50 else m) + np.array([s.start for s in sl])).mean(axis=0)
        bodies.append({"z": float(c[2]), "x": float(c[0]), "y": float(c[1]), "ai": l})
    bodies.sort(key=lambda b: float(up(b["z"], g)))
    return bodies


def _ribs_by_touch(seg, rmask, g, bodies):
    """Each accepted rib goes to the vertebra its inner end touches (nearest labelled
    voxel in 3D). Returns the ribs and a mask (100 + body index)."""
    idx = {b["ai"]: k for k, b in enumerate(bodies)}
    lab, _ = ndimage.label(rmask > 0, structure=S26)
    pad = np.ceil((CFG["rib_attach_mm"] + 2.0) / np.asarray(g.zooms)).astype(int)
    ribs, mask = [], np.zeros(seg.shape, np.uint8)
    for c, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        box = tuple(slice(max(s.start - p, 0), min(s.stop + p, n))
                    for s, p, n in zip(sl, pad, seg.shape))
        rib = lab[box] == c
        spine = seg[box] > 0
        if not spine.any():
            continue
        d, near = ndimage.distance_transform_edt(~spine, sampling=g.zooms, return_indices=True)
        pts = np.argwhere(rib)
        dd = d[tuple(pts.T)]
        q = pts[dd.argmin()]
        v = int(seg[box][tuple(near[:, q[0], q[1], q[2]])])
        zq = int(q[2] + box[2].start)
        lat = (q[0] + box[0].start - g.cx[zq]) * g.zooms[0]
        ribs.append({"body": idx[v], "vertebra": NAMES[v], "side": "+x" if lat > 0 else "-x",
                     "inner_end_mm": round(float(dd.min()), 1)})
        mask[box][rib] = 100 + idx[v]
    return ribs, mask


def check_array(seg, ct, affine):
    """Returns (seg, report, landmark_mask). Report only: the labels are returned as is."""
    zooms = tuple(float(z) for z in nib.affines.voxel_sizes(affine))
    g = build_geometry(seg, ct, zooms, axis_frame(affine))
    reg = body_region(seg, g, 0, seg.shape[2] - 1)
    bodies = _label_bodies(seg, g, reg)
    lm = np.zeros(seg.shape, np.uint8)
    rep = {"vertebrae": len(bodies), "notes": []}
    if len(bodies) < 3:
        rep["verdict"] = "UNCHECKED (fewer than 3 vertebrae labelled)"
        return seg, rep, lm
    ai = [b["ai"] for b in bodies]
    n = len(bodies)

    # --- landmarks
    _, rmask, rib_diag = find_ribs(seg, ct, g, bodies)
    ribs, rmask = _ribs_by_touch(seg, rmask, g, bodies)
    skull = [r for r in ribs if ai[r["body"]] >= ID["C2"]]   # no rib ever reaches C1/C2:
    if skull:                                                 # that is skull-base bone
        ribs = [r for r in ribs if ai[r["body"]] < ID["C2"]]
        for k in {r["body"] for r in skull}:
            rmask[rmask == 100 + k] = 0
        rep["notes"].append(f"{len(skull)} bone piece(s) at the skull base (next to C1/C2) "
                            "looked like ribs - ignored")
    sac, smask = find_sacrum(seg, ct, g, bodies)
    sides = {}
    for r in ribs:
        sides.setdefault(r["body"], set()).add(r["side"])
    rib_bodies = sorted(b for b, sd in sides.items() if len(sd) == 2)
    anchors = []
    if sac.get("status") == "PRESENT":
        anchors.append({"rule": "sacrum under the lowest vertebra -> L5", "body": 0, "name": L5})
    if rib_bodies:
        lo, hi = rib_bodies[0], rib_bodies[-1]
        if lo > 0:
            anchors.append({"rule": "lowest vertebra with ribs -> T12", "body": lo, "name": T12})
        if hi - lo == T1 - T12 and hi < n - 1:
            anchors.append({"rule": "highest vertebra with ribs -> T1", "body": hi, "name": T1})
        else:
            rep["notes"].append(f"ribs found on {hi - lo + 1} levels ({NAMES[ai[lo]]}..{NAMES[ai[hi]]}), "
                                "not 12 - the first rib (or another) was not found, so the T1 "
                                "landmark is not used")

    # --- count: runs split where a vertebra seems missing; each run named by its anchors
    hs = np.array([float(up(b["z"], g)) for b in bodies])
    gaps = np.diff(hs)
    runs, cur = [], [0]
    for i, gp in enumerate(gaps):
        nb = [gaps[j] for j in range(max(0, i - 2), min(len(gaps), i + 3)) if j != i]
        if nb and gp > CFG["missing_body_ratio"] * float(np.median(nb)):
            runs.append(cur)
            cur = []
        cur.append(i + 1)
    runs.append(cur)
    names = [None] * n
    disagree = []
    for run in runs:
        offs = {a["name"] - a["body"] for a in anchors if a["body"] in run}
        if len(offs) == 1:
            o = offs.pop()
            for i in run:
                names[i] = o + i if 1 <= o + i <= 24 else None
        elif offs:
            disagree.append(", ".join(f"{a['rule']} (on {NAMES[ai[a['body']]]})"
                                      for a in anchors if a["body"] in run))

    # --- boundary check with label-independent cores
    cores, radius = find_cores(seg, ct, g)
    on_body = []
    for c in cores:
        p = tuple(min(max(int(round(c[k])), 0), seg.shape[i] - 1) for i, k in enumerate("xyz"))
        if reg[p]:
            on_body.append(c)
    owners = {}
    for c in on_body:
        owners.setdefault(c["ai"], []).append(c)
    suspects = [({a}, f"{NAMES[a]} covers {len(cs)} bodies")
                for a, cs in owners.items() if len(cs) > 1]
    merged = []
    for c in on_body:
        if c["purity"] >= 0.9 or not c["second"]:
            continue
        s = c["second"]
        if s not in owners and abs(s - c["ai"]) == 1:
            merged.append((c["ai"], s))
        else:
            suspects.append(({c["ai"], s}, f"the {NAMES[c['ai']]} body also holds {NAMES[s]} "
                                           f"(purity {c['purity']})"))
    involved = set().union(*(cols for cols, _ in suspects)) if suspects else set()
    for a, s in merged:
        if not {a, s} & involved:
            rep["notes"].append(f"{NAMES[a]} and {NAMES[s]} came out as one body core (the disc "
                                "between them is not resolved in the CT) - not a label error")
    suspects = [txt for _, txt in suspects]
    if len(cores) > len(on_body):
        rep["notes"].append(f"{len(cores) - len(on_body)} core(s) on a vertebral arch, not a body - ignored")

    # --- table + verdict
    table = []
    for i, b in enumerate(bodies):
        table.append({"body": i, "colour_now": NAMES[ai[i]],
                      "anatomy_says": NAMES[names[i]] if names[i] else None,
                      "match": names[i] is None or ai[i] == names[i],
                      "ribs": "both" if len(sides.get(i, ())) == 2 else
                              ("one side" if i in sides else "none"),
                      "on_sacrum": i == 0 and sac.get("status") == "PRESENT",
                      "body_core_found": ai[i] in owners})
    for k in range(n):
        lm[rmask == 100 + k] = names[k] or ai[k]
    if smask is not None:
        lm = np.maximum(lm, smask)
    rep.update(anchors=[dict(a, name=NAMES[a["name"]]) for a in anchors],
               anchors_agree=not disagree, sacrum=sac, rib_search=rib_diag, ribs=ribs,
               bodies=table, body_cores={"found": len(on_body), "shrink_radius_mm": radius})
    wrong = [t for t in table if not t["match"]]
    if not anchors:
        rep["verdict"] = "UNCHECKED (no sacrum or rib landmark found)"
    elif disagree:
        rep["verdict"] = "UNSURE"
        rep["why"] = ["landmarks disagree: " + d for d in disagree]
    elif wrong:
        rep["verdict"] = "WRONG_NAMES"
        rep["renames"] = {t["colour_now"]: t["anatomy_says"] for t in wrong}
    elif suspects:
        rep["verdict"] = "NEEDS_BOUNDARY_FIX"
    else:
        rep["verdict"] = "CONFIRMED"
    if suspects:
        rep.setdefault("why", []).extend(suspects)
    if wrong:
        rep["wrong_bodies"] = [f"{t['colour_now']} should be {t['anatomy_says']}" for t in wrong]
    return seg, rep, lm


# ==========================================================================
# PARTS 2-4a  FINAL POLISH + CHECKS
#
#  1. CLEANUP (only small, local edits; nothing is deleted)
#     a. Stray pieces: a piece of one vertebra that is NOT connected to that vertebra's
#        main body (or hangs on it only by a thread ~1 voxel thin), is small
#        (< island_max_cm3), and is surrounded mostly by ANOTHER
#        vertebra takes that vertebra's colour (e.g. cyan specks on a magenta body).
#        Unlike Pro's island rule, only vertebra voxels around the piece are counted,
#        so patches on the outer surface are caught too.
#     b. Holes: background enclosed by one vertebra (in 3D, or in any 2D slice
#        direction) is filled with that vertebra's colour - only if the hole is bone
#        on the CT (mean >= 100 HU), so the spinal canal is never filled.
#     c. Pits: small dents in a vertebra's surface are closed (about 1.5 mm), but only
#        where the CT shows bone - so discs (darker) and soft tissue are never filled.
#     Never filled: any voxel touching a DIFFERENT vertebra (that gap is the disc).
#     Guard: a vertebra may grow by at most max_growth (3%); otherwise its fill is
#     skipped. The whole cleanup is kept only if no vertebra ends up in more pieces.
#
#  2. CHECKS (report only - they never change the labels)
#     a. Hao agreement (if --hao_dir is given): Pro's and Hao's BODIES are compared for
#        L2..T5 (the range Hao edits). Agreement per vertebra (Dice); disagreeing body
#        voxels go into review.nii.gz (value 1).
#     b. Anatomy check (ours, PART 4b): vertebrae counted from the sacrum (= L5) and
#        the ribs (lowest rib pair = T12, highest = T1 when 12 levels are found).
#        Verdict CONFIRMED / WRONG_NAMES / NEEDS_BOUNDARY_FIX / UNSURE with a
#        per-vertebra table; vertebrae whose name disagrees with the anatomy go into
#        review.nii.gz (value 2). landmarks.nii.gz shows the ribs and sacrum it found.
#
# ==========================================================================
FIN_CFG = {
    "island_max_cm3": 2.0,      # stray pieces larger than this are left alone
    "island_host_frac": 0.6,    # share of the surrounding vertebra voxels that must be one host
    "hole_max_cm3": 3.0,        # enclosed holes larger than this are left alone
    "hole_min_hu": 100.0,       # a hole is filled only if it is bone on the CT (not the canal)
    "pit_max_mm3": 200.0,       # dents larger than this are left alone
    "pit_close_mm": 1.5,        # surface dents closed up to about this size
    "bone_hu": 150.0,           # pits are only closed where CT >= this
    "max_growth": 0.03,         # a vertebra may grow by at most 3 %
    "hao_first": "L2", "hao_last": "T5", "agree_dice": 0.90,
    # specks (PART 3)
    "speck_neck_radii_mm": (1.0, 1.5),  # shrink sizes: necks thinner than ~2x are cut
    "speck_max_mm3": 150.0,       # loose parts up to this size are specks
    "speck_host_frac": 0.5,       # other vertebrae must make up this share of a speck's
                                  # outside for it to take their colour (else deleted)
    "speck_thick_max_mm3": 50.0,  # thick slices (a shrink of speck_neck_radii_mm cannot
                                  # reach the next slice): pieces up to this size joined
                                  # to their vertebra only through one slice, touching no
                                  # other vertebra, are deleted too
    "speck_rounds": 4,
}
S6 = ndimage.generate_binary_structure(3, 1)


# ============================================================== cleanup ===
def _pieces(seg, vox):
    out = {}
    objs = ndimage.find_objects(seg)
    for l in range(1, 25):
        if l <= len(objs) and objs[l - 1] is not None:
            _, sz = components(seg[objs[l - 1]] == l)
            out[NAMES[l]] = int((sz * vox > 20).sum())
    return out


def _split_threaded_pieces(out, vox, moved):
    """Specks often hang on their own vertebra by a thread one or two voxels thin, so
    they are not separate pieces. Shave one voxel off the vertebra (threads vanish,
    real bone stays connected), find parts that come apart, grow each part back by
    that voxel inside its own colour, and recolour it if it sits on another vertebra."""
    objs = ndimage.find_objects(out)
    for l in range(1, 25):
        if l > len(objs) or objs[l - 1] is None:
            continue
        sl = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in objs[l - 1])
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
            # grow the part back (2 voxels) inside its own colour, away from the main body
            piece = ndimage.binary_dilation(lab == cid, structure=S26, iterations=2,
                                            mask=m & ~near_main) | ((lab == cid) & m)
            n = int(piece.sum())
            if n == 0 or n * vox / 1e3 > FIN_CFG["island_max_cm3"]:
                continue
            shell = ndimage.binary_dilation(piece, structure=S26) & ~piece
            v = cur[shell]
            v = v[(v != 0) & (v != l)]
            if v.size < 5:
                continue
            ids, cnt = np.unique(v, return_counts=True)
            if cnt.max() / v.size >= FIN_CFG["island_host_frac"]:
                host = int(ids[cnt.argmax()])
                cur[piece] = host
                moved.append({"from": NAMES[l], "to": NAMES[host], "mm3": round(n * vox, 1),
                              "how": "was hanging by a thin thread"})


def fix_stray_pieces(seg, vox, rep):
    """Non-main pieces of a vertebra that sit on another vertebra take its colour
    (first the ones hanging by a thin thread, then fully separate ones)."""
    out = seg.copy()
    moved = []
    for _ in range(2):
        _split_threaded_pieces(out, vox, moved)
    for _ in range(2):                                   # a second pass catches chains
        objs = ndimage.find_objects(out)
        for l in range(1, 25):
            if l > len(objs) or objs[l - 1] is None:
                continue
            sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in objs[l - 1])
            lab, sizes = components(out[sl] == l)
            if (sizes > 0).sum() <= 1:
                continue
            main = sizes.argmax()
            for cid in np.nonzero(sizes)[0]:
                if cid == main or sizes[cid] * vox / 1e3 > FIN_CFG["island_max_cm3"]:
                    continue
                comp = lab == cid
                shell = ndimage.binary_dilation(comp, structure=S26) & ~comp
                v = out[sl][shell]
                v = v[(v != 0) & (v != l)]
                if v.size < 5:
                    continue                             # touches no other vertebra: leave it
                ids, n = np.unique(v, return_counts=True)
                host = int(ids[n.argmax()])
                if n.max() / v.size >= FIN_CFG["island_host_frac"]:
                    out[sl][comp] = host
                    moved.append({"from": NAMES[l], "to": NAMES[host],
                                  "mm3": round(float(sizes[cid] * vox), 1)})
    rep["stray_pieces_moved"] = moved
    return out


def _ball(radius_mm, zooms):
    r = [max(int(np.floor(radius_mm / z)), 1) for z in zooms]
    g = np.ogrid[tuple(slice(-k, k + 1) for k in r)]
    return sum((gi * z / radius_mm) ** 2 for gi, z in zip(g, zooms)) <= 1.0


def fill_holes_and_pits(seg, ct, zooms, rep):
    """Fill holes enclosed by one vertebra and close small surface pits on bone,
    never touching a different vertebra."""
    vox = float(np.prod(zooms))
    out = seg.copy()
    ball = _ball(FIN_CFG["pit_close_mm"], zooms)
    pad = max(ball.shape) + 2
    filled = {}
    skipped = []
    objs = ndimage.find_objects(seg)
    for l in range(1, 25):
        if l > len(objs) or objs[l - 1] is None:
            continue
        sl = tuple(slice(max(s.start - pad, 0), s.stop + pad) for s in objs[l - 1])
        cur = out[sl]
        m = cur == l
        others = (cur > 0) & ~m
        near_other = ndimage.binary_dilation(others, structure=S26)
        # holes: 3D, then 2D in each of the three slice directions
        cand = ndimage.binary_fill_holes(m) & ~m
        for ax in range(3):
            mm = np.moveaxis(m, ax, 0)
            f2 = np.zeros_like(mm)
            for k in range(mm.shape[0]):
                if mm[k].any():
                    f2[k] = ndimage.binary_fill_holes(mm[k])
            cand |= np.moveaxis(f2, 0, ax) & ~m
        cand &= (cur == 0) & ~near_other
        # keep only holes that are BONE on the CT: the spinal canal (fluid, dark) is
        # also "enclosed" by the vertebral ring, and must never be filled
        lab, sizes = components(cand)
        hu = ndimage.mean(ct[sl].astype(np.float32), labels=lab,
                          index=np.arange(1, len(sizes))) if len(sizes) > 1 else []
        keep = [i + 1 for i, h in enumerate(hu)
                if h >= FIN_CFG["hole_min_hu"] and sizes[i + 1] * vox / 1e3 <= FIN_CFG["hole_max_cm3"]]
        cand = np.isin(lab, keep) if keep else np.zeros_like(cand)
        # pits: closing, only on bone, never next to another vertebra, small pieces only
        closed = ndimage.binary_closing(m, structure=ball)
        pit = closed & ~m & (cur == 0) & ~near_other & (ct[sl] >= FIN_CFG["bone_hu"])
        plab, psz = components(pit)
        big = np.nonzero(psz * vox > FIN_CFG["pit_max_mm3"])[0]
        if big.size:
            pit &= ~np.isin(plab, big)
        add = cand | pit
        n = int(add.sum())
        if n == 0:
            continue
        if n > FIN_CFG["max_growth"] * int(m.sum()):
            skipped.append({"vertebra": NAMES[l], "would_add_mm3": round(n * vox, 1)})
            continue
        cur[add] = l
        filled[NAMES[l]] = round(n * vox, 1)
    rep["filled_mm3"] = filled
    rep["fill_skipped_too_large"] = skipped
    return out


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


# =============================================================== checks ===
def hao_agreement(seg, hao, ct, affine, rep):
    """Compare Pro's and Hao's BODIES for L2..T5. Returns review mask (1 = disagree)."""
    zooms = tuple(float(z) for z in nib.affines.voxel_sizes(affine))
    g = build_geometry(seg, ct, zooms, axis_frame(affine))
    body = body_region(np.maximum(seg, hao), g, 0, seg.shape[2] - 1)
    lo, hi = ID[FIN_CFG["hao_first"]], ID[FIN_CFG["hao_last"]]
    rows, review = {}, np.zeros(seg.shape, np.uint8)
    for l in range(lo, hi + 1):
        a, b = body & (seg == l), body & (hao == l)
        s = int(a.sum() + b.sum())
        if s == 0:
            continue
        d = 2 * int((a & b).sum()) / s
        rows[NAMES[l]] = round(d, 3)
        if d < FIN_CFG["agree_dice"]:
            review[a ^ b] = 1
    rep["body_dice_pro_vs_hao"] = rows
    rep["agree"] = [k for k, v in rows.items() if v >= FIN_CFG["agree_dice"]]
    rep["disagree"] = [k for k, v in rows.items() if v < FIN_CFG["agree_dice"]]
    return review


def anatomy_check(seg, ct, affine, rep):
    """Our sacrum + rib name check. Report only. Returns (review, landmarks)."""
    _, r, lm = check_array(seg, ct, affine)
    rep.update({k: r[k] for k in ("verdict", "anchors", "anchors_agree", "bodies", "why",
                                  "wrong_bodies", "renames", "notes", "ribs", "body_cores",
                                  "rib_search", "sacrum") if k in r})
    review = np.zeros(seg.shape, np.uint8)
    for t in r.get("bodies", []):
        if not t["match"]:
            review[seg == ID[t["colour_now"]]] = 2
    return review, lm


# ================================================================ specks ===
def _neck_parts(m, shrink, zooms):
    """Shrink the mask by the structure `shrink` (a ball of r mm: necks thinner than
    ~2 r come apart). Every voxel goes to its nearest core, so a part comes back at
    full size (its core + its half of the neck)."""
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


def _face_parts(m):
    """Pieces that are not FACE-connected to the main piece (a corner or edge join is
    invisible to 26-connectivity but shows as a separate blob in a 3D view)."""
    lab, n = ndimage.label(m, structure=S6)
    if n <= 1:
        return []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = sizes.argmax()
    return [lab == c for c in np.nonzero(sizes)[0] if c != main]


def _loose_pieces(seg):
    """Face-connected pieces per vertebra (only those in more than one piece)."""
    out = {}
    objs = ndimage.find_objects(seg)
    for l in range(1, 25):
        if l <= len(objs) and objs[l - 1] is not None:
            n = ndimage.label(seg[objs[l - 1]] == l, structure=S6)[1]
            if n > 1:
                out[NAMES[l]] = int(n)
    return out


def _fix_speck(cur, p, l, vox, rep, why):
    """Speck p of vertebra l takes the colour of the vertebra it sits on, else is deleted."""
    ring = cur[ndimage.binary_dilation(p, structure=S6) & ~p]
    outside = ring[ring != l]
    other = outside[outside != 0]
    mm3 = round(float(p.sum()) * vox, 1)
    if other.size and other.size >= FIN_CFG["speck_host_frac"] * outside.size:
        ids, n = np.unique(other, return_counts=True)
        host = int(ids[n.argmax()])
        cur[p] = host
        rep["recoloured"].append({"from": NAMES[l], "to": NAMES[host], "mm3": mm3, "why": why})
    else:
        cur[p] = 0
        rep["deleted_mm3"][NAMES[l]] = round(rep["deleted_mm3"].get(NAMES[l], 0.0) + mm3, 1)
        rep["deleted_pieces"] += 1


def remove_specks(seg, zooms, rep):
    """PART 3. Pieces hanging on by a thin neck (or by a corner / edge only) and not
    bigger than speck_max_mm3 are specks: each takes the colour of the vertebra it
    sits on, or is deleted when it sits on none. Repeated until nothing changes."""
    vox = float(np.prod(zooms))
    out = seg.copy()
    max_vox = FIN_CFG["speck_max_mm3"] / vox
    # thick slices: no shrink below reaches the next slice, so a piece joined only
    # through the slice above / below is never cut; one extra voxel shrink that does
    thick = max(zooms) > max(FIN_CFG["speck_neck_radii_mm"])
    thick_vox = FIN_CFG["speck_thick_max_mm3"] / vox
    rep.update(loose_pieces_before=_loose_pieces(seg), recoloured=[], deleted_mm3={},
               deleted_pieces=0)
    for _ in range(FIN_CFG["speck_rounds"]):
        n0 = len(rep["recoloured"]) + rep["deleted_pieces"]
        objs = ndimage.find_objects(out)
        for l in range(1, 25):
            if l > len(objs) or objs[l - 1] is None:
                continue
            sl = tuple(slice(max(s.start - 4, 0), s.stop + 4) for s in objs[l - 1])
            cur = out[sl]
            for r in FIN_CFG["speck_neck_radii_mm"]:
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
        if len(rep["recoloured"]) + rep["deleted_pieces"] == n0:
            break
    rep["loose_pieces_after"] = _loose_pieces(out)
    rep["changed_mm3"] = round(float((out != seg).sum()) * vox, 1)
    return out


# ================================================================ driver ===
def _spine_crop(seg_full, affine):
    """Box around the spine (+ crop_margin_mm) and the affine of that box."""
    zooms = nib.affines.voxel_sizes(affine)
    lo, hi = spine_bbox(seg_full)
    pad = np.maximum(np.round(CFG["crop_margin_mm"] / zooms).astype(int), 2)
    lo, hi = np.maximum(lo - pad, 0), np.minimum(hi + pad + 1, seg_full.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    aff = affine.copy()
    aff[:3, 3] = (affine @ np.r_[lo, 1.0])[:3]
    return sl, aff


def run_pro(seg_full, ct_full, img, case):
    """PART 1: ShapeKit-Pro on one case. Returns (full-grid labels, Pro's QA record)."""
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    vox_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
    # crop to the spine region (+margin) for all processing
    pad = np.maximum(np.round(P["crop_margin_mm"] / np.asarray(zooms)).astype(int), 2)
    nz = np.nonzero(seg_full)
    lo = np.maximum(np.array([c.min() for c in nz]) - pad, 0)
    hi = np.minimum(np.array([c.max() for c in nz]) + pad + 1, seg_full.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    seg = seg_full[sl].copy()
    ct = np.ascontiguousarray(ct_full[sl])
    aff = img.affine.copy()
    aff[:3, 3] = (img.affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]

    qa = {"case": case, "params": P, "records": [], "bands": [],
          "flags": [], "polish": [], "smooth": []}
    _, s0 = pro_audit(seg, aff, vox_mm3)
    log.info("%s input audit: %s", case, s0)
    raw = seg.copy()
    seg = stage1_triage(seg, ct, zooms, vox_mm3, qa["records"])
    seg = stage2a_islands(seg, vox_mm3, qa["records"])
    seg = stage2b_arbitrate(seg, raw, ct, aff, zooms, vox_mm3, qa)
    seg = stage2c_interface_polish(seg, ct, zooms, vox_mm3, qa)
    seg = stage3_smooth(seg, ct, zooms, vox_mm3, qa)
    seg = stage2d_reclaim_pool(seg, raw, ct, zooms, vox_mm3, qa)
    seg = stage2e_multiview_recolor(seg, ct, aff, zooms, vox_mm3, qa)
    seg = stage2f_skeleton_relabel(seg, ct, aff, zooms, vox_mm3, qa)
    seg = stage2g_imbrication(seg, ct, aff, zooms, vox_mm3, qa)
    upv_t, upv_p = _imbrication_cm3(seg, ct, zooms, vox_mm3)
    qa["imbrication_upv_cm3"] = {"total": upv_t, "per_level": upv_p}
    bone_f = ct >= P["bone_hu"]
    qa["envelope"] = {
        "dropped_bone_cm3": round(float(((raw > 0) & (seg == 0) & bone_f).sum())
                                  * vox_mm3 / 1e3, 2),
        "dropped_subbone_cm3": round(float(((raw > 0) & (seg == 0) & ~bone_f).sum())
                                     * vox_mm3 / 1e3, 2),
        "recolored_cm3": round(float(((raw > 0) & (seg > 0) & (raw != seg)).sum())
                               * vox_mm3 / 1e3, 2),
        "added_beyond_raw_cm3": round(float(((raw == 0) & (seg > 0)).sum())
                                      * vox_mm3 / 1e3, 2)}
    rows, s1 = pro_audit(seg, aff, vox_mm3)
    qa["audit_before"], qa["audit_after"] = s0, s1
    qa["audit_rows_after"] = rows
    log.info("%s output audit: %s", case, s1)
    out_full = np.zeros(seg_full.shape, dtype=np.uint8)
    out_full[sl] = seg
    return out_full, qa


def run_hao(seg_full, ct_full, img):
    """The Hao pipeline (refine) on one case, for the agreement check. Full-grid labels."""
    sl, aff = _spine_crop(seg_full, img.affine)
    out, _ = refine(np.ascontiguousarray(seg_full[sl]), np.ascontiguousarray(ct_full[sl]),
                    aff, {})
    final = np.zeros_like(seg_full)              # outside the crop: far blobs removed
    final[sl] = out
    return final


def process_case(case_dir: Path, ct_root: Path, out_dir: Path, hao_check=True, specks=True):
    t0 = time.time()
    case = case_dir.name
    img, seg_full, ct_full = load_case(case_dir, ct_root)
    log.info("%s: part 1 - ShapeKit-Pro", case)
    pro_full, qa = run_pro(seg_full, ct_full, img, case)
    hao_full = None
    if hao_check:
        log.info("%s: Hao pipeline (for the agreement check)", case)
        hao_full = run_hao(seg_full, ct_full, img)
    del seg_full

    log.info("%s: parts 2-4 - cleanup, specks, checks", case)
    sl, aff = _spine_crop(pro_full, img.affine)
    zooms = tuple(float(z) for z in nib.affines.voxel_sizes(img.affine))
    seg, ct = np.ascontiguousarray(pro_full[sl]), np.ascontiguousarray(ct_full[sl])
    del ct_full
    rep = {"case": case,
           "pro": {k: qa[k] for k in ("audit_before", "audit_after", "envelope",
                                      "imbrication_upv_cm3")},
           "cleanup": {}, "specks": {}, "hao_check": {}, "anatomy_check": {}}
    clean = cleanup(seg, ct, zooms, rep["cleanup"])
    if specks:
        clean = remove_specks(clean, zooms, rep["specks"])
    review = np.zeros(seg.shape, np.uint8)
    if hao_full is not None:
        review = np.maximum(review, hao_agreement(clean, np.ascontiguousarray(hao_full[sl]),
                                                  ct, aff, rep["hao_check"]))
    r2, lm = anatomy_check(clean, ct, aff, rep["anatomy_check"])
    review = np.maximum(review, r2)

    final = pro_full.copy()
    final[sl] = clean
    save_case(out_dir, case, final, img)
    d = out_dir / case
    for name, vol in (("review", review), ("landmarks", lm)):
        full = np.zeros(final.shape, np.uint8)
        full[sl] = vol
        nib.save(nib.Nifti1Image(full, img.affine), str(d / f"{name}.nii.gz"))
    rep["seconds"] = round(time.time() - t0, 1)
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=_js))
    (d / "pro_qa.json").write_text(json.dumps(qa, indent=1, default=float))

    p, c, s, h, a = rep["pro"], rep["cleanup"], rep["specks"], rep["hao_check"], rep["anatomy_check"]
    print(f"\n{case}  ({rep['seconds']:.0f} s)")
    print(f"  Pro audit: {p['audit_before']} -> {p['audit_after']}")
    print(f"  cleanup: {c['result']} | stray pieces moved: {len(c['stray_pieces_moved'])} | "
          f"filled: {sum(c['filled_mm3'].values()):.0f} mm3")
    if s:
        print(f"  specks: recoloured {len(s['recoloured'])}, deleted {s['deleted_pieces']} "
              f"({sum(s['deleted_mm3'].values()):.0f} mm3) | vertebrae in >1 piece: "
              f"{len(s['loose_pieces_before'])} -> {len(s['loose_pieces_after'])}")
    if h:
        print(f"  Hao agreement (bodies L2-T5): agree {h['agree']} | disagree {h['disagree']}")
    print(f"  anatomy check: {a.get('verdict')}"
          + ("" if not a.get("anchors") else " | " + "; ".join(x["rule"] for x in a["anchors"])))
    for w in a.get("wrong_bodies", []) + a.get("why", []):
        print("    ", w)
    for w in a.get("notes", []):
        print("     note:", w)
    return rep


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Repair SuPreM vertebra labels using the CT.")
    ap.add_argument("--pred_dir", required=True, type=Path,
                    help="SuPreM vertebra predictions: <case>/segmentations or combined_labels")
    ap.add_argument("--ct_root", required=True, type=Path, help="holds <case>/ct.nii.gz")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--case", action="append", help="case id(s); default: all")
    ap.add_argument("--no_hao_check", action="store_true",
                    help="skip the Hao agreement check (saves the time of a second pipeline)")
    ap.add_argument("--no_specks", action="store_true", help="skip part 3 (speck removal)")
    a = ap.parse_args()
    cases = sorted(p for p in a.pred_dir.iterdir()
                   if (p / "combined_labels.nii.gz").exists() or (p / "segmentations").is_dir())
    if a.case:
        want = {c for v in a.case for c in v.split(",")}
        cases = [c for c in cases if c.name in want]
    for c in cases:
        process_case(c, a.ct_root, a.out_dir, hao_check=not a.no_hao_check,
                     specks=not a.no_specks)


if __name__ == "__main__":
    main()
