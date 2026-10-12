"""score_vertebrae.py - score vertebra predictions against an ideal (reference) mask.

    python score_vertebrae.py --ideal ../Final02 \
        --pred raw=../AbdomenAtlasDemoPredict --pred pro=../Refined_Pro \
        --pred hao=../Refined_Hao --out_dir ../scores [--case BDMAP_00000031]

Each folder holds <case>/combined_labels.nii.gz (or <case>/segmentations/vertebrae_*.nii.gz)
in the 1=L5 ... 5=L1, 6=T12 ... 17=T1, 18=C7 ... 24=C1 coding, on the same grid as the ideal.

Per vertebra (every label present in the ideal):
  dice        overlap with the ideal vertebra, 0..1 (1 = identical)
  hd95_mm     95th percentile of the surface-to-surface distance, both ways (lower = better)
  assd_mm     average surface-to-surface distance, both ways (lower = better)
  vol_diff_%  predicted volume vs ideal volume
  pieces      face-connected pieces of the predicted vertebra; extra_pieces = how many more
              than the ideal vertebra has (0 = clean, more = specks / breaks)
  named_as    the label most of the ideal vertebra's voxels carry in the prediction
              (= the vertebra itself when the naming is right)

Per case:
  mean_dice, mean_hd95_mm, mean_assd_mm, spine_dice (all vertebrae as one mask, names
  ignored), naming_rate (share of vertebrae named right), clean_rate (share with no extra
  pieces compared with the ideal),
  extra_labels (labels in the prediction that the ideal does not have), and

  SCORE (0..100) = 100 * ( 0.50 * mean_dice
                         + 0.20 * naming_rate
                         + 0.20 * mean surface score   [per vertebra: 1 - hd95 / 10 mm, clipped 0..1]
                         + 0.10 * clean_rate )

  The weights are in WEIGHTS below. A vertebra missing from the prediction scores dice 0,
  surface score 0 and counts as misnamed.

Outputs in --out_dir: per_vertebra.csv, per_case.csv, and a leaderboard printed on screen.
numpy, scipy, nibabel only.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

NAMES = ([""] + ["L5", "L4", "L3", "L2", "L1"] + [f"T{i}" for i in range(12, 0, -1)]
         + [f"C{i}" for i in range(7, 0, -1)])            # NAMES[1] = "L5" ... NAMES[24] = "C1"
ID = {n: i for i, n in enumerate(NAMES) if n}
S6 = ndimage.generate_binary_structure(3, 1)
WEIGHTS = {"dice": 0.50, "naming": 0.20, "surface": 0.20, "clean": 0.10}
HD95_ZERO_MM = 10.0          # an HD95 of this many mm (or more) gives a surface score of 0


#  I/O 
def load_labels(case_dir: Path):
    """combined_labels.nii.gz, or the per-vertebra masks if there is no combined file."""
    f = case_dir / "combined_labels.nii.gz"
    if f.exists():
        img = nib.load(str(f))
        seg = np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)
    else:
        files = sorted((case_dir / "segmentations").glob("vertebrae_*.nii.gz"))
        if not files:
            raise FileNotFoundError(f"{case_dir}: no combined_labels.nii.gz or segmentations/")
        img = nib.load(str(files[0]))
        seg = np.zeros(img.shape, np.uint8)
        for p in files:
            name = p.name[len("vertebrae_"):-len(".nii.gz")]
            if name in ID:
                seg[np.asanyarray(nib.load(str(p)).dataobj) > 0] = ID[name]
    if seg.max() > 24:
        raise ValueError(f"{case_dir}: label values up to {seg.max()} - not the 1..24 vertebra coding")
    return img, seg


# metrics 
def _surface(m):
    return m & ~ndimage.binary_erosion(m, structure=S6, border_value=0)


def surface_distances(a, b, zooms):
    """Both-way distances (mm) between the surfaces of masks a and b (same crop)."""
    sa, sb = _surface(a), _surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=zooms)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=zooms)[sb]
    return np.concatenate([da, db])


def score_case(ideal, pred, zooms):
    """Per-vertebra rows + case summary for one case."""
    vox = float(np.prod(zooms))
    lab_i = [l for l in range(1, 25) if (ideal == l).any()]
    lab_p = set(np.unique(pred).tolist()) - {0}
    oi, op = ndimage.find_objects(ideal), ndimage.find_objects(pred)
    rows = []
    for l in lab_i:
        a_box = oi[l - 1]
        b_box = op[l - 1] if l <= len(op) and op[l - 1] is not None else None
        box = a_box if b_box is None else tuple(slice(min(x.start, y.start), max(x.stop, y.stop))
                                                for x, y in zip(a_box, b_box))
        box = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, n)) for s, n in zip(box, ideal.shape))
        a, b = ideal[box] == l, pred[box] == l
        na, nb = int(a.sum()), int(b.sum())
        dice = 2.0 * int((a & b).sum()) / (na + nb)
        if nb:
            d = surface_distances(a, b, zooms)
            hd95, assd = float(np.percentile(d, 95)), float(d.mean())
            pieces = int(ndimage.label(b, structure=S6)[1])
        else:
            hd95 = assd = float("nan")
            pieces = 0
        extra = max(pieces - int(ndimage.label(a, structure=S6)[1]), 0) if nb else 1
        votes = np.bincount(pred[box][a], minlength=25)
        votes[0] = 0
        named = int(votes.argmax()) if votes.any() else 0
        rows.append({"vertebra": NAMES[l], "dice": round(dice, 4),
                     "hd95_mm": round(hd95, 2), "assd_mm": round(assd, 3),
                     "vol_diff_%": round(100.0 * (nb - na) / na, 2),
                     "ideal_cm3": round(na * vox / 1e3, 2), "pred_cm3": round(nb * vox / 1e3, 2),
                     "pieces": pieces, "extra_pieces": extra,
                     "named_as": NAMES[named] if named else "-"})
    n = len(rows)
    dices = np.array([r["dice"] for r in rows])
    hd = np.array([r["hd95_mm"] for r in rows])
    surf = np.where(np.isnan(hd), 0.0, np.clip(1.0 - hd / HD95_ZERO_MM, 0.0, 1.0))
    naming = sum(r["named_as"] == r["vertebra"] for r in rows) / n
    clean = sum(r["extra_pieces"] == 0 for r in rows) / n
    a, b = ideal > 0, pred > 0
    summary = {"vertebrae": n, "mean_dice": round(float(dices.mean()), 4),
               "min_dice": round(float(dices.min()), 4),
               "worst": rows[int(dices.argmin())]["vertebra"],
               "mean_hd95_mm": round(float(np.nanmean(hd)), 2) if np.isfinite(hd).any() else float("nan"),
               "mean_assd_mm": round(float(np.nanmean([r["assd_mm"] for r in rows])), 3),
               "spine_dice": round(2.0 * int((a & b).sum()) / (int(a.sum()) + int(b.sum())), 4),
               "naming_rate": round(naming, 4), "clean_rate": round(clean, 4),
               "missing": [r["vertebra"] for r in rows if r["pred_cm3"] == 0],
               "extra_labels": [NAMES[l] for l in sorted(lab_p - set(lab_i))]}
    summary["score"] = round(100.0 * (WEIGHTS["dice"] * dices.mean() + WEIGHTS["naming"] * naming
                                      + WEIGHTS["surface"] * surf.mean()
                                      + WEIGHTS["clean"] * clean), 2)
    return rows, summary


# driver 
def main():
    ap = argparse.ArgumentParser(description="Score vertebra predictions against an ideal mask.")
    ap.add_argument("--ideal", required=True, type=Path, help="folder with the ideal <case>/ masks")
    ap.add_argument("--pred", required=True, action="append",
                    help="name=folder of a prediction to score (repeat for several)")
    ap.add_argument("--out_dir", type=Path, default=Path("scores"))
    ap.add_argument("--case", action="append", help="case id(s); default: every case in --ideal")
    a = ap.parse_args()
    preds = []
    for p in a.pred:
        name, _, path = p.partition("=")
        preds.append((name, Path(path)) if path else (Path(name).name, Path(name)))
    cases = sorted(c.name for c in a.ideal.iterdir() if c.is_dir())
    if a.case:
        want = {c for v in a.case for c in v.split(",")}
        cases = [c for c in cases if c in want]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    per_v, per_c = [], []
    for case in cases:
        img_i, ideal = load_labels(a.ideal / case)
        zooms = tuple(float(z) for z in nib.affines.voxel_sizes(img_i.affine))
        print(f"\n{case}  (voxel {zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f} mm)")
        print(f"  {'method':<16}{'SCORE':>7}{'mean Dice':>11}{'min Dice':>15}{'HD95 mm':>9}"
              f"{'ASSD mm':>9}{'spine Dice':>12}{'naming':>8}{'clean':>7}")
        for name, folder in preds:
            if not (folder / case).is_dir():
                print(f"  {name:<16}  (no {case} in {folder})")
                continue
            img_p, pred = load_labels(folder / case)
            if pred.shape != ideal.shape or not np.allclose(img_p.affine, img_i.affine, atol=1e-3):
                print(f"  {name:<16}  (different grid from the ideal - skipped)")
                continue
            rows, s = score_case(ideal, pred, zooms)
            for r in rows:
                per_v.append({"method": name, "case": case, **r})
            per_c.append({"method": name, "case": case,
                          **{k: (";".join(v) if isinstance(v, list) else v) for k, v in s.items()}})
            print(f"  {name:<16}{s['score']:>7.1f}{s['mean_dice']:>11.4f}"
                  f"{s['min_dice']:>9.4f} ({s['worst']:>3}){s['mean_hd95_mm']:>9.2f}"
                  f"{s['mean_assd_mm']:>9.3f}{s['spine_dice']:>12.4f}"
                  f"{s['naming_rate']:>8.2f}{s['clean_rate']:>7.2f}"
                  + (f"   missing: {','.join(s['missing'])}" if s["missing"] else "")
                  + (f"   extra: {','.join(s['extra_labels'])}" if s["extra_labels"] else ""))
    for fname, rows in (("per_vertebra.csv", per_v), ("per_case.csv", per_c)):
        if rows:
            with open(a.out_dir / fname, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\nwritten: {a.out_dir / 'per_vertebra.csv'}, {a.out_dir / 'per_case.csv'}")
    print(f"SCORE = 100 x ({WEIGHTS['dice']} mean Dice + {WEIGHTS['naming']} naming + "
          f"{WEIGHTS['surface']} surface [1 - HD95/{HD95_ZERO_MM:g} mm] + {WEIGHTS['clean']} clean)")


if __name__ == "__main__":
    main()
