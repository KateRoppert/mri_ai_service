#!/usr/bin/env python3
"""
Analyse the skull-strip-order experiment: after (today) vs before (exp).

Sessions are matched across runs by (set, original patient id, scan date) —
sub-NNN numbers are issued afresh in every run. Writes per-session rows to
<exp>/analysis/sessions.csv and per-patient inter-session consistency to
<exp>/analysis/patients.csv, and prints a summary.

Run inside brain-lesion-web:demo (ants, nibabel) with the experiment folder
mounted at /e. Spec: docs/superpowers/specs/2026-10-07-skull-strip-before-registration-design.md
"""
import csv
import itertools
import json
import os
import re
from pathlib import Path

import ants
import numpy as np

EXP = Path(os.environ.get("EXP", "/e"))
ORDERS, RUNS = ("after", "before"), ("run1", "run2")
SETS = {"ms_sibbms": "multiple_sclerosis", "ms_ms5": "multiple_sclerosis",
        "gbm_dropbox33": "glioblastoma", "gbm_bo": "glioblastoma", "gbm_upenn": "glioblastoma"}
STD = os.environ.get("FSLDIR", "/usr/local/fsl") + "/data/standard"
TEMPLATE = ants.image_read(f"{STD}/MNI152_T1_1mm_brain.nii.gz")
TMASK = ants.image_read(f"{STD}/MNI152_T1_1mm_brain_mask.nii.gz").numpy() > 0
BOX = [np.array(p, float) for p in itertools.product((-90, 90), (-110, 110), (-80, 80))] + [np.zeros(3)]


def sessions(run_dir: Path):
    """{(orig_id, date): (sub, ses)} from Stage 01's mapping."""
    mfile = run_dir / "bids_organized" / "dataset_mapping.json"
    if not mfile.exists():
        return {}
    data = json.loads(mfile.read_text())
    out = {}
    for sub, p in data.get("patients", {}).items():
        for ses, s in p.get("sessions", {}).items():
            if s.get("status") == "complete":
                out[(p.get("original_id"), s.get("original_date"))] = (sub, ses)
    return out


def files(run_dir, sub, ses, lesion):
    a = f"{sub}/{ses}/anat"
    seg = run_dir / "segmentation" / a / lesion
    return {
        "t1": run_dir / "preprocessed" / a / f"{sub}_{ses}_t1.nii.gz",
        "tx": run_dir / "transformations" / a / f"{sub}_{ses}_t1_to_atlas.mat",
        "brain": run_dir / "transformations" / a / f"{sub}_{ses}_brain_mask.nii.gz",
        "lesion_native": seg / f"{sub}_{ses}_t1_segmask_native_t1.nii.gz",
        "volume": seg / f"{sub}_{ses}_t1_volume_report.json",
        "lobar": seg / f"{sub}_{ses}_t1_lobar_report.json",
        "mcdonald": seg / f"{sub}_{ses}_t1_mcdonald_report.json",
    }


def stage05_times(run_dir):
    log = run_dir / "logs" / "stages" / "stage_05_preprocessing.log"
    if not log.exists():
        return {}
    t = {}
    for m in re.finditer(r"Completed (sub-\d+)/(ses-\d+) in ([\d.]+)s", log.read_text(errors="ignore")):
        t[(m.group(1), m.group(2))] = float(m.group(3))
    return t


def reg_quality(t1_path):
    img = ants.image_read(str(t1_path))
    v = img.numpy()
    return (ants.image_mutual_information(TEMPLATE, img),
            float(np.corrcoef(TEMPLATE.numpy()[TMASK], v[TMASK])[0, 1]))


def spread(tx_a, tx_b):
    a, b = ants.read_transform(str(tx_a)), ants.read_transform(str(tx_b))
    return max(np.linalg.norm(np.array(a.apply_to_point(tuple(p))) - np.array(b.apply_to_point(tuple(p))))
               for p in BOX)


def dice(pa, pb):
    a = ants.image_read(str(pa)).numpy() > 0
    b = ants.image_read(str(pb)).numpy() > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def lesion_ml(vol_path):
    d = json.loads(vol_path.read_text())
    return sum(c.get("volume_cm3", 0) for c in d.get("classes", {}).values())


def distribution(path, kind):
    d = json.loads(path.read_text())
    if kind == "lobar":
        parts = {k: v.get("total_volume_cm3", 0) for k, v in d.get("lobes", {}).items()}
    else:
        parts = {k: v.get("total_volume_cm3", 0) for k, v in d.get("zones", {}).items()}
    tot = sum(parts.values())
    return {k: v / tot for k, v in parts.items()} if tot else {}


def shift(da, db):
    """Share of lesion volume that sits in a different region (half L1)."""
    keys = set(da) | set(db)
    return 0.5 * sum(abs(da.get(k, 0) - db.get(k, 0)) for k in keys) if keys else float("nan")


rows, patients = [], []
for set_name, lesion in SETS.items():
    runs = {(o, r): EXP / o / r / set_name for o in ORDERS for r in RUNS}
    maps = {k: sessions(v) for k, v in runs.items()}
    times = {k: stage05_times(v) for k, v in runs.items()}
    keys = sorted(set().union(*[set(m) for m in maps.values()]))
    per_patient = {}
    for key in keys:
        row = {"set": set_name, "patient": key[0], "date": key[1]}
        f = {}
        for (o, r), run_dir in runs.items():
            if key in maps[(o, r)]:
                sub, ses = maps[(o, r)][key]
                f[(o, r)] = files(run_dir, sub, ses, lesion)
                row[f"t05_{o}_{r}"] = times[(o, r)].get((sub, ses))
        for (o, r), fi in f.items():
            if fi["t1"].exists():
                row[f"mi_{o}_{r}"], row[f"corr_{o}_{r}"] = reg_quality(fi["t1"])
                per_patient.setdefault((key[0], o, r), []).append(fi["t1"])
            if fi["brain"].exists():
                row[f"brain_ml_{o}_{r}"] = float((ants.image_read(str(fi["brain"])).numpy() > 0).sum()) / 1000
            if fi["volume"].exists():
                row[f"lesion_ml_{o}_{r}"] = lesion_ml(fi["volume"])
        for o in ORDERS:
            a, b = f.get((o, "run1")), f.get((o, "run2"))
            if a and b and a["tx"].exists() and b["tx"].exists():
                row[f"spread_{o}"] = spread(a["tx"], b["tx"])
            if a and b and a["lesion_native"].exists() and b["lesion_native"].exists():
                row[f"dice_repeat_{o}"] = dice(a["lesion_native"], b["lesion_native"])
        a, b = f.get(("after", "run1")), f.get(("before", "run1"))
        if a and b:
            if a["lesion_native"].exists() and b["lesion_native"].exists():
                row["dice_after_vs_before"] = dice(a["lesion_native"], b["lesion_native"])
            kind = "mcdonald" if lesion == "multiple_sclerosis" else "lobar"
            if a[kind].exists() and b[kind].exists():
                row[f"{kind}_shift_after_vs_before"] = shift(distribution(a[kind], kind),
                                                            distribution(b[kind], kind))
            for o in ORDERS:
                x, y = f.get((o, "run1")), f.get((o, "run2"))
                if x and y and x[kind].exists() and y[kind].exists():
                    row[f"{kind}_shift_repeat_{o}"] = shift(distribution(x[kind], kind),
                                                           distribution(y[kind], kind))
        rows.append(row)
        print(f"{set_name} {key[0]} {key[1]}", flush=True)

    # same patient, several sessions: do they land on each other in atlas space?
    for (pat, o, r), t1s in per_patient.items():
        if len(t1s) < 2:
            continue
        vols = [ants.image_read(str(p)).numpy()[TMASK] for p in t1s]
        cs = [np.corrcoef(x, y)[0, 1] for x, y in itertools.combinations(vols, 2)]
        patients.append({"set": set_name, "patient": pat, "order": o, "run": r,
                         "sessions": len(t1s), "mean_pairwise_corr": float(np.mean(cs)),
                         "min_pairwise_corr": float(np.min(cs))})

out = EXP / "analysis"
out.mkdir(exist_ok=True)
fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("set", "patient", "date"), k))
with open(out / "sessions.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
with open(out / "patients.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(patients[0]) if patients else ["set"])
    w.writeheader()
    w.writerows(patients)
print(f"wrote {len(rows)} sessions, {len(patients)} patient rows to {out}")
