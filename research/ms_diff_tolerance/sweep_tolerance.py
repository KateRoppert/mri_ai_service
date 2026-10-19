"""How much does dilation_voxels decide the MS lesion diff verdicts?

Runs backend.lesion_diff.compare_labeled_masks over a tolerance sweep on every
already-processed MS run that has at least two sessions. Reads existing outputs
only — no GPU, no re-processing.
"""
import sys, pathlib, itertools, json
from collections import defaultdict

REPO = pathlib.Path("/home/ubuntu/mri_ai_service")
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import nibabel as nib
import yaml
from scipy import ndimage
from lesion_diff import compare_labeled_masks, MIN_LESION_VOLUME_MM3

CFG = yaml.safe_load(open(REPO / "configs" / "ms_longitudinal_config.yaml"))
BASE_KW = {
    "growth_threshold_relative": CFG["growth_threshold_relative"],
    "growth_threshold_absolute_cm3": CFG["growth_threshold_absolute_cm3"],
}
PRODUCTION_DILATION = CFG.get("dilation_voxels", 3)
SWEEP = [1, 2, 3, 4, 5, 6]


def labeled_masks():
    """{(run, subject): [(session, path), ...]} for MS sessions, ordered."""
    found = defaultdict(list)
    for run_dir in sorted((REPO / "demo_workspace" / "input").iterdir()):
        seg = run_dir / "segmentation"
        if not seg.is_dir():
            continue
        for p in sorted(seg.rglob("*_segmask_labels.nii.gz")):
            if "multiple_sclerosis" not in str(p):
                continue
            parts = p.relative_to(seg).parts
            subject, session = parts[0], parts[1]
            found[(run_dir.name, subject)].append((session, p))
    return {k: sorted(v) for k, v in found.items() if len(v) >= 2}


def nearest_distance_mm(src_labeled, dst_labeled, zoom, label):
    dt = ndimage.distance_transform_edt(dst_labeled == 0, sampling=zoom)
    return float(dt[src_labeled == label].min())


def kept(labeled, voxel_mm3):
    """Zero out components below the reported volume floor."""
    out = np.zeros_like(labeled)
    for lbl in np.unique(labeled):
        if lbl and (labeled == lbl).sum() * voxel_mm3 >= MIN_LESION_VOLUME_MM3:
            out[labeled == lbl] = lbl
    return out


def main():
    groups = labeled_masks()
    print(f"Пар сессий к разбору: {sum(len(v) - 1 for v in groups.values())} "
          f"в {len(groups)} сериях\n")

    rows = []
    for (run, subject), sessions in groups.items():
        for (s_prev, p_prev), (s_curr, p_curr) in zip(sessions, sessions[1:]):
            i_prev, i_curr = nib.load(p_prev), nib.load(p_curr)
            if i_prev.shape != i_curr.shape:
                print(f"  ПРОПУСК {run}/{subject} {s_prev}->{s_curr}: "
                      f"формы различаются {i_prev.shape} vs {i_curr.shape}")
                continue
            zoom = i_prev.header.get_zooms()[:3]
            vmm3 = float(np.prod(zoom))
            a = kept(np.asarray(i_prev.dataobj), vmm3)
            b = kept(np.asarray(i_curr.dataobj), vmm3)

            per_dil = {}
            for d in SWEEP:
                r = compare_labeled_masks(p_prev, p_curr, dilation_voxels=d, **BASE_KW)
                per_dil[d] = (r["new_count"], r["growing_count"],
                              r["stable_count"], r["resolved_count"])

            # Which lesions sit near the decision boundary: distance to the
            # other session's nearest tissue, for lesions that are reported.
            dists_new, dists_res = [], []
            dt_b = ndimage.distance_transform_edt(b == 0, sampling=zoom)
            dt_a = ndimage.distance_transform_edt(a == 0, sampling=zoom)
            for lbl in np.unique(a):
                if lbl:
                    dists_res.append((float(dt_b[a == lbl].min()),
                                      (a == lbl).sum() * vmm3))
            for lbl in np.unique(b):
                if lbl:
                    dists_new.append((float(dt_a[b == lbl].min()),
                                      (b == lbl).sum() * vmm3))

            borderline = [(d, v) for d, v in dists_new + dists_res
                          if min(SWEEP) < d <= max(SWEEP)]
            rows.append({
                "run": run, "subject": subject,
                "pair": f"{s_prev}->{s_curr}",
                "per_dilation": per_dil,
                "lesions_prev": int(len([x for x in np.unique(a) if x])),
                "lesions_curr": int(len([x for x in np.unique(b) if x])),
                "borderline": sorted(borderline, reverse=True),
            })

    print(f"{'серия / пара':38} {'очаги':>9}  " +
          "  ".join(f"d={d}" for d in SWEEP))
    unstable = 0
    for r in rows:
        counts = r["per_dilation"]
        cells = []
        for d in SWEEP:
            n, g, s, res = counts[d]
            cells.append(f"{n}/{res}")
        verdicts = {counts[d][0] + counts[d][3] for d in SWEEP}
        flag = ""
        if len(verdicts) > 1:
            unstable += 1
            flag = "  ← меняется"
        print(f"{r['run']+'/'+r['pair']:38} "
              f"{r['lesions_prev']}→{r['lesions_curr']:<5} " +
              "  ".join(f"{c:5}" for c in cells) + flag)

    print(f"\nв клетках: новых/разрешившихся")
    print(f"пар, где итог зависит от допуска: {unstable} из {len(rows)}")

    print(f"\n── очаги в зоне решения (расстояние {min(SWEEP)}–{max(SWEEP)} мм) ──")
    tot_border = 0
    for r in rows:
        if not r["borderline"]:
            continue
        tot_border += len(r["borderline"])
        near = ", ".join(f"{v:.0f}мм³@{d:.2f}мм" for d, v in r["borderline"][:6])
        print(f"  {r['run']}/{r['pair']}: {len(r['borderline'])} — {near}")
    print(f"\nвсего пограничных очагов: {tot_border}")
    print(f"боевой допуск: dilation_voxels = {PRODUCTION_DILATION}")

    out = pathlib.Path(__file__).parent / "sweep_result.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nсырые данные: {out}")


if __name__ == "__main__":
    main()
