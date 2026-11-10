#!/usr/bin/env python3
"""Меняет ли число потоков ANTs/ITK сам результат (пункт 7).

Зачем. В матрице стратегии получают разное число воркеров, а число потоков на
воркер выводится из него: `threads_per_worker = effective_cpu // actual_workers`
(`scripts/05_preprocessing.py`). Если результат ANTs зависит от числа потоков —
порядок суммирования с плавающей точкой при распараллеливании свёрток и
метрик меняется, — то стратегии сравниваются по makespan **разной работы**, и
сравнение недействительно. Это надо либо исключить, либо раскрыть в статье.

Проба мерит механизм, а не прокси: тот же вход, те же параметры, разное число
потоков, сравнение по данным. Целый этап для этого гонять не нужно.

Каждое число потоков исполняется в **своём процессе**: ITK читает
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` при инициализации пула, то есть внутри
одного процесса переменную менять поздно. Переменные выставляются ровно те же,
что в этапе 05.

Сравнивается не файл, а воксели и аффинная матрица: gzip пишет в заголовок
время, и одинаковые по смыслу выходы дали бы разные хеши файла.

    python research/mas_resource_negotiation/thread_determinism.py \\
        --input <t1.nii.gz> --threads 1 4 8 --op n4
    ... --op register --moving <t2.nii.gz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _digest(nifti_path: Path) -> dict:
    """Хеш по данным и геометрии, не по сжатому файлу."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(nifti_path))
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    return {
        "voxels_sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest(),
        "affine_sha256": hashlib.sha256(
            np.ascontiguousarray(img.affine, dtype=np.float64).tobytes()).hexdigest(),
        "shape": list(data.shape),
        "sum": float(data.sum()),
        "max": float(data.max()),
    }


def _compare(a: Path, b: Path) -> dict:
    """Насколько расходятся два выхода, если расходятся."""
    import nibabel as nib
    import numpy as np

    x = np.asanyarray(nib.load(str(a)).dataobj, dtype=np.float64)
    y = np.asanyarray(nib.load(str(b)).dataobj, dtype=np.float64)
    if x.shape != y.shape:
        return {"same_shape": False}
    diff = np.abs(x - y)
    scale = float(np.abs(x).max()) or 1.0
    nonzero = diff[diff > 0]
    # Максимум сам по себе обманчив: сдвиг трансформации на доли вокселя даёт
    # на резкой границе разницу во весь динамический диапазон. Решает
    # распределение, поэтому считаем и его.
    return {
        "same_shape": True,
        "differing_voxels": int(nonzero.size),
        "total_voxels": int(diff.size),
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float(diff.max() / scale),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff_where_differing": (float(np.median(nonzero))
                                            if nonzero.size else 0.0),
        "p99_abs_diff_where_differing": (float(np.percentile(nonzero, 99))
                                         if nonzero.size else 0.0),
        "mean_rel_diff": float(diff.mean() / scale),
    }


# ── режим ребёнка: один прогон с фиксированным числом потоков ──────────────


def _run_worker(args) -> int:
    # Ровно те переменные, что выставляет воркер этапа 05, и в том же порядке.
    threads = str(args.worker_threads)
    os.environ["ANTS_RANDOM_SEED"] = "42"
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
    os.environ["ANTS_NUMBER_OF_THREADS"] = threads

    import ants  # импорт ПОСЛЕ переменных: ITK читает их при инициализации

    fixed = ants.image_read(str(args.input))
    started = time.time()
    if args.op == "n4":
        out = ants.n4_bias_field_correction(fixed)
    elif args.op == "register":
        moving = ants.image_read(str(args.moving))
        extra = {} if args.seed is None else {"random_seed": int(args.seed)}
        result = ants.registration(fixed=fixed, moving=moving,
                                   type_of_transform=args.transform, **extra)
        out = result["warpedmovout"]
    else:
        raise SystemExit(f"неизвестная операция: {args.op}")
    elapsed = time.time() - started

    ants.image_write(out, str(args.worker_output))
    print(json.dumps({"threads": args.worker_threads, "seconds": round(elapsed, 2)}))
    return 0


# ── режим родителя ────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--moving", type=Path, help="второй том для --op register")
    parser.add_argument("--op", choices=["n4", "register"], default="n4")
    parser.add_argument("--transform", default="Affine")
    parser.add_argument("--seed", type=int, default=None,
                        help="передать random_seed в ants.registration; "
                             "ANTS_RANDOM_SEED в окружении antspyx НЕ читает")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--keep", type=Path, help="куда сложить выходы")
    parser.add_argument("--json", action="store_true")
    # служебное
    parser.add_argument("--worker-threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_threads is not None:
        return _run_worker(args)

    if args.op == "register" and not args.moving:
        parser.error("--op register требует --moving")

    workdir = args.keep or Path(tempfile.mkdtemp(prefix="thread_determinism_"))
    workdir.mkdir(parents=True, exist_ok=True)

    runs = []
    # Индекс в имени позволяет повторить одно и то же число потоков
    # (`--threads 4 4 4`) — это контроль на воспроизводимость самой операции,
    # без которого вывод «виноваты потоки» ничем не обоснован.
    for idx, threads in enumerate(args.threads, 1):
        out = workdir / f"{args.op}_t{threads}_r{idx}.nii.gz"
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--input", str(args.input), "--op", args.op,
               "--transform", args.transform,
               "--worker-threads", str(threads), "--worker-output", str(out)]
        if args.moving:
            cmd += ["--moving", str(args.moving)]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
            return 1
        seconds = json.loads(proc.stdout.strip().splitlines()[-1])["seconds"]
        runs.append({"threads": threads, "output": out, "seconds": seconds,
                     **_digest(out)})

    reference = runs[0]
    verdict = {
        "op": args.op,
        "input": str(args.input),
        "threads_tested": args.threads,
        "identical": all(r["voxels_sha256"] == reference["voxels_sha256"]
                         for r in runs),
        "runs": [{k: v for k, v in r.items() if k != "output"} for r in runs],
    }
    if not verdict["identical"]:
        verdict["differences"] = [
            {"threads": r["threads"], **_compare(reference["output"], r["output"])}
            for r in runs[1:] if r["voxels_sha256"] != reference["voxels_sha256"]
        ]

    if args.json:
        print(json.dumps(verdict, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"\noперация: {args.op}   вход: {Path(args.input).name}")
        for r in runs:
            print(f"  потоков {r['threads']:3d}: {r['seconds']:7.2f} с   "
                  f"воксели {r['voxels_sha256'][:16]}  сумма {r['sum']:.6e}")
        same_threads = len(set(args.threads)) == 1
        what = ("повторные прогоны при одном числе потоков"
                if same_threads else "прогоны с разным числом потоков")
        if verdict["identical"]:
            print(f"\n  ВЫВОД: {what} дают побитово одинаковый результат")
        elif same_threads:
            print("\n  ВЫВОД: результат НЕ ВОСПРОИЗВОДИМ сам по себе — "
                  "число потоков тут не при чём")
        else:
            print("\n  ВЫВОД: результаты различаются. Чтобы приписать это "
                  "потокам, нужен контроль с одинаковым их числом "
                  "(`--threads N N N`)")
            for d in verdict["differences"]:
                print(f"    потоков {d['threads']}: различается вокселей "
                      f"{d['differing_voxels']}/{d['total_voxels']}"
                      f" ({100 * d['differing_voxels'] / d['total_voxels']:.0f}%)")
                print(f"      отличие: среднее {d['mean_abs_diff']:.3e} "
                      f"({d['mean_rel_diff']:.1e} от макс.) | медиана по "
                      f"различающимся {d['median_abs_diff_where_differing']:.3e} | "
                      f"p99 {d['p99_abs_diff_where_differing']:.3e} | "
                      f"макс {d['max_abs_diff']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
