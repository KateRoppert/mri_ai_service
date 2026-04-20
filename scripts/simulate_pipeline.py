"""
Pipeline simulation: 4 modes comparison.
  1. Sequential         — 1 process, stages one after another
  2. Stage-Sequential   — all CPUs per stage, stages one after another (current system)
  3. Pipeline Parallel  — fixed CPU split, stages overlap
  4. MAS Dynamic        — pipeline with CNP-inspired reallocation

Uses empirical throughput data. GPU stage always uses TOTAL_GPU.

Usage: python simulate_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# EMPIRICAL DATA
# ============================================================
EMP = {
    1: {"name":"BIDS Org.", "n":110, "sat":8,
        "t":{1:658.4,2:338.4,4:211.5,6:190.8,8:205.2,10:192.2,12:188.6,14:190.1,16:261.0,18:324.4,20:225.0}},
    2: {"name":"Metadata", "n":110, "sat":8,
        "t":{1:0.81,2:0.52,4:0.32,6:0.27,8:0.23,10:0.24,12:0.24,14:0.23,16:0.23,18:0.24,20:0.24}},
    3: {"name":"NIfTI Conv.", "n":110, "sat":8,
        "t":{1:134.5,2:72.3,4:45.1,6:38.5,8:35.5,10:35.8,12:35.8,14:36.0,16:35.5,18:36.2,20:36.0}},
    4: {"name":"Quality", "n":110, "sat":12,
        "t":{1:594.8,2:352.9,4:198.2,6:153.1,8:127.7,10:117.4,12:114.0,14:114.0,16:115.2,18:116.1,20:117.0}},
    5: {"name":"Preproc.", "n":103, "sat":6,
        "t":{1:3229.2,2:2139.7,4:1648.6,6:1572.6,8:1512.7,10:1506.7,12:1493.5,14:1493.8,16:1492.2,18:1490.3,20:1490.1}},
    6: {"name":"Segment.", "n":98, "sat":2,
        "gpu":{1:24.31, 2:12.55}},
}

CPU = 24; GPU = 2
BATCHES = [20, 50, 100]
SURV = {1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:103/110, 6:98/110}
DT = 0.5; REALLOC_INT = 10.0; REALLOC_OH = 1.0

np.random.seed(42)


def tp(s, w):
    """Throughput (sessions/sec) for stage s with w workers."""
    if w <= 0: return 0.0
    d = EMP[s]
    if s == 6: return 1.0 / d["gpu"][min(w, 2)]
    ts = d["t"]; n = d["n"]; ks = sorted(ts.keys())
    w = max(ks[0], min(w, ks[-1]))
    for i in range(len(ks)-1):
        if ks[i] <= w <= ks[i+1]:
            f = (w-ks[i]) / (ks[i+1]-ks[i])
            return n / (ts[ks[i]] + f*(ts[ks[i+1]] - ts[ks[i]]))
    return n / ts[ks[-1]]


def n_at(batch, s):
    return max(1, int(round(batch * SURV[s])))


def best_w(s, budget):
    """Best worker count for stage s within budget (minimize time)."""
    if s == 6: return min(budget, GPU)
    d = EMP[s]; ts = d["t"]; ks = sorted(ts.keys())
    best_k = 1; best_t = ts[1]
    for k in ks:
        if k <= budget and ts[k] < best_t:
            best_k = k; best_t = ts[k]
    return best_k


# ============================================================
# MODE 1: SEQUENTIAL
# ============================================================
def mode_sequential(batch):
    total = 0
    for s in range(1, 7):
        n = n_at(batch, s)
        total += n / tp(s, 1 if s < 6 else 1)
    return {"makespan": total, "cpu_util": 1/CPU*100}


# ============================================================
# MODE 2: STAGE-SEQUENTIAL PARALLEL (current real system)
# Each stage gets ALL CPUs, but stages run one after another.
# Uses optimal worker count (up to CPU pool) for each stage.
# ============================================================
def mode_stage_sequential(batch):
    total = 0
    stage_times = {}
    for s in range(1, 7):
        n = n_at(batch, s)
        if s == 6:
            w = GPU
        else:
            w = best_w(s, CPU)  # can use all 24 CPUs
        t = n / tp(s, w)
        stage_times[s] = t
        total += t

    # CPU utilization: only 1 stage active at a time
    # weighted by best_w / CPU
    weighted = sum(best_w(s, CPU) / CPU * stage_times[s] for s in range(1, 6))
    cpu_util = weighted / total * 100 if total > 0 else 0

    return {"makespan": total, "cpu_util": cpu_util, "stage_times": stage_times}


# ============================================================
# MODE 3: PIPELINE PARALLEL (stages overlap, fixed allocation)
# CPU budget split across all stages simultaneously.
# ============================================================
def pipe_alloc():
    """Allocate CPU proportionally to optimal, scaled to budget."""
    opt = {1:8, 2:8, 3:8, 4:12, 5:6}  # sum=42
    s = sum(opt.values())
    a = {k: max(1, round(v*CPU/s)) for k,v in opt.items()}
    diff = CPU - sum(a.values())
    for tgt in [5,4,3,1,2]:
        if diff == 0: break
        if diff > 0:
            add = min(diff, EMP[tgt]["sat"]-a[tgt]); a[tgt] += add; diff -= add
        else:
            sub = min(-diff, a[tgt]-1); a[tgt] -= sub; diff += sub
    a[6] = GPU
    return a


def mode_pipeline(batch):
    a = pipe_alloc()
    start = {}; end = {}
    for s in range(1, 7):
        n = n_at(batch, s)
        w = a[s]
        if s == 1:
            start[s] = 0
        else:
            start[s] = start[s-1] + 1/tp(s-1, a[s-1])
        end[s] = start[s] + n / tp(s, w)

    makespan = max(end.values())
    cpu_sec = sum(a[s] * (end[s]-start[s]) for s in range(1,6))
    cpu_util = cpu_sec / (CPU*makespan) * 100 if makespan > 0 else 0

    return {"makespan": makespan, "cpu_util": cpu_util, "alloc": a,
            "start": start, "end": end}


# ============================================================
# MODE 4: MAS DYNAMIC (discrete-event simulation)
# ============================================================
def mode_mas(batch):
    a = pipe_alloc()
    w = {s: a[s] for s in range(1,7)}
    q = {s: 0.0 for s in range(1,7)}
    q[1] = float(n_at(batch, 1))
    done = {s: 0 for s in range(1,7)}
    tgt = {s: n_at(batch, s) for s in range(1,7)}

    t = 0.0; nr = 0; next_r = REALLOC_INT
    cpu_log = []; alloc_log = []

    while t < 300000:
        if all(done[s] >= tgt[s] for s in range(1,7)): break

        # Reallocation
        if t >= next_r:
            next_r = t + REALLOC_INT
            pool = 0
            for s in range(1,6):
                sat = EMP[s]["sat"]
                if done[s] >= tgt[s]:
                    pool += w[s]; w[s] = 0
                elif q[s] < 0.5 and done[s] < tgt[s]:
                    rel = max(0, w[s]-1); pool += rel; w[s] -= rel
                elif w[s] > sat:
                    rel = w[s]-sat; pool += rel; w[s] -= rel
            if pool > 0:
                cands = [(s, q[s], EMP[s]["sat"]-w[s])
                         for s in range(1,6)
                         if tgt[s]-done[s] > 0 and q[s] >= 0.5 and w[s] < EMP[s]["sat"]]
                cands.sort(key=lambda x: x[1], reverse=True)
                for s, _, cap in cands:
                    if pool <= 0: break
                    g = min(cap, pool); w[s] += g; pool -= g
                for s in range(1,6):
                    if pool <= 0: break
                    if tgt[s]-done[s] > 0 and w[s] < EMP[s]["sat"]:
                        g = min(EMP[s]["sat"]-w[s], pool); w[s] += g; pool -= g
                nr += 1
                alloc_log.append({"time":round(t,1),
                    **{f"w{s}":w[s] for s in range(1,6)},
                    **{f"q{s}":int(q[s]) for s in range(1,6)}})

        # Process
        for s in range(1,7):
            rem = tgt[s] - done[s]
            if rem <= 0 or q[s] < 0.5 or w[s] <= 0: continue
            prod = min(tp(s, w[s]) * DT, q[s], rem)
            whole = int(prod)
            if np.random.random() < (prod-whole) and whole < rem and whole < q[s]:
                whole += 1
            whole = min(whole, int(q[s]), rem)
            q[s] -= whole; done[s] += whole
            if whole > 0 and s < 6:
                q[s+1] += whole

        act = sum(w[s] for s in range(1,6) if q[s] >= 0.5 and done[s] < tgt[s])
        cpu_log.append(act/CPU*100)
        t += DT

    mk = t + nr * REALLOC_OH
    return {"makespan": mk, "cpu_util": np.mean(cpu_log) if cpu_log else 0,
            "alloc_log": alloc_log, "n_reallocs": nr}


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs("simulation_results", exist_ok=True)
    pa = pipe_alloc()

    print("="*70)
    print("PIPELINE SIMULATION — 4 MODES")
    print("="*70)
    print(f"CPU: {CPU} | GPU: {GPU}")
    print(f"Pipeline alloc: {', '.join(f'S{s}={pa[s]}' for s in range(1,7))}")
    print()

    modes = ["Sequential", "Stage-Seq. Parallel", "Pipeline Parallel", "MAS (Dynamic)"]
    rows = []

    for bs in BATCHES:
        print(f"\n{'='*70}")
        print(f"BATCH: {bs} patients")
        print(f"{'='*70}")

        r1 = mode_sequential(bs)
        r2 = mode_stage_sequential(bs)
        r3 = mode_pipeline(bs)
        r4 = mode_mas(bs)

        seq_t = r1["makespan"]

        for label, r in zip(modes, [r1, r2, r3, r4]):
            mk = r["makespan"]
            sp = seq_t / mk
            thr = bs / mk * 3600
            print(f"\n  {label}:")
            print(f"    Makespan:   {mk:8.1f}s ({mk/60:.1f} min)")
            print(f"    CPU util:   {r['cpu_util']:7.1f}%")
            print(f"    Throughput: {thr:7.1f} pat/hr")
            if label != "Sequential":
                print(f"    Speedup:    {sp:7.2f}x vs sequential")

            rows.append({"batch": bs, "mode": label,
                         "makespan_s": round(mk,1), "makespan_min": round(mk/60,1),
                         "cpu_util": round(r["cpu_util"],1),
                         "thr_hr": round(thr,1), "speedup": round(sp,2)})

        # Comparisons
        print(f"\n  Comparison:")
        print(f"    Stage-Seq vs Pipeline: Stage-Seq {'faster' if r2['makespan']<r3['makespan'] else 'slower'} "
              f"({abs(r2['makespan']-r3['makespan']):.1f}s diff)")
        imp = (r3["makespan"]-r4["makespan"])/r3["makespan"]*100
        print(f"    MAS vs Pipeline:  {imp:+.1f}% makespan")
        imp2 = (r2["makespan"]-r4["makespan"])/r2["makespan"]*100
        print(f"    MAS vs Stage-Seq: {imp2:+.1f}% makespan")

    df = pd.DataFrame(rows)
    df.to_csv("simulation_results/simulation_comparison.csv", index=False)

    # ========================================================
    # FIGURE 1: Makespan bars — all 4 modes
    # ========================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    clr = {"Sequential":"#d62728", "Stage-Seq. Parallel":"#ff7f0e",
           "Pipeline Parallel":"#1f77b4", "MAS (Dynamic)":"#2ca02c"}

    for i, bs in enumerate(BATCHES):
        ax = axes[i]; bdf = df[df["batch"]==bs]
        bars = ax.bar(range(len(modes)), bdf["makespan_min"].values,
                      color=[clr[m] for m in bdf["mode"]], width=0.65)
        for b, v in zip(bars, bdf["makespan_min"]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                    f"{v:.1f}", ha="center", fontsize=9)
        ax.set_title(f"{bs} patients", fontsize=13)
        ax.set_ylabel("Makespan (min)" if i==0 else "")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(["Seq.", "Stage-Seq.", "Pipeline", "MAS"],
                           fontsize=9, rotation=15)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Pipeline Makespan: 4 Resource Allocation Modes", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig("simulation_results/makespan_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # FIGURE 2: CPU utilization — all 4 modes
    # ========================================================
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(BATCHES)); wd = 0.2
    for i, mode in enumerate(modes):
        vals = [df[(df["batch"]==b)&(df["mode"]==mode)]["cpu_util"].values[0] for b in BATCHES]
        bars = ax.bar(x + i*wd, vals, wd, label=mode, color=list(clr.values())[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x + 1.5*wd); ax.set_xticklabels(BATCHES)
    ax.set_xlabel("Batch size (patients)"); ax.set_ylabel("Avg CPU Utilization (%)")
    ax.set_title("CPU Utilization: 4 Modes"); ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(0, 100)
    plt.savefig("simulation_results/cpu_utilization_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # FIGURE 3: MAS allocation timeline (100 patients)
    # ========================================================
    mas100 = mode_mas(100)
    if mas100["alloc_log"]:
        adf = pd.DataFrame(mas100["alloc_log"])
        fig, ax = plt.subplots(figsize=(12, 5))
        for s in range(1, 6):
            ax.plot(adf["time"], adf[f"w{s}"], "o-", label=EMP[s]["name"], markersize=3)
        ax.axhline(y=CPU, color="k", ls="--", alpha=.3, label=f"Pool ({CPU})")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Workers")
        ax.set_title("MAS: Dynamic Worker Allocation (100 patients)")
        ax.legend(fontsize=8); ax.grid(True, alpha=.3); ax.set_ylim(0, CPU+2)
        plt.savefig("simulation_results/mas_allocation_timeline.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print("\n✅ All results in simulation_results/")


if __name__ == "__main__":
    main()