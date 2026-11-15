"""Stage 2: Ptarmigan scan over collision parameters and real/sampled Gemini beams.

  python scripts/stage2_scan.py design --n 4096          # beam pool + Sobol design (~1 min)
  python scripts/stage2_scan.py extend --n 4096          # append the next 4096 points of the same sequence
  python scripts/stage2_scan.py run --workers 8          # run all pending rows; resumable (~25 s/run/core)
  python scripts/stage2_scan.py run --workers 4 --limit 16   # a pilot
  python scripts/stage2_scan.py collect                  # merge into runs.parquet + spectra.npz

Radiation-reaction model: --model stochastic (default, data/processed/stage2/), classical or continuous
(data/processed/stage2_<model>/). Other models reuse the stochastic scan's design and beam pool, so every
model is simulated at exactly the same collisions:  python scripts/stage2_scan.py --model classical run
Needs Ptarmigan (scripts/install_ptarmigan.sh); set PTARMIGAN_BIN if it is not in ~/build/ptarmigan.
Files go to data/processed/stage2/ (gitignored). Each finished run is appended to results/part-*.jsonl,
so an interrupted scan restarts where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlsfqed.simulation.doe import RANGES, beam_pool, sobol_design
from mlsfqed.simulation.ptarmigan import E_BINS_E, E_BINS_G, ptarmigan_bin, run_one

BASE = ROOT / "data" / "processed" / "stage2"
OUT = BASE
MODEL = "stochastic"
N_ELECTRONS = 5000


def model_dir(model):
    return BASE if model == "stochastic" else BASE.with_name(f"stage2_{model}")


def ensure_model_design():
    """For a non-default model, copy the stochastic scan's design and beam pool (same collisions)."""
    import shutil
    if OUT == BASE or (OUT / "design.parquet").exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("design.parquet", "design_meta.json", "beam_pool.npz"):
        shutil.copy(BASE / f, OUT / f)
    meta = json.loads((OUT / "design_meta.json").read_text())
    meta["model"] = MODEL
    (OUT / "design_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{MODEL}: copied design and beam pool from {BASE.name}")


def cmd_design(n, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pool = beam_pool(seed)
    np.savez_compressed(OUT / "beam_pool.npz", **pool)
    des = sobol_design(n, len(pool["Q_pC"]), seed)
    import pandas as pd
    pd.DataFrame(des).to_parquet(OUT / "design.parquet")
    (OUT / "design_meta.json").write_text(json.dumps(
        {"n": n, "seed": seed, "ranges": RANGES, "n_beams": len(pool["Q_pC"]), "n_electrons": N_ELECTRONS,
         "rates": "LMA, quantum stochastic, RR on, pairs off",
         "dz_sampling": "250 um * sign(v) * v^2, v ~ U(-1, 1)"}, indent=2))
    print(f"design: {n} runs over {len(pool['Q_pC'])} beams ({time.time() - t0:.0f} s) -> {OUT}")


def cmd_extend(n):
    import pandas as pd
    des = pd.read_parquet(OUT / "design.parquet")
    meta = json.loads((OUT / "design_meta.json").read_text())
    new = pd.DataFrame(sobol_design(n, meta["n_beams"], meta["seed"], skip=len(des)))
    assert new.run_id.min() == des.run_id.max() + 1
    pd.concat([des, new], ignore_index=True).to_parquet(OUT / "design.parquet")
    meta["n"] = len(des) + n
    meta.setdefault("extensions", []).append({"added": n, "skip": len(des)})
    (OUT / "design_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"design extended: {len(des)} -> {len(des) + n} runs (run_id {len(des)}-{len(des) + n - 1})")


def _work(row, spec, E, q, model="stochastic"):
    p = {k: row[k] for k in ("a0", "waist_um", "fwhm_fs", "dx_um", "dy_um", "dz_um",
                             "radius_um", "length_um", "div_mrad")}
    p["charge_pC"] = float(q)
    o, Ne, Ng = run_one(p, spec, E, N_ELECTRONS, seed=int(row["run_id"]) + 1, model=model)
    o["run_id"] = int(row["run_id"])
    if o["ok"]:
        o["Ne"] = Ne.tolist()
        o["Ng"] = Ng.tolist()
    return o


def done_ids():
    ids = set()
    for f in (OUT / "results").glob("part-*.jsonl"):
        for line in f.read_text().splitlines():
            ids.add(json.loads(line)["run_id"])
    return ids


def cmd_run(workers, limit):
    import pandas as pd
    ensure_model_design()
    if not ptarmigan_bin().exists():
        sys.exit(f"Ptarmigan not found at {ptarmigan_bin()} - run scripts/install_ptarmigan.sh or set PTARMIGAN_BIN")
    des = pd.read_parquet(OUT / "design.parquet")
    pool = np.load(OUT / "beam_pool.npz")
    E, spec, Q = pool["E_MeV"], pool["dNdE_per_MeV"], pool["Q_pC"]
    todo = des[~des.run_id.isin(done_ids())]
    if limit:
        todo = todo.head(limit)
    (OUT / "results").mkdir(exist_ok=True)
    part = OUT / "results" / f"part-{int(time.time())}-{os.getpid()}.jsonl"
    t0, n_ok = time.time(), 0
    print(f"{len(todo)} runs to do with {workers} workers")
    with ProcessPoolExecutor(workers) as ex, open(part, "a") as fh:
        futs = [ex.submit(_work, r, spec[int(r["beam_index"])], E, Q[int(r["beam_index"])], MODEL)
                for r in todo.to_dict("records")]
        for i, f in enumerate(as_completed(futs), 1):
            o = f.result()
            fh.write(json.dumps(o) + "\n")
            fh.flush()
            n_ok += o["ok"]
            if i % max(1, len(futs) // 20) == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  {i}/{len(futs)} done ({n_ok} ok), {el / 60:.1f} min, ~{el / i * (len(futs) - i) / 60:.0f} min left", flush=True)


def cmd_collect():
    import pandas as pd
    recs = [json.loads(line) for f in sorted((OUT / "results").glob("part-*.jsonl")) for line in f.read_text().splitlines()]
    ok = [r for r in recs if r["ok"]]
    Ne = np.array([r.pop("Ne") for r in ok])
    Ng = np.array([r.pop("Ng") for r in ok])
    res = pd.DataFrame(recs).drop(columns=["Ne", "Ng"], errors="ignore")
    des = pd.read_parquet(OUT / "design.parquet")
    df = des.merge(res, on="run_id", how="inner").sort_values("run_id")
    pool = np.load(OUT / "beam_pool.npz")
    E = pool["E_MeV"]
    w = pool["dNdE_per_MeV"][df.beam_index.values]
    df["beam_Q_pC"] = pool["Q_pC"][df.beam_index.values]
    df["beam_mean_MeV"] = (w * E).sum(1) / w.sum(1)
    df["beam_source"] = pool["source"][df.beam_index.values]
    df.to_parquet(OUT / "runs.parquet")
    ids = [r["run_id"] for r in ok]
    np.savez_compressed(OUT / "spectra.npz", run_id=ids, Ne=Ne, Ng=Ng, E_edges=E_BINS_E, G_edges=E_BINS_G)
    print(f"collected {len(df)} runs ({len(ok)} ok, {len(recs) - len(ok)} failed) -> runs.parquet, spectra.npz")
    if len(recs) > len(ok):
        print(pd.Series([r.get("error") for r in recs if not r["ok"]]).value_counts().head())


def main():
    global OUT, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["stochastic", "classical", "continuous"], default="stochastic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design"); d.add_argument("--n", type=int, default=4096); d.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run"); r.add_argument("--workers", type=int, default=os.cpu_count()); r.add_argument("--limit", type=int)
    e = sub.add_parser("extend"); e.add_argument("--n", type=int, default=4096)
    sub.add_parser("collect")
    a = ap.parse_args()
    MODEL, OUT = a.model, model_dir(a.model)
    if a.cmd == "design" and MODEL != "stochastic":
        sys.exit("design is shared: create it with the default model, other models copy it")
    if a.cmd == "design":
        cmd_design(a.n, a.seed)
    elif a.cmd == "extend":
        cmd_extend(a.n)
    elif a.cmd == "run":
        cmd_run(a.workers, a.limit)
    else:
        cmd_collect()


if __name__ == "__main__":
    main()
