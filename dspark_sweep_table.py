"""Build the proper-sampling sweep table from sw_{method}_{ds}_c{conc}.json."""
import json, glob, os, re

cells = {}  # (ds, conc, method) -> result
for f in glob.glob("dspark_results/sw_*.json"):
    m = re.match(r"sw_(dflash|dspark)_(.+)_c(\d+)\.json", os.path.basename(f))
    if not m:
        continue
    method, ds, c = m.group(1), m.group(2), int(m.group(3))
    r = json.load(open(f))["results"]
    # single-conc json: results keyed by that conc
    cells[(ds, c, method)] = r.get(str(c)) or next(iter(r.values()))

datasets = sorted({ds for (ds, _, _) in cells})
for ds in datasets:
    concs = sorted({c for (d, c, _) in cells if d == ds})
    print(f"\n=== {ds} (n=max(1024, conc*32)) ===")
    print(f"{'conc':>4} | {'DFl tok/s':>9} {'DSp tok/s':>9} {'ratio':>6} | {'DFl acc':>7} {'DSp acc':>7} | {'n':>5}")
    for c in concs:
        a = cells.get((ds, c, "dflash")); b = cells.get((ds, c, "dspark"))
        if not a or not b:
            print(f"{c:>4} | missing"); continue
        r = b["tok_s"] / a["tok_s"] if a["tok_s"] else 0
        n = max(1024, c * 32)
        print(f"{c:>4} | {a['tok_s']:>9} {b['tok_s']:>9} {r:>6.3f} | "
              f"{a['accept_len']:>7} {b['accept_len']:>7} | {n:>5}")
