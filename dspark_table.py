"""Build the deliverable table from pulled dspark_results/{dflash,dspark}_{ds}.json.

Per dataset: rows = concurrency; cols = DFlash tok/s, DSpark tok/s, ratio,
DSpark accept, DFlash step_ms, DSpark step_ms, head-overhead (=dspark_step -
dflash_step) us & % of dspark step."""
import json, glob, os

def load(method, ds):
    p = f"dspark_results/{method}_{ds}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["results"]

datasets = sorted({os.path.basename(f).split("_",1)[1].rsplit(".",1)[0]
                   for f in glob.glob("dspark_results/dflash_*.json")})
for ds in datasets:
    df, ds_ = load("dflash", ds), load("dspark", ds)
    if not df or not ds_:
        print(f"{ds}: missing ({'dflash' if not df else ''}{'dspark' if not ds_ else ''})"); continue
    print(f"\n=== {ds} ===")
    print(f"{'conc':>4} | {'DFl tok/s':>9} {'DSp tok/s':>9} {'ratio':>6} | "
          f"{'DSp acc':>7} {'DFl acc':>7} | {'DFl ms':>7} {'DSp ms':>7} | {'head us':>8} {'%step':>6}")
    for c in sorted(int(k) for k in df):
        a, b = df[str(c)], ds_[str(c)]
        r = (b["tok_s"]/a["tok_s"]) if a["tok_s"] else None
        head_us = ((b["step_ms"]-a["step_ms"])*1000) if (a["step_ms"] and b["step_ms"]) else None
        pct = (head_us/(b["step_ms"]*1000)*100) if (head_us and b["step_ms"]) else None
        print(f"{c:>4} | {a['tok_s']:>9} {b['tok_s']:>9} {r:>6.3f} | "
              f"{b['accept_len']:>7} {a['accept_len']:>7} | {a['step_ms']:>7} {b['step_ms']:>7} | "
              f"{(head_us or 0):>8.0f} {(pct or 0):>5.1f}%")
