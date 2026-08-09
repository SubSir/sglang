"""What the DFlash2 export adds to a DFlash config, key by key.

    modal run modal_cfg_diff.py

Reads draft-final for both drafts -- the trainer's own export, not the sglang-converted
copy -- and diffs their configs, so the serving side's config surface can be checked
against what the checkpoint actually declares.
"""

import json
from pathlib import Path

import modal

VOL = Path("/vol")
DIRS = {
    "dflash2": "dflash2-clean-g16-k2-top16-qwen3-4b-thinking100k-gbs64-seed42/draft-final",
    "dflash": "dflash-clean-qwen3-4b-thinking100k-block8-L5-gbs64-lr6e4-3ep-seed42-b300/draft-final",
}

app = modal.App("dflash2-cfg-diff")
runs = modal.Volume.from_name("autodflash-dspark-train")


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
              volumes={str(VOL): runs}, timeout=600)
def diff() -> str:
    out = []
    loaded = {}
    for name, d in DIRS.items():
        path = VOL / d / "config.json"
        if not path.exists():
            out.append(f"{name}: {path} MISSING; dir holds "
                       f"{sorted(p.name for p in (VOL / d).iterdir())[:12]}")
            continue
        loaded[name] = json.loads(path.read_text())

    if len(loaded) != 2:
        return "\n".join(out)

    a, b = loaded["dflash2"], loaded["dflash"]
    out.append("--- top level: in dflash2, not in dflash ---")
    for k in sorted(set(a) - set(b)):
        out.append(f"  {k} = {json.dumps(a[k])[:120]}")
    out.append("--- top level: differing values ---")
    for k in sorted(set(a) & set(b)):
        if a[k] != b[k] and k != "dflash_config":
            out.append(f"  {k}: dflash2={json.dumps(a[k])[:60]} dflash={json.dumps(b[k])[:60]}")

    ca, cb = a.get("dflash_config") or {}, b.get("dflash_config") or {}
    out.append("--- dflash_config: in dflash2, not in dflash ---")
    for k in sorted(set(ca) - set(cb)):
        out.append(f"  {k} = {json.dumps(ca[k])[:120]}")
    out.append("--- dflash_config: differing values ---")
    for k in sorted(set(ca) & set(cb)):
        if ca[k] != cb[k]:
            out.append(f"  {k}: dflash2={json.dumps(ca[k])[:60]} dflash={json.dumps(cb[k])[:60]}")
    out.append("--- dflash_config: in dflash, not in dflash2 ---")
    for k in sorted(set(cb) - set(ca)):
        out.append(f"  {k} = {json.dumps(cb[k])[:120]}")
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(diff.remote())
