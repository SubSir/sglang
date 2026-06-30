"""Inspect the hf-cache modal Volume to locate the pre-downloaded Kimi-K2.6 target +
DFlash draft and figure out the cache layout / HF_HOME to point sglang at."""
import modal

app = modal.App("kimi-inspect")
image = modal.Image.debian_slim(python_version="3.12")
vol = modal.Volume.from_name("hf-cache")


@app.function(image=image.pip_install("huggingface_hub"), volumes={"/cache": vol},
              timeout=600, secrets=[modal.Secret.from_name("huggingface-secret")])
def inspect():
    import os, json, glob
    # --- online existence/access check for the 397B repos (not in volume) ---
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    print("########## ONLINE check (397B) ##########")
    for repo in ("Qwen/Qwen3.5-397B-A17B", "z-lab/Qwen3.5-397B-A17B-DFlash"):
        try:
            files = list_repo_files(repo, token=tok)
            nst = len([f for f in files if f.endswith(".safetensors")])
            print(f"  {repo}: EXISTS, {len(files)} files, {nst} safetensors")
            try:
                cfg = json.load(open(hf_hub_download(repo, "config.json", token=tok)))
                tc = cfg.get("text_config", {})
                print(f"     config OK: arch={cfg.get('architectures')} mt={cfg.get('model_type')} "
                      f"block_size={cfg.get('block_size')} hidden={cfg.get('hidden_size') or tc.get('hidden_size')} "
                      f"layers={cfg.get('num_hidden_layers') or tc.get('num_hidden_layers')} "
                      f"quant={ (cfg.get('quantization_config') or {}).get('quant_method') }")
            except Exception as e:
                print(f"     config DOWNLOAD FAILED (gated?): {str(e)[:160]}")
        except Exception as e:
            print(f"  {repo}: {str(e)[:160]}")

    for repo in ("models--Qwen--Qwen3.5-397B-A17B", "models--z-lab--Qwen3.5-397B-A17B-DFlash",
                 "models--moonshotai--Kimi-K2.6", "models--z-lab--Kimi-K2.6-DFlash"):
        base = f"/cache/hub/{repo}/snapshots"
        print(f"\n##### {repo} #####")
        if not os.path.isdir(base):
            print("  NO snapshots dir"); continue
        for snap in os.listdir(base):
            d = os.path.join(base, snap)
            files = os.listdir(d)
            nst = len(glob.glob(os.path.join(d, "*.safetensors")))
            # blobs may be symlinks; check real sizes
            tot = 0
            for f in glob.glob(os.path.join(d, "*.safetensors")):
                try: tot += os.path.getsize(os.path.realpath(f))
                except Exception: pass
            print(f"  snap {snap}: {len(files)} files, {nst} safetensors, "
                  f"{tot/1e9:.1f} GB on disk")
            print("    json:", [f for f in files if f.endswith('.json')])
            cfgp = os.path.join(d, "config.json")
            if os.path.exists(cfgp):
                c = json.load(open(cfgp))
                tc = c.get("text_config", {})
                print("    arch=", c.get("architectures"), "model_type=", c.get("model_type"),
                      "block_size=", c.get("block_size"), "num_draft=", c.get("num_draft_tokens"),
                      "draft_window=", c.get("draft_window_size"),
                      "hidden=", c.get("hidden_size") or tc.get("hidden_size"),
                      "layers=", c.get("num_hidden_layers") or tc.get("num_hidden_layers"))
    print("\n=== /cache top-level ===")
    for p in sorted(os.listdir("/cache")):
        full = os.path.join("/cache", p)
        kind = "d" if os.path.isdir(full) else "f"
        print(f"  {kind} {p}")
    # Walk a couple levels to find kimi / dflash model dirs.
    print("\n=== dirs containing 'kimi' or 'dflash' (depth<=4) ===")
    hits = []
    for root, dirs, files in os.walk("/cache"):
        depth = root[len("/cache"):].count(os.sep)
        if depth > 4:
            dirs[:] = []
            continue
        low = root.lower()
        if "kimi" in low or "dflash" in low or "k2" in low:
            nsafet = len([f for f in files if f.endswith(".safetensors")])
            print(f"  {root}  (files={len(files)} safetensors={nsafet})")
            if "config.json" in files:
                hits.append(os.path.join(root, "config.json"))
    print("\n=== config.json found in kimi/dflash dirs ===")
    for c in hits:
        try:
            cfg = json.load(open(c))
            arch = cfg.get("architectures") or cfg.get("model_type")
            qc = cfg.get("quantization_config", {})
            print(f"  {c}")
            print(f"     arch={arch} model_type={cfg.get('model_type')} "
                  f"hidden={cfg.get('hidden_size') or (cfg.get('text_config') or {}).get('hidden_size')} "
                  f"quant_bits={qc.get('config_groups',{}).get('group_0',{}).get('weights',{}).get('num_bits') if qc else None} "
                  f"dflash={cfg.get('dflash') or cfg.get('block_size') or cfg.get('draft_window_size')}")
        except Exception as e:
            print(f"  {c}  ERR {e}")
    print("\nHF env:", [k for k in os.environ if "HF" in k.upper() or "HUGG" in k.upper()])


@app.local_entrypoint()
def main():
    inspect.remote()
