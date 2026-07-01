"""DDTree-in-SGLang (PR #27509, HaiyuZhou/sglang) — overlay onto a prebuilt nightly.

The PR is 100% Python (ddtree_{info,utils,worker}.py + edits to 16 srt files, NO .cu),
so we overlay HaiyuZhou's full python/sglang over the nightly image's installed package
and keep the nightly's compiled sgl_kernel (_C). Same trick as modal_vllm_jetspec.py.

DDTree extends DFlashWorker and uses the SAME z-lab DFlash draft head as our fork, so
DFLASH (chain) and DDTREE (tree) run in ONE engine — a clean same-engine tree-vs-chain
comparison at higher concurrency.

  modal run modal_ddtree_sglang.py::check                 # no-GPU: overlay imports + DDTREE registered
  modal run modal_ddtree_sglang.py::bench                 # conc sweep: dflash vs ddtree
  modal run modal_ddtree_sglang.py::pull
"""
import os
import modal

app = modal.App("ddtree-sglang-bench")
vol = modal.Volume.from_name("ddtree-sglang-results", create_if_missing=True)

PR_SHA = "ad7ab160cbb3"

image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu12-20260627-13b5bd96")
    .run_commands(
        "echo dds1 > /tmp/bt",
        f"git clone https://github.com/HaiyuZhou/sglang /root/ddtree-sglang && "
        f"cd /root/ddtree-sglang && git checkout {PR_SHA}",
        # overlay PR's python over installed sglang (keep nightly's compiled .so)
        "SITE=$(python -c 'import sglang,os;print(os.path.dirname(os.path.dirname(sglang.__file__)))') && "
        "echo \"sglang site at $SITE\" && cp -rf /root/ddtree-sglang/python/sglang/. \"$SITE/sglang/\" && "
        "echo overlaid DDTree-PR python on nightly sglang",
    )
    .add_local_file("ddtree_fix.py", "/root/ddtree_fix.py", copy=True)
    .run_commands(
        "echo ddtree_fix_v2 && "
        "SITE=$(python -c 'import sglang,os;print(os.path.dirname(os.path.dirname(sglang.__file__)))') && "
        "python /root/ddtree_fix.py \"$SITE/sglang/srt\"",
    )
)


@app.function(image=image, timeout=600)
def check():
    """No-GPU: confirm the overlay imports and DDTREE is registered."""
    import subprocess as sp
    r = sp.run(["python", "-c",
                "import sglang; print('sglang', sglang.__version__);"
                "from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as S;"
                "print('DDTREE' , hasattr(S,'DDTREE'));"
                "from sglang.srt.server_args import ServerArgs;"
                "print('has ddtree_budget', hasattr(ServerArgs,'speculative_ddtree_budget'))"],
               capture_output=True, text=True)
    print("STDOUT:", r.stdout)
    print("STDERR(tail):", "\n".join(r.stderr.splitlines()[-30:]))
    return r.stdout


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def bench(algo: str, budget: int, concurrencies: str, tag: str,
          target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
          num_prompts: int = 128, extra_flags: list = None):
    """Launch a DDTree/DFLASH sglang server and bench_serving at each concurrency."""
    import subprocess, time, signal, json

    env = dict(os.environ)
    spec = ["--speculative-algorithm", algo, "--speculative-draft-model-path", draft]
    if algo == "DDTREE":
        spec += ["--speculative-ddtree-budget", str(budget)]
    # dflash/ddtree share block-size style draft config; let server defaults apply.
    server = [
        "python", "-m", "sglang.launch_server",
        "--model-path", target, "--trust-remote-code",
        "--mem-fraction-static", "0.85", "--max-running-requests",
        str(max(int(c) for c in concurrencies.split(","))),
        *spec, "--port", "30000", *(extra_flags or []),
    ]
    print(">>> SERVER:", " ".join(server), flush=True)
    srv = subprocess.Popen(server, env=env)
    # wait for readiness
    import urllib.request
    ready = False
    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2)
            ready = True
            break
        except Exception:
            time.sleep(2)
    out = {"algo": algo, "budget": budget, "ready": ready, "rows": []}
    if ready:
        for conc in concurrencies.split(","):
            b = subprocess.run(
                ["python", "-m", "sglang.bench_serving", "--backend", "sglang",
                 "--dataset-name", "sharegpt", "--num-prompts", str(num_prompts),
                 "--max-concurrency", conc, "--port", "30000"],
                env=env, capture_output=True, text=True)
            tail = b.stdout + "\n" + "\n".join(b.stderr.splitlines()[-10:])
            out["rows"].append({"conc": conc, "log": tail[-2500:]})
            print(f"=== conc={conc} ===\n{tail[-1500:]}", flush=True)
    srv.send_signal(signal.SIGINT)
    with open(f"/results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return f"{tag}: ready={ready}"


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def bench_gsm8k(algo: str, budget: int, concurrencies: str, tag: str,
                target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
                n: int = 32, extra_flags: list = None):
    """Launch DFLASH/DDTREE server, hit it with the SAME gsm8k first-N prompts as our
    fork (same format, enable_thinking off), measure decode tok/s per concurrency.
    Isolates engine per-forward speed vs our fork (same dataset+head)."""
    import subprocess, time, signal, json, threading, urllib.request
    from datasets import load_dataset
    from transformers import AutoTokenizer

    env = dict(os.environ)
    spec = ["--speculative-algorithm", algo, "--speculative-draft-model-path", draft]
    if algo == "DDTREE":
        spec += ["--speculative-ddtree-budget", str(budget)]
    server = ["python", "-m", "sglang.launch_server", "--model-path", target,
              "--trust-remote-code", "--mem-fraction-static", "0.85",
              "--max-running-requests", str(max(int(c) for c in concurrencies.split(","))),
              *spec, "--port", "30000", *(extra_flags or [])]
    print(">>> SERVER:", " ".join(server), flush=True)
    srv = subprocess.Popen(server, env=env)
    ready = False
    for _ in range(240):
        try:
            urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); ready = True; break
        except Exception:
            time.sleep(2)

    tok = AutoTokenizer.from_pretrained(target)
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    prompts = [tok.apply_chat_template([{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for i in range(n)]

    def gen(p, out):
        req = urllib.request.Request("http://127.0.0.1:30000/generate",
            data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        out.append(r["meta_info"]["completion_tokens"])

    out = {"algo": algo, "budget": budget, "ready": ready, "rows": []}
    if ready:
        # warmup
        gen(prompts[0], [])
        for conc in [int(c) for c in concurrencies.split(",")]:
            toks = []
            t0 = time.perf_counter()
            i = 0
            while i < len(prompts):
                batch = prompts[i:i+conc]
                threads = [threading.Thread(target=gen, args=(p, toks)) for p in batch]
                [t.start() for t in threads]; [t.join() for t in threads]
                i += conc
            dt = time.perf_counter() - t0
            total = sum(toks)
            out["rows"].append({"conc": conc, "tok_s": round(total/dt, 1), "tokens": total, "secs": round(dt, 2)})
            print(f"  conc={conc}: {round(total/dt,1)} tok/s ({total} tok / {round(dt,2)}s)", flush=True)
    srv.send_signal(signal.SIGINT)
    with open(f"/results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return f"{tag}: ready={ready} {out['rows']}"


def _load_prompts(dataset: str, n: int, target: str):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
        fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        msgs = [[{"role": "user", "content": fmt.format(q=ds[i]["question"])}] for i in range(n)]
    elif dataset == "mt-bench":
        # mt-bench: 80 questions, take turn-1 prompt of first n
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        msgs = [[{"role": "user", "content": ds[i % len(ds)]["prompt"][0]}] for i in range(n)]
    else:
        raise ValueError(dataset)
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False) for m in msgs]


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def bench_matrix(algo: str, budget: int, dataset: str, concurrencies: str, tag: str,
                 target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
                 n: int = 40, extra_flags: list = None):
    """One server, sweep concurrencies on a dataset; capture tok/s + accept length.
    DDTree tree mode runs on triton eager (the PR's working tree-verify config on B200)."""
    import subprocess, time, signal, json, threading, urllib.request
    env = dict(os.environ)
    spec = ["--speculative-algorithm", algo, "--speculative-draft-model-path", draft]
    if algo == "DDTREE":
        spec += ["--speculative-ddtree-budget", str(budget)]
    server = ["python", "-m", "sglang.launch_server", "--model-path", target,
              "--trust-remote-code", "--mem-fraction-static", "0.85",
              "--max-running-requests", str(max(int(c) for c in concurrencies.split(","))),
              *spec, "--port", "30000", *(extra_flags or [])]
    print(">>> SERVER:", " ".join(server), flush=True)
    srv = subprocess.Popen(server, env=env)
    ready = False
    for _ in range(300):
        if srv.poll() is not None:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); ready = True; break
        except Exception:
            time.sleep(2)

    out = {"algo": algo, "budget": budget, "dataset": dataset, "ready": ready,
           "exit": srv.poll(), "rows": []}
    if ready:
        prompts = _load_prompts(dataset, n, target)

        def gen(p, acc):
            req = urllib.request.Request("http://127.0.0.1:30000/generate",
                data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}).encode(),
                headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            mi = r["meta_info"]
            acc.append((mi["completion_tokens"], mi.get("spec_accept_length", 0.0)))

        gen(prompts[0], [])  # warmup
        for conc in [int(c) for c in concurrencies.split(",")]:
            acc = []
            t0 = time.perf_counter()
            i = 0
            while i < len(prompts):
                batch = prompts[i:i+conc]
                threads = [threading.Thread(target=gen, args=(p, acc)) for p in batch]
                [t.start() for t in threads]; [t.join() for t in threads]
                i += conc
            dt = time.perf_counter() - t0
            total = sum(t for t, _ in acc)
            mean_acc = round(sum(a for _, a in acc) / len(acc), 3) if acc else 0
            row = {"conc": conc, "tok_s": round(total/dt, 1), "accept_len": mean_acc,
                   "tokens": total, "secs": round(dt, 2)}
            out["rows"].append(row)
            print(f"  {dataset} conc={conc}: {row['tok_s']} tok/s  accept={mean_acc}", flush=True)
        srv.send_signal(signal.SIGINT)
    with open(f"/results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return f"{tag}: ready={ready} {out['rows']}"


@app.function(gpu="H200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def bench_matrix_h200(algo: str, budget: int, dataset: str, concurrencies: str, tag: str,
                      target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
                      n: int = 40, extra_flags: list = None, mem_fraction: str = "0.85"):
    """Same as bench_matrix but on H200 (SM 9.0) — test whether DDTree's cuda-graph
    tree path (fa3, the PR's validated backend) loads + runs here, unlike B200."""
    import subprocess, time, signal, json, threading, urllib.request
    env = dict(os.environ)
    spec = ["--speculative-algorithm", algo, "--speculative-draft-model-path", draft]
    if algo == "DDTREE":
        spec += ["--speculative-ddtree-budget", str(budget)]
    server = ["python", "-m", "sglang.launch_server", "--model-path", target,
              "--trust-remote-code", "--mem-fraction-static", mem_fraction,
              "--max-running-requests", str(max(int(c) for c in concurrencies.split(","))),
              *spec, "--port", "30000", *(extra_flags or [])]
    print(">>> SERVER:", " ".join(server), flush=True)
    log = open(f"/results/{tag}_server.log", "w")
    srv = subprocess.Popen(server, env=env, stdout=log, stderr=subprocess.STDOUT)
    ready = False
    for _ in range(300):
        if srv.poll() is not None:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); ready = True; break
        except Exception:
            time.sleep(2)
    out = {"algo": algo, "budget": budget, "dataset": dataset, "gpu": "H200",
           "ready": ready, "exit": srv.poll(), "flags": extra_flags, "rows": []}
    if ready:
        prompts = _load_prompts(dataset, n, target)
        def gen(p, acc):
            req = urllib.request.Request("http://127.0.0.1:30000/generate",
                data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}).encode(),
                headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            mi = r["meta_info"]
            acc.append((mi["completion_tokens"], mi.get("spec_accept_length", 0.0)))
        gen(prompts[0], [])
        for conc in [int(c) for c in concurrencies.split(",")]:
            acc = []; t0 = time.perf_counter(); i = 0
            while i < len(prompts):
                batch = prompts[i:i+conc]
                threads = [threading.Thread(target=gen, args=(p, acc)) for p in batch]
                [t.start() for t in threads]; [t.join() for t in threads]
                i += conc
            dt = time.perf_counter() - t0
            total = sum(t for t, _ in acc)
            mean_acc = round(sum(a for _, a in acc) / len(acc), 3) if acc else 0
            row = {"conc": conc, "tok_s": round(total/dt, 1), "accept_len": mean_acc, "secs": round(dt, 2)}
            out["rows"].append(row)
            print(f"  {dataset} conc={conc}: {row['tok_s']} tok/s accept={mean_acc}", flush=True)
        srv.send_signal(signal.SIGINT)
    log.flush(); time.sleep(2); log.close()
    if not ready:
        tail = open(f"/results/{tag}_server.log").read()[-4000:]
        out["stderr_tail"] = tail
        print("=== SERVER LOG TAIL ===\n", tail, flush=True)
    with open(f"/results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return f"{tag}: ready={ready} exit={out['exit']} {out['rows']}"


@app.local_entrypoint()
def h200(budget: int = 32, concurrencies: str = "1,8,32", n: int = 40):
    """Test DDTree cuda-graph on H200: fa3+cg, triton+cg, and triton-eager baseline."""
    configs = [
        ("ddtree_h200_fa3_cg",     ["--attention-backend", "fa3"]),
        ("ddtree_h200_triton_cg",  ["--attention-backend", "triton"]),
        ("ddtree_h200_triton_eager", ["--attention-backend", "triton", "--disable-cuda-graph"]),
    ]
    hs = [(tag, bench_matrix_h200.spawn("DDTREE", budget, "gsm8k", concurrencies, tag,
                                        n=n, extra_flags=flags)) for tag, flags in configs]
    for tag, h in hs:
        try: print(tag, "->", h.get())
        except Exception as e: print(tag, "FAILED", e)


@app.local_entrypoint()
def h200_one(budget: int = 32, backend: str = "fa3", concurrencies: str = "1", n: int = 8,
             nocg: bool = False):
    """Single H200 config — fast validation of the fa3 cuda-graph fix."""
    flags = ["--attention-backend", backend] + (["--disable-cuda-graph"] if nocg else [])
    tag = f"ddtree_h200_one_{backend}{'_nocg' if nocg else '_cg'}_b{budget}"
    print(bench_matrix_h200.remote("DDTREE", budget, "gsm8k", concurrencies, tag,
                                   n=n, extra_flags=flags))


@app.local_entrypoint()
def h200_cg_lowmem(budget: int = 32, n: int = 24):
    """Rule out OOM: fa3 & triton cuda-graph on H200 with low mem + only bs=1 captured."""
    configs = [
        ("ddtree_h200_fa3_cg_lm",    ["--attention-backend", "fa3", "--cuda-graph-bs", "1"]),
        ("ddtree_h200_triton_cg_lm", ["--attention-backend", "triton", "--cuda-graph-bs", "1"]),
    ]
    hs = [(tag, bench_matrix_h200.spawn("DDTREE", budget, "gsm8k", "1", tag,
                                        n=n, extra_flags=flags, mem_fraction="0.70"))
          for tag, flags in configs]
    for tag, h in hs:
        try: print(tag, "->", h.get())
        except Exception as e: print(tag, "FAILED", e)


@app.local_entrypoint()
def matrix(concurrencies: str = "1,8,32", budgets: str = "16,32,64", n: int = 40):
    """Full DDTree tree-mode matrix: gsm8k + mt-bench x conc x budget on triton eager."""
    flags = ["--attention-backend", "triton", "--disable-cuda-graph"]
    handles = []
    for ds in ["gsm8k", "mt-bench"]:
        for b in [int(x) for x in budgets.split(",")]:
            h = bench_matrix.spawn("DDTREE", b, ds, concurrencies,
                                   f"ddtree_{ds}_b{b}", n=n, extra_flags=flags)
            handles.append(h)
    for h in handles:
        print(h.get())


@app.local_entrypoint()
def upstream_gsm8k(concurrencies: str = "1,8,32"):
    """Upstream-sglang DFLASH chain on gsm8k — apples-to-apples vs our fork."""
    print(bench_gsm8k.remote("DFLASH", 16, concurrencies, "upstream_dflash_gsm8k"))


@app.local_entrypoint()
def smoke():
    print(check.remote())


@app.local_entrypoint()
def run_bench(concurrencies: str = "1,8,32", budget: int = 64, num_prompts: int = 128):
    handles = [
        bench.spawn("DFLASH", budget, concurrencies, "dflash_chain", num_prompts=num_prompts),
        bench.spawn("DDTREE", budget, concurrencies, f"ddtree_b{budget}", num_prompts=num_prompts),
    ]
    for h in handles:
        print(h.get())


@app.local_entrypoint()
def run_ddtree(concurrencies: str = "1,8,32", budget: int = 64, num_prompts: int = 128):
    """DDTREE tree mode sharegpt bench on the WORKING config: triton eager.
    (B200 has no fa3; flashinfer has no tree-mask support; triton cuda-graph
    capture is an unfixed PR bug. ddtree_fix.py fixes the verify token-count
    mismatch so triton eager runs the tree correctly.)"""
    print(bench.remote("DDTREE", budget, concurrencies, f"ddtree_triton_eager_b{budget}",
                       num_prompts=num_prompts,
                       extra_flags=["--attention-backend", "triton", "--disable-cuda-graph"]))


@app.function(image=image, volumes={"/results": vol})
def _list():
    import os as o
    return {fn: open(f"/results/{fn}").read()
            for fn in sorted(o.listdir("/results")) if fn.endswith(".json")}


@app.local_entrypoint()
def pull():
    os.makedirs("ddtree_sglang_results", exist_ok=True)
    for fn, c in _list.remote().items():
        with open(f"ddtree_sglang_results/{fn}", "w") as f:
            f.write(c)
        print(f"\n===== {fn} =====\n{c[:1500]}")
