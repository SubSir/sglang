import gzip, json, bisect
from collections import defaultdict

TREEBUILD = ('build_tree', '_dflash_expand_topk4', '_dflash_tree_verify_steps',
             'mbtopk', 'gatherTopK', 'computeBlockDigit', 'radixFindKth', 'Sort')
ACCEPT = ('accept', 'fill_bonus', 'fill_accept', 'Memcpy', 'Memset')

def cls(nm):
    if any(t in nm for t in TREEBUILD): return 'tree_build'
    if any(t in nm for t in ACCEPT): return 'accept_overhead'
    if 'nvjet' in nm or 'cublas' in nm or 'gemm' in nm.lower(): return 'model_gemm'
    if '_fwd_kernel' in nm or 'flashinfer' in nm or 'store_kvcache' in nm or 'attn' in nm.lower():
        return 'model_attn'
    if 'Norm' in nm or 'rope' in nm or 'qknorm' in nm or 'act_and_mul' in nm: return 'model_other'
    return 'accept_overhead'

def load(path):
    ev = json.loads(gzip.open(path).read())['traceEvents']
    rb, tv, kern = [], [], []
    for e in ev:
        if e.get('ph') != 'X' or 'dur' not in e: continue
        c = e.get('cat', ''); n = e.get('name', '')
        if c == 'user_annotation':
            if n == 'scheduler.run_batch': rb.append((e['ts'], e['ts'] + e['dur']))
            elif n.startswith('step[TARGET_VERIFY'): tv.append((e['ts'], e['ts'] + e['dur']))
        elif c in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            kern.append((e['ts'], e['ts'] + e['dur'], n))
    rb.sort(); tv.sort(); kern.sort()
    return rb, tv, kern

def gpu_busy(kern, a, b):
    ivs = sorted((max(a, ks), min(b, ke)) for ks, ke, _ in kern if ks < b and ke > a and min(b, ke) > max(a, ks))
    tot = 0.0; cs = ce = None
    for s, e in ivs:
        if cs is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    if cs is not None: tot += ce - cs
    return tot

def analyze(path, label, n_verify_per_step):
    rb, tv, kern = load(path)
    tvs = [a for a, _ in tv]
    # decode steps = run_batch with exactly n_verify_per_step verify spans
    steps = []
    for a, b in rb:
        lo = bisect.bisect_left(tvs, a)
        sp = [tv[i] for i in range(lo, len(tv)) if tv[i][0] <= b]
        if len(sp) == n_verify_per_step:
            steps.append((a, b, sp))
    ns = len(steps)
    kstarts = [k[0] for k in kern]
    buckets = defaultdict(float); gpu_tot = wall_tot = 0.0
    fwd_split = defaultdict(float)  # per-verify-span index -> gpu-busy (draft vs verify)
    for a, b, sp in steps:
        wall_tot += b - a
        gpu_tot += gpu_busy(kern, a, b)
        # name-based bucketing of all kernels in step
        lo = bisect.bisect_left(kstarts, a) - 3
        for i in range(max(0, lo), len(kern)):
            ks, ke, nm = kern[i]
            if ks >= b: break
            if ke <= a: continue
            buckets[cls(nm)] += ke - ks
        # draft vs verify: GPU busy between step-start..end-of-span0  vs  start-of-span1..step-end
        # span0 region = [a, sp[0][1]] ; span1 region = [sp[1][0], b]  (kernels launched by each fwd land after its cpu span)
        if n_verify_per_step == 2:
            fwd_split['draft'] += gpu_busy(kern, a, sp[1][0])       # everything up to 2nd span start
            fwd_split['verify'] += gpu_busy(kern, sp[1][0], b)       # 2nd span onward
    return dict(label=label, steps=ns, per_step_gpu=gpu_tot/ns, per_step_wall=wall_tot/ns,
                per_step_idle=(wall_tot-gpu_tot)/ns, util=100*gpu_tot/wall_tot,
                buckets={k: v/ns for k, v in buckets.items()},
                buckets_raw_tot=sum(buckets.values())/ns,
                fwd_split={k: v/ns for k, v in fwd_split.items()})

tree = analyze('prof_results/tree_b32_k4_1782883876.trace.json.gz', 'tree_b32_k4', 2)
chain = analyze('prof_results/chain_1782884059.trace.json.gz', 'chain', 2)

def scale(d):
    # scale name-bucket raw shares onto de-overlapped per-step gpu-busy
    tot = d['buckets_raw_tot']
    return {k: v/tot*d['per_step_gpu'] for k, v in d['buckets'].items()}

print("="*74)
print(f"{'':22}{'CHAIN':>14}{'TREE':>14}{'tree-chain':>16}")
print("-"*74)
def row(name, ck, tk):
    print(f"{name:>22}{ck:>14.0f}{tk:>14.0f}{tk-ck:>+16.0f}")
cs, ts = scale(chain), scale(tree)
order = ['model_gemm', 'model_attn', 'model_other', 'tree_build', 'accept_overhead']
for k in order:
    row(k, cs.get(k, 0), ts.get(k, 0))
print("-"*74)
row('== MODEL fwd total', sum(cs.get(k,0) for k in ['model_gemm','model_attn','model_other']),
    sum(ts.get(k,0) for k in ['model_gemm','model_attn','model_other']))
row('per-step GPU-busy', chain['per_step_gpu'], tree['per_step_gpu'])
row('per-step WALL', chain['per_step_wall'], tree['per_step_wall'])
row('per-step IDLE(gap)', chain['per_step_idle'], tree['per_step_idle'])
print(f"{'util %':>22}{chain['util']:>14.0f}{tree['util']:>14.0f}")
print("="*74)
print(f"steps: chain={chain['steps']} tree={tree['steps']}")
print(f"TREE forward split (2 passes/step): draft={tree['fwd_split']['draft']:.0f}us  verify(32tok)={tree['fwd_split']['verify']:.0f}us")
print(f"CHAIN is 1 verify pass (16 tok) + draft folded; per-step GPU={chain['per_step_gpu']:.0f}us")
