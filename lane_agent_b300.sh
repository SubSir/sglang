#!/usr/bin/env bash
# Table 9 (tab:kvheadline) agent-session lane on a smaix B300: Qwen3.6-35B-A3B + one 35B DFlash2 drafter, stock
# c1c287a4 image + the one-copy shared-K/V overlay (~/ev/ovl_skv1), explicit pools, bench_agent2.py sessions.
# Port of lane_agent13/23_pod.sh (GB300, overlay_glm25 + dflash-kv12) to stock flags:
#   bf16 checkpoint pool (--mamba-checkpoint-dtype bfloat16)  -> stock int8 pool (--enable-int8-mamba-checkpoint)
#   SGLANG_MAMBA_DROP_UNRESUMABLE_LEAVES (tree fix)          -> no stock equivalent, not set
#   --speculative-draft-window-size 2048                      -> omitted: stock turns it into the compact draft cache;
#                                                                the drafters are natively SWA-2048, draft rows stay 1:1 with P
#   lane_agent_b300.sh <gpu> <arm: base | shared> <ckpt slots N> <full tokens P>
#     base = q36-dflash2 (per-layer context K/V, 24 KiB/token), shared = q36-fusekv (one shared K/V, 4 KiB/token)
# env: OUTDIR TAG KV8(1 = fp8 target KV, draft KV kept bf16) MEMFRAC(0.6) RECORD(1) REPLAY(host path of a --record file)
#      SESS ROUNDS SYSTOK TOOLTOK OUTTOK INFLIGHT SEED LOGINT OVL
set -uo pipefail
GPU=$1; ARM=$2; N=$3; P=$4
IMG=lmsysorg/sglang:nightly-dev-cu13-20261021-c1c287a4
EV=$HOME/ev; HF=$EV/hf; OVL=${OVL:-$EV/ovl_skv1}
case $ARM in base) DS=$EV/drafts/q36-dflash2;; shared) DS=$EV/drafts/q36-fusekv;; *) echo "bad arm $ARM"; exit 1;; esac
TD=models--Qwen--Qwen3.6-35B-A3B; TS=/root/.cache/huggingface/hub/$TD/snapshots/$(ls $HF/hub/$TD/snapshots | head -1)
SESS=${SESS:-192}; ROUNDS=${ROUNDS:-5}; SYSTOK=${SYSTOK:-3800}; TOOLTOK=${TOOLTOK:-1000}; OUTTOK=${OUTTOK:-48}; INFLIGHT=${INFLIGHT:-32}
KVARG=""; [ "${KV8:-0}" = 1 ] && KVARG="--kv-cache-dtype fp8_e4m3 --speculative-draft-kv-cache-dtype bfloat16"
OUT=${OUTDIR:-$EV/out-agent}; STEM=${TAG:-}35b_${ARM}_n${N}_s${SESS}_o${OUTTOK}
# Holds isolate GPUs: resolve the session index to the card's UUID for podman CDI (as ev_lane.sh / tp_lane.sh).
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU 2>/dev/null) || U=$GPU
K=$(printf %s "$U" | cksum | cut -d" " -f1); S=$(printf %s "$U" | tail -c 8); PORT=$((33000 + K % 900)); NAME=ag_$(hostname -s)_$S
JIT=$EV/jit_ag_$(hostname -s)_$S; mkdir -p $OUT $JIT $EV/agent_prompts
REC=""; REPLAYARG=""
[ -n "${REPLAY:-}" ] && { REC="-v $(dirname $REPLAY):/rec:ro"; REPLAYARG="--replay /rec/$(basename $REPLAY)"; }
[ "${RECORD:-0}" = 1 ] && REPLAYARG="$REPLAYARG --record"
gpuapps() { echo "== $1 $(date '+%F %T')"; nvidia-smi -i $GPU --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
gpuapps start > $OUT/$STEM.gpuapps.txt 2>&1
trap 'podman rm -f $NAME >/dev/null 2>&1' EXIT
podman rm -f $NAME >/dev/null 2>&1
podman run -d --name $NAME --device nvidia.com/gpu=$U --pids-limit -1 --network host --ipc host \
  -v $HF:/root/.cache/huggingface -v $DS:/draft:ro -v $OVL:/ovl:ro -v $EV/agent:/lane:ro -v $EV/agent_prompts:/prompts \
  -v $JIT:/root/.cache/sglang -v $OUT:/out $REC \
  -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e TOKENIZERS_PARALLELISM=false -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 \
  $IMG bash -c "
    SITE=\$(python -c 'import sglang,os;print(os.path.dirname(os.path.dirname(sglang.__file__)))') &&
    cp -rf /ovl/sglang_pkg/. \$SITE/sglang/ && find \$SITE/sglang -name __pycache__ -type d -exec rm -rf {} + ;
    if [ -n \"\${LD_AUDIT:-}\" ]; then
      # SMAIX gpu-identity hook kills a process holding several libcudart copies: point them all at torch's.
      SP=\$(python -c 'import torch,os;print(os.path.dirname(os.path.dirname(torch.__file__)))')
      P=\$(realpath \$SP/nvidia/cu13/lib/libcudart.so.13)
      for c in \$SP/torchvision.libs/libcudart.*.so.13 /usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so.13.*; do ln -sf \$P \$c; done
    fi;
    exec python -m sglang.launch_server --model-path $TS --trust-remote-code $KVARG \
      --mem-fraction-static ${MEMFRAC:-0.6} --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
      --max-mamba-cache-size 160 --max-total-tokens $P --enable-int8-mamba-checkpoint --int8-mamba-ckpt-size $N \
      --attention-backend trtllm_mha --speculative-algorithm DFLASH --speculative-draft-model-path /draft \
      --speculative-num-draft-tokens 8 --speculative-draft-attention-backend trtllm_mha \
      --enable-linear-replayssm-spec --linear-replayssm-cache-len 32 \
      --decode-log-interval ${LOGINT:-40} --enable-request-time-stats-logging --enable-metrics \
      --host 127.0.0.1 --port $PORT" > $OUT/$STEM.podman.log 2>&1
ok=0
for i in $(seq 1 240); do
  podman ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo "$STEM: container died"; break; }
  podman logs $NAME 2>&1 | grep -q "Scheduler hit an exception" && { echo "$STEM: scheduler died during boot"; break; }
  curl -sf --max-time 5 http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1 && { ok=1; break; }
  sleep 10
done
podman logs $NAME > $OUT/$STEM.server.log 2>&1
[ $ok = 1 ] || exit 1
# --max-total-tokens is a cap: a budget that does not fit shrinks the pool silently, so check both pools.
got=$(grep -ao "max_total_num_tokens=[0-9]*" $OUT/$STEM.server.log | head -1 | cut -d= -f2)
gotn=$(grep -ao "int8 mamba checkpoint pool: [0-9]* slots" $OUT/$STEM.server.log | head -1 | awk '{print $5}')
if [ "$got" != "$P" ] || [ "$gotn" != "$N" ]; then echo "$STEM: POOL_MISMATCH got P=$got N=$gotn want P=$P N=$N"; exit 1; fi
grep -ah "max_total_num_tokens=\|KV Cache is allocated\|mamba checkpoint pool\|Mamba Cache is allocated\|ReplaySSM ring\|DFLASH_SHARED_KV_ONE_COPY\|DFLASH_LOAD_CHECK\|avail mem\|Memory pool end" \
  $OUT/$STEM.server.log | grep -v server_args | head -30 > $OUT/$STEM.pool.txt
podman exec $NAME python /lane/bench_agent2.py --prompt-cache /prompts --port $PORT --model-path $TS \
  --sessions $SESS --rounds $ROUNDS --system-tokens $SYSTOK --tool-tokens $TOOLTOK --output-tokens $OUTTOK --max-inflight $INFLIGHT \
  --seed ${SEED:-1} --tag $STEM --out /out/$STEM.jsonl $REPLAYARG > $OUT/$STEM.client.log 2>&1
rc=$?
gpuapps end >> $OUT/$STEM.gpuapps.txt 2>&1
OURS=" $(podman top $NAME hpid 2>/dev/null | tail -n +2 | tr '\n' ' ') "
podman logs $NAME > $OUT/$STEM.server.log 2>&1
grep -q "Scheduler hit an exception" $OUT/$STEM.server.log && echo "$STEM: scheduler died"
podman rm -f $NAME >/dev/null 2>&1
cot=$(awk -v ours="$OURS" '/^== start/{s=1;next} /^== end/{s=0;e=1;next} s&&NF{a++} e&&NF{split($0,f,","); if (index(ours, " " f[1] " ") == 0) b++} END{print (a>0||b>0)?"COTENANT":"clean"}' $OUT/$STEM.gpuapps.txt)
echo $cot > $OUT/$STEM.cot.txt
echo "$STEM: DONE rc=$rc [$(hostname -s) $S $cot] $(tail -1 $OUT/$STEM.client.log | python3 -c 'import sys,json; s=json.loads(sys.stdin.read()); print("resumed %d/%d ttft %.1f ms p90 %.1f tpot %.2f ms recomputed %d accept %.3f wall %.0f s diverged %d" % (s["resumed"], s["follow_ups"], 1e3*s["ttft_mean"], 1e3*s["ttft_p90"], 1e3*s["tpot_mean"], s["recomputed_tokens"], s["accept_mean"], s["wall"], s["diverged_from_replay"]))' 2>/dev/null || tail -c 300 $OUT/$STEM.client.log)"
