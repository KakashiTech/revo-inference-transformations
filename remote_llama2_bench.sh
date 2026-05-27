#!/usr/bin/env bash
set -euo pipefail

PASS="${PASS:-}"
if [ -z "$PASS" ]; then echo "PASS env required (sudo password)"; exit 1; fi

TARGET_DIR="/home/faith/Work/proyecto REVO"
cd "$TARGET_DIR"

LOGDIR="quality/compare_remote/_logs"
mkdir -p "$LOGDIR"

sudorun() { echo "$PASS" | sudo -S -p "" "$@"; }

# Memory before
{
  echo "==== FREE BEFORE ===="
  free -h
  grep MemAvailable /proc/meminfo || true
} | tee "$LOGDIR/free_before.txt"

# Temporary swap + sysctl to survive GGUF repack peak (do this BEFORE stopping services)
SWAP_FILE="/swapfile_bench_llama2"
SWAP_SIZE_GB=16
ORIG_SWAPPINESS=$(cat /proc/sys/vm/swappiness || echo 60)
ORIG_OVERCOMMIT=$(cat /proc/sys/vm/overcommit_memory || echo 0)
{
  echo "==== SWAP / SYSCTL BEFORE ===="
  echo "swappiness=$ORIG_SWAPPINESS overcommit_memory=$ORIG_OVERCOMMIT"
} | tee "$LOGDIR/swap_sysctl_before.txt"

# Recreate swapfile with desired size and priority every run
if grep -q "$SWAP_FILE" /proc/swaps 2>/dev/null; then
  sudorun swapoff "$SWAP_FILE" || true
fi
[ -f "$SWAP_FILE" ] && sudorun rm -f "$SWAP_FILE" || true
if ! sudorun bash -lc "fallocate -l ${SWAP_SIZE_GB}G '$SWAP_FILE'"; then
  sudorun bash -lc "dd if=/dev/zero of='$SWAP_FILE' bs=1M count=0 seek=$((SWAP_SIZE_GB*1024)) status=none"
fi
sudorun chmod 600 "$SWAP_FILE" || true
sudorun mkswap "$SWAP_FILE" >/dev/null || true
sudorun swapon -p 200 "$SWAP_FILE" || sudorun swapon "$SWAP_FILE" || true

# Tune VM to prefer swap usage and allow overcommit during load
sudorun sysctl -w vm.swappiness=80 >/dev/null || true
sudorun sysctl -w vm/overcommit_memory=1 >/dev/null 2>&1 || sudorun sysctl -w vm.overcommit_memory=1 >/dev/null || true

{
  echo "==== FREE AFTER SWAP ADD ===="
  free -h
  grep MemAvailable /proc/meminfo || true
  echo "==== SWAP / SYSCTL AFTER ===="
  cat /proc/swaps || true
  echo -n "swappiness="; cat /proc/sys/vm/swappiness || true
  echo -n "overcommit_memory="; cat /proc/sys/vm/overcommit_memory || true
} | tee "$LOGDIR/free_after_swap.txt"

# Stop K3s fully to free RAM (master deactivation)
if systemctl is-active --quiet k3s; then
  sudorun systemctl stop k3s || true
fi
if systemctl list-units --type=service | grep -q k3s-agent; then
  sudorun systemctl stop k3s-agent || true
fi

# Stop docker container if present (aion-bot-direct)
if sudorun docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^aion-bot-direct$'; then
  sudorun docker stop aion-bot-direct || true
  export AION_STOPPED=1
fi

# Stop docker and containerd services if present
if systemctl list-unit-files | grep -q '^docker.service'; then
  sudorun systemctl stop docker || true
  export DOCKER_STOPPED=1
fi
if systemctl list-unit-files | grep -q '^containerd.service'; then
  sudorun systemctl stop containerd || true
  export CONTAINERD_STOPPED=1
fi

# Switch to multi-user target (disable GUI) if available
if systemctl list-unit-files | grep -q '^graphical.target'; then
  sudorun systemctl isolate multi-user.target || true
fi

sleep 3
{
  echo "==== FREE AFTER SCALE ===="
  free -h
  grep MemAvailable /proc/meminfo || true
} | tee "$LOGDIR/free_after_scale.txt"

# Additional memory cleanup: report after K3S stop and cache drop
{
  echo "==== DROP CACHES ===="
  sync; echo 3 | sudorun tee /proc/sys/vm/drop_caches >/dev/null || true
  echo "==== FREE AFTER K3S/GUI/DOCKER STOP ===="
  free -h
  grep MemAvailable /proc/meminfo || true
} | tee "$LOGDIR/free_after_k3s_stop.txt"

# Python deps (use venv to avoid externally-managed env / PEP 668)
PY=python3
VENV_DIR=".venv"
[ -d "$VENV_DIR" ] || $PY -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
$PIP -q install --upgrade pip
$PIP -q install numpy==1.26.4 llama-cpp-python==0.3.16 huggingface_hub==0.23.2



# Model download
MODEL_PATH="models/gguf/llama-2-7b.Q4_0.gguf"
mkdir -p "models/gguf"
if [ ! -s "$MODEL_PATH" ]; then
$PY - <<'PYDL'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="TheBloke/Llama-2-7B-GGUF", filename="llama-2-7b.Q4_0.gguf", local_dir="models/gguf", local_dir_use_symlinks=False)
print("downloaded", p)
PYDL
fi

# Experiments matrix (Option A tightened for 3.2Gi host)
CONTEXTS="128"
TAUS="0.96 0.98 1.00"
REPEATS=3

# Pilot run to ensure model reaches inference (tau=0.98, ctx=128, run=1)
PILOT_CTX=128
PILOT_TAU=0.98
PILOT_OUT="quality/compare_remote/ctx_${PILOT_CTX}/tau_${PILOT_TAU}/run1"
mkdir -p "$PILOT_OUT"
echo "[PILOT] ctx=$PILOT_CTX tau=$PILOT_TAU run=1 -> $PILOT_OUT"
if ! env MALLOC_ARENA_MAX=1 $PY examples/exp_4gb_llama2_runner.py --gguf "$MODEL_PATH" --outdir "$PILOT_OUT" --threads 1 --ctx "$PILOT_CTX" --prompts 10 --tau "$PILOT_TAU" --prompt-char-limit 96 >/dev/null; then
  echo "[PILOT FAILED] Could not reach inference at ctx=$PILOT_CTX tau=$PILOT_TAU" | tee "$LOGDIR/pilot_failed.txt"
  SKIP_FULL=1
else
  echo "[PILOT OK] Proceeding with full matrix" | tee "$LOGDIR/pilot_ok.txt"
fi
if [ -z "${SKIP_FULL:-}" ]; then
  for CTX in $CONTEXTS; do
    for TAU in $TAUS; do
      for R in $(seq 1 $REPEATS); do
        OUTDIR="quality/compare_remote/ctx_${CTX}/tau_${TAU}/run${R}"
        mkdir -p "$OUTDIR"
        echo "[RUN] ctx=$CTX tau=$TAU run=$R -> $OUTDIR"
        env MALLOC_ARENA_MAX=1 $PY examples/exp_4gb_llama2_runner.py --gguf "$MODEL_PATH" --outdir "$OUTDIR" --threads 1 --ctx "$CTX" --prompts 10 --tau "$TAU" --prompt-char-limit 96 >/dev/null || { echo "[FAILED] ctx=$CTX tau=$TAU run=$R" | tee "$OUTDIR/run_error.log"; continue; }
      done
    done
  done
else
  echo "[INFO] Skipping full matrix because pilot failed" | tee -a "$LOGDIR/pilot_failed.txt"
fi

# Memory after
{
  echo "==== FREE AFTER EXPERIMENTS ===="
  free -h
  grep MemAvailable /proc/meminfo || true
} | tee "$LOGDIR/free_after_experiments.txt"

# Aggregate
$PY - <<'PYAGG'
import os, json, glob, statistics as stats
base = "quality/compare_remote"
metrics = ["nll_general","nll_ood","tokens_per_s","cold_start_s","warm_avg_s","rss_peak_mb"]
agg = {"by_ctx_tau": {}, "notes": {"metrics": metrics, "variants": ["baseline","revo"]}}
for ctx_dir in sorted(glob.glob(os.path.join(base, "ctx_*"))):
    ctx = int(os.path.basename(ctx_dir).split("_")[1])
    for tau_dir in sorted(glob.glob(os.path.join(ctx_dir, "tau_*"))):
        tau = float(os.path.basename(tau_dir).split("_")[1])
        rows = {"baseline": {m: [] for m in metrics}, "revo": {m: [] for m in metrics}}
        for summ in glob.glob(os.path.join(tau_dir, "run*", "exp_4gb_llama2_summary.json")):
            try:
                with open(summ, "r", encoding="utf-8") as f:
                    S = json.load(f)
            except Exception:
                continue
            for var in S.get("variants", []):
                vname = var.get("variant","baseline")
                vkey = "revo" if vname.startswith("revo") else "baseline"
                for m in metrics:
                    if m in var and var[m] is not None:
                        rows[vkey][m].append(float(var[m]))
        def summarize(arr):
            if not arr:
                return {"n":0, "median": None, "iqr": None}
            arr_sorted = sorted(arr)
            q1 = stats.quantiles(arr_sorted, n=4)[0]
            q3 = stats.quantiles(arr_sorted, n=4)[2]
            return {"n": len(arr), "median": float(stats.median(arr_sorted)), "iqr": float(q3 - q1)}
        agg.setdefault("by_ctx_tau", {}).setdefault(str(ctx), {})[str(tau)] = {
            "baseline": {m: summarize(rows["baseline"][m]) for m in metrics},
            "revo": {m: summarize(rows["revo"][m]) for m in metrics},
        }
with open(os.path.join(base, "aggregate_summary.json"), "w", encoding="utf-8") as f:
    json.dump(agg, f, ensure_ascii=False, indent=2)
print("WROTE", os.path.join(base, "aggregate_summary.json"))
PYAGG


# Restart K3s
if systemctl list-unit-files | grep -q '^k3s.service'; then
  sudorun systemctl start k3s || true
fi

# Restore docker container
if [ "${AION_STOPPED:-}" = "1" ]; then
  sudorun docker start aion-bot-direct || true
fi

# Restart docker and containerd if they were stopped
if [ "${CONTAINERD_STOPPED:-}" = "1" ]; then
  sudorun systemctl start containerd || true
fi
if [ "${DOCKER_STOPPED:-}" = "1" ]; then
  sudorun systemctl start docker || true
fi

# Switch back to GUI if available
if systemctl list-unit-files | grep -q '^graphical.target'; then
  sudorun systemctl isolate graphical.target || true
fi

# Restore sysctl and remove temporary swap
sudorun sysctl -w vm.swappiness="$ORIG_SWAPPINESS" >/dev/null || true
sudorun sysctl -w vm/overcommit_memory="$ORIG_OVERCOMMIT" >/dev/null 2>&1 || sudorun sysctl -w vm.overcommit_memory="$ORIG_OVERCOMMIT" >/dev/null || true
if grep -q "$SWAP_FILE" /proc/swaps 2>/dev/null; then
  sudorun swapoff "$SWAP_FILE" || true
fi
sudorun rm -f "$SWAP_FILE" || true

echo "DONE"
