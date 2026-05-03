#!/usr/bin/env bash
# Phase 0 smoke test for the vllm-port project.
# Validates: torch+vllm import, single-GPU inference, TP=2 NCCL on this box.
# Run from the vllm repo root with .venv already created and active or present.
#   bash port-scripts/01_smoke.sh
# Auto-falls back to NCCL_P2P_DISABLE=1 on the TP=2 test if the first attempt
# hangs (the common WSL2 failure mode).

set -u  # don't `set -e` — we want every test to run even if an earlier one fails

cd "$(dirname "$0")/.."  # repo root, regardless of where the script was invoked from

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv/bin/python not found. Are you in the vllm repo root and is the venv created?"
    exit 1
fi

PY=.venv/bin/python

cat > /tmp/t1_env.py <<'PYEOF'
import torch, vllm
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'gpu{i}: {p.name} sm_{p.major}{p.minor} {p.total_memory/1e9:.1f}GB {p.multi_processor_count}SMs')
print('vllm:', vllm.__version__)
PYEOF

cat > /tmp/t2_single.py <<'PYEOF'
# `if __name__ == '__main__'` is required: vLLM forces spawn start method on
# WSL (NVML is not fork-safe), so without the guard each spawned worker
# re-imports the script and tries to construct another LLM, recursively.
if __name__ == '__main__':
    from vllm import LLM, SamplingParams
    llm = LLM(model='facebook/opt-125m', max_model_len=256,
              gpu_memory_utilization=0.3, enforce_eager=True)
    out = llm.generate(['Hello, world. The capital of France is'],
                       SamplingParams(max_tokens=8, temperature=0))
    print('SINGLE_GPU_OK:', repr(out[0].outputs[0].text))
PYEOF

cat > /tmp/t3_tp2.py <<'PYEOF'
if __name__ == '__main__':
    from vllm import LLM, SamplingParams
    llm = LLM(model='facebook/opt-125m', max_model_len=256,
              gpu_memory_utilization=0.3, enforce_eager=True,
              tensor_parallel_size=2)
    out = llm.generate(['Hello, world. The capital of France is'],
                       SamplingParams(max_tokens=8, temperature=0))
    print('TP2_OK:', repr(out[0].outputs[0].text))
PYEOF

echo '======== T1: ENV CHECK ========'
"$PY" /tmp/t1_env.py
T1=$?

echo
echo '======== T2: SINGLE-GPU SMOKE (downloads ~250MB on first run) ========'
"$PY" /tmp/t2_single.py
T2=$?

echo
echo '======== T3: TP=2 SMOKE (120s timeout, NCCL_DEBUG=WARN) ========'
NCCL_DEBUG=WARN timeout 120 "$PY" /tmp/t3_tp2.py
T3=$?
T3R=
if [[ $T3 -eq 124 ]]; then
    echo
    echo '!! T3 hit 120s timeout — likely NCCL p2p stuck on WSL2.'
    echo '!! Retrying with NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 ...'
    echo
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=WARN \
        timeout 120 "$PY" /tmp/t3_tp2.py
    T3R=$?
fi

echo
echo '======== SUMMARY ========'
echo "T1 env-check       : exit $T1"
echo "T2 single-GPU      : exit $T2"
echo "T3 tp=2            : exit $T3"
[[ -n "$T3R" ]] && echo "T3 tp=2 (P2P off)  : exit $T3R"
echo 'DONE'
