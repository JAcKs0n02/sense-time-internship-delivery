cd /root/autodl-tmp/week8-distillation-gpu-20260918
nohup timeout --signal=KILL 2200s env -u USE_TORCH OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 TOKENIZERS_PARALLELISM=false /root/autodl-tmp/conda/envs/llm_exp/bin/python -u run_generation_session.py > coordinator.log 2>&1 < /dev/null &
