# Setting up RPG-RL (SkyRL + GPU) on a new server

Goal: reproduce the training/eval box on a fresh multi-GPU server. Almost everything is public or
regenerable — you only need to *receive* two things out-of-band: **credentials** and access to the
**personal_docs** repo (launch scripts). Assumes ~8× GPUs (L40S/A100/H100), NVIDIA driver + Docker +
nvidia-container-runtime, and a large data volume (the working dir grows to ~60 GB:
SkyRL/.venv ~17 GB, hf_cache ~34 GB, data + checkpoints).

## 0. Why Docker (not optional)
Bare-metal `uv` fails on Amazon Linux 2023 (glibc 2.34): SkyRL pins a `vllm-router` wheel that needs
glibc ≥ 2.35. The published image provides it. So everything runs **inside the container**.

## 1. Lay out a working dir (this becomes `/work` in the container)
Pick a path on the big volume, e.g. `~/rpgrl`. Under it:
```
rpgrl/
  ADS_shared/     # you already cloned this  (git@github.com:vivianc2/ADS_shared.git, branch rpg)
  SkyRL/          # step 2
  hf_cache/       # step 5 (weights)
  data/           # created by the dataset build (step 6)
  logs/  rl_ckpt/ # created at runtime
  key.txt  wandb_key.txt   # step 4 (from Vivian, do NOT commit)
```

## 2. Clone SkyRL (public) at the pinned commit
```
cd ~/rpgrl
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL && git checkout bce9ee9a        # the commit this box uses (for dep/behavior parity)
```

## 3. Recreate the RPG launcher symlink
The SkyRL RPG example is a symlink into ADS_shared (so it's already in your clone):
```
ln -s ~/rpgrl/ADS_shared/dataset_generation_code/skyrl_rpg ~/rpgrl/SkyRL/examples/train/rpg
```

## 4. Credentials (get from Vivian, securely — never commit)
- `key.txt` — exports `AWS_BEARER_TOKEN_BEDROCK`, `AWS_DEFAULT_REGION`, `NAUTILUS_API_KEY`
  (only needed for the free-text API *eval*; the GPU RL run does not need these).
- `wandb_key.txt` — W&B logging. Use your **own** W&B account/key if you prefer; edit the project in
  the launch script. HuggingFace weights are public (no token needed).

## 5. Pull the image + start the container
```
docker pull novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8
docker run -d --name skyrl --runtime=nvidia --gpus all --shm-size=16g \
  -v ~/rpgrl:/work  novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8  sleep infinity
```
Everything below runs inside: `docker exec -it skyrl bash`.

## 6. Install SkyRL into an isolated uv venv (inside the container)
```
cd /work/SkyRL
uv sync --extra fsdp
uv pip install scipy pandas          # oracle needs scipy/pandas
uv pip install boto3                 # only if you'll run the Bedrock/Opus API eval
```
This creates `/work/SkyRL/.venv`. Sanity: `/work/SkyRL/.venv/bin/python -c "import vllm, torch; print(torch.cuda.device_count())"` should print your GPU count.

## 7. Weights (public — download, don't copy 34 GB)
```
docker exec skyrl bash -lc 'HF_HOME=/work/hf_cache huggingface-cli download Qwen/Qwen3.5-9B'
docker exec skyrl bash -lc 'HF_HOME=/work/hf_cache huggingface-cli download Qwen/Qwen3-8B'   # if needed
```

## 8. Dataset (already in your clone; or regenerate — deterministic)
The canonical de-leaked v9 set is committed at
`ADS_shared/dataset_generation_code/rpg_v9/data_v9_deleaked/{train,validation}.parquet`. To (re)build
more (≈20–25 min, CPU):
```
docker exec skyrl bash -lc '
  cd /work/SkyRL
  export RPG_SRC=/work/ADS_shared/dataset_generation_code RPG_PROTO=rpg_v9 PYTHONUNBUFFERED=1
  uv run --isolated python -m examples.train.rpg.rpg_dataset \
    --output_dir /work/data/rpg_v9_deleaked --train_size 1536 --val_size 128 \
    --train_seed0 10000000 --val_seed0 20000000'
```

## 9. Launch a run (scripts live in the personal_docs repo)
The launch scripts (`run9b_launch.sh`, `run27b_launch.sh`) and setup notes (`rl_run6_setup.md`,
`rpg_v7_rl_infra_decision.md`) are in **git@github.com:vivianc2/personal_docs.git** — ask Vivian for
access, or she can send the two scripts. Before launching: `docker exec skyrl bash -lc 'ray stop --force'`
and confirm all GPUs at 0 MiB. The 9B script points `DATA_DIR=/work/data/rpg_v9_deleaked` and carries
the required Qwen3.5 vLLM flags (`gpu_memory_utilization=0.8`, `max_num_seqs=512`).
**Verify after launch:** step-0 eval `stop_reason=stop` (no truncation), no OOM.

## What Vivian needs to send you (that's it)
1. `key.txt` + `wandb_key.txt` (secure channel).
2. Access to `personal_docs` (launch scripts) — or the two launch scripts directly.
Everything else = `docker pull` + `git clone` + `huggingface-cli download` + the ADS_shared clone you have.
```
```
