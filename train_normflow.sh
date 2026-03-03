#!/usr/bin/bash
# Wrapper script for launching a training run for the normalizing flow model.

pythonEnv=~/miniconda3/envs/fresh_ml_sdf/bin/python3
dataDir=/data/davidn/PhXD/GeometricDeepLearning/RiemannEBM_PLightning/afhq_download_repo/data

$pythonEnv -m torch.distributed.run --standalone --nproc_per_node=4 train.py  \
  --dataset=afhq --img_size=128 --channel_size=3\
  --patch_size=4 --channels=384 --blocks=4 --layers_per_block=8\
  --noise_std=0.07 --batch_size=64 --epochs=4000 --lr=1e-4 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=200 --logdir=runs/afhq256_small --data $dataDir