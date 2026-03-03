#!/usr/bin/bash


~/miniconda3/envs/fresh_ml_sdf/bin/python3 -m torch.distributed.run --standalone --nproc_per_node=4 prepare_fid_stats.py --dataset afhq --data /data/davidn/PhXD/GeometricDeepLearning/RiemannEBM_PLightning/afhq_download_repo/data \
    --batch_size=128 \
    --img_size=128 --channel_size=3

    # parser.add_argument('--data', default='data', type=pathlib.Path, help='Path for training data')
    # parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'imagenet64', 'afhq'], help='Name of dataset')
    # parser.add_argument('--img_size', default=32, type=int, help='Image size')
    # parser.add_argument('--channel_size', default=3, type=int, help='Image channel size')
    # parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')

# ~/miniconda3/envs/fresh_ml_sdf/bin/python3 -m torch.distributed.run --standalone --nproc_per_node=4 train.py --dataset=afhq --img_size=128 --channel_size=3\
#   --patch_size=4 --channels=384 --blocks=4 --layers_per_block=8\
#   --noise_std=0.07 --batch_size=32 --epochs=4000 --lr=1e-4 --nvp --cfg=0 --drop_label=0.1\
#   --sample_freq=200 --logdir=runs/afhq256_small --data /data/davidn/PhXD/GeometricDeepLearning/RiemannEBM_PLightning/afhq_download_repo/data