"""
Simple script to attempt to load a Tarflow model based off some pretrained weights.
"""

import transformer_flow
import torch
import ipdb
from pathlib import Path

if __name__ == "__main__":

    # torchrun --standalone --nproc_per_node=8 train.py --dataset=afhq --img_size=256 --channel_size=3\
    #   --patch_size=8 --channels=768 --blocks=8 --layers_per_block=8\
    #   --noise_std=0.07 --batch_size=256 --epochs=4000 --lr=1e-4 --nvp --cfg=0 --drop_label=0.1\
    #   --sample_freq=200 --logdir=runs/afhq256


    model = transformer_flow.Model(
        in_channels=3,
        img_size=256,
        patch_size=8,
        channels=768,
        num_blocks=8,
        layers_per_block=8,
        nvp=True,
        num_classes=3,
    ).to('cuda')


    ipdb.set_trace()
    afhq_weighs_path: Path = Path("/data/davidn/PhXD/GeometricDeepLearning/ml-tarflow/Transformer_Autoregressive_Flow/afhq_model_8_768_8_8_0.07.pth")
    weights: dict[str, torch.Tensor] = torch.load(afhq_weighs_path, map_location='cpu')

    model.load_state_dict(weights)

    model.forward(torch.randn(1, 3, 256, 256)).to('cuda')