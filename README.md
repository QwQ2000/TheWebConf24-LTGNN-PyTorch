## LTGNN-PyTorch

This is the Pytorch implementation for our TheWebConf'24 paper "Linear-Time Graph Neural Networks for Scalable Recommendations" [[pdf](https://arxiv.org/pdf/2402.13973.pdf)].

This codebase was adapted from [LightGCN-pytorch](https://github.com/gusye1234/LightGCN-PyTorch). 


## Enviroment Requirement

`pip install -r requirements.txt`

## Command

` cd code && python main.py --decay=2e-4 --lr=0.0015 --layer=1 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model="ltgnn" --appnp_alpha=0.45 --num_neighbors=15 --device=0`

` cd code && python main.py --decay=2e-4 --lr=0.0015 --layer=1 --seed=2020 --dataset="alibaba-ifashion" --topks="[20]" --recdim=64 --model="ltgnn" --appnp_alpha=0.45 --num_neighbors=15 --device=0`
