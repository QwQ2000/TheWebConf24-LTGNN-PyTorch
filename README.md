## LTGNN-PyTorch

This is the Pytorch implementation for our TheWebConf'24 paper "**Linear-Time Graph Neural Networks for Scalable Recommendations**". Please find our paper in
ACM Digital Library (https://dl.acm.org/doi/10.1145/3589334.3645486) or arXiv (https://arxiv.org/abs/2402.13973).

This codebase was adapted from [LightGCN-pytorch](https://github.com/gusye1234/LightGCN-PyTorch). 


## Enviroment Requirement

`pip install -r requirements.txt`

## Command

` cd code && python main.py --decay=2e-4 --lr=0.0015 --layer=1 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model="ltgnn" --appnp_alpha=0.45 --num_neighbors=15 --device=0`

` cd code && python main.py --decay=2e-4 --lr=0.0015 --layer=1 --seed=2020 --dataset="alibaba-ifashion" --topks="[20]" --recdim=64 --model="ltgnn" --appnp_alpha=0.45 --num_neighbors=15 --device=0`


## BibTeX
If you find **LTGNN** useful in your research, please cite the following in your manuscript:


```
@inproceedings{zhang2024linear,
  title={Linear-Time Graph Neural Networks for Scalable Recommendations},
  author={Zhang, Jiahao and Xue, Rui and Fan, Wenqi and Xu, Xin and Li, Qing and Pei, Jian and Liu, Xiaorui},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages = {3533-3544},
  year={2024}
}
```
