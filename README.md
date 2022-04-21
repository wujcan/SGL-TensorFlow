# SGL
This is our Tensorflow implementation for our SIGIR 2021 paper. We also provide PyTorch implementation for SGL: https://github.com/wujcan/SGL-Torch.

>Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian,and Xing Xie. 2021. Self-supervised Graph Learning for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2010.10783).

This project is based on [NeuRec](https://github.com/wubinzzu/NeuRec/). Thanks to the contributors.

## Environment Requirement

The code runs well under python 3.7.7. The required packages are as follows:

- Tensorflow-gpu == 1.15.0
- numpy == 1.19.1
- scipy == 1.5.2
- pandas == 1.1.1
- cython == 0.29.21

## Quick Start
**Firstly**, compline the evaluator of cpp implementation with the following command line:

```bash
python setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

**Note that the cpp implementation is much faster than python.**

**Secondly**, specify dataset and recommender in configuration file *NeuRec.properties*.

Model specific hyperparameters are in configuration file *./conf/SGL.properties*.

Some important hyperparameters (taking a 3-layer SGL-ED as example):

### yelp2018 dataset
```
aug_type=1
reg=1e-4
embed_size=64
n_layers=3
ssl_reg=0.1
ssl_ratio=0.1
ssl_temp=0.2
```

### amazon-book dataset
```
aug_type=1
reg=1e-4
embed_size=64
n_layers=3
ssl_reg=0.5
ssl_ratio=0.1
ssl_temp=0.2
```

### ifashion dataset
```
aug_type=1
reg=1e-3
embed_size=64
n_layers=3
ssl_reg=0.02
ssl_ratio=0.4
ssl_temp=0.5
```


**Finally**, run [main.py](./main.py) in IDE or with command line:

```bash
python main.py --recommender=SGL
```

## About Long-tail Recommendation
To re-implement Figure 4 (Group-wise Long-tail Recommendation), first pretrain the model and save the user/item embeddings to file by additionally setting
```
save_flag=1
pretrain=0
```
then re-run the code by changing
```
save_flag=0
pretrain=1
```



## About Robustness to Noisy Interactions
To re-implement Figure 6 (Model performance wrt. noise ratio), first run [add_noise.py](./add_noise.py) to generate the contaminated training data, for example,
```bash
python add_noise.py --data.input.dataset=yelp2018 --ratio=0.05
```
then run [main.py](./main.py), for example,
```bash
python main.py --recommender=SGL --data.input.dataset=yelp2018_0.05
```