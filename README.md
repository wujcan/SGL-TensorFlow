# SGL
This is our Tensorflow implementation for our SIGIR 2021 paper:

>Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian,and Xing Xie. 2021. Self-supervised Graph Learning for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2010.10783).

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

Further details, please refer to [NeuRec](https://github.com/wubinzzu/NeuRec/)

**Secondly**, specify dataset and recommender in configuration file *NeuRec.properties*.

Model specific hyperparameters are in configuration file *./conf/SGL.properties*.

Some important hyperparameters (taking a 3-layer SGL-ED as example):

### yelp2018 dataset
```
aug_tyep=1
reg=1e-4
embed_size=64
n_layers=3
ssl_reg=0.1
ssl_ratio=0.1
ssl_temp=0.2
```

### amazon-book dataset
```
aug_tyep=1
reg=1e-4
embed_size=64
n_layers=3
ssl_reg=0.5
ssl_ratio=0.1
ssl_temp=0.2
```

### ifashion dataset
```
aug_tyep=1
reg=1e-3
embed_size=64
n_layers=3
ssl_reg=0.02
ssl_ratio=0.4
ssl_temp=0.5
```


**Finally**, run [main.py](./main.py) in IDE or with command line:

```bash
python main.py
```

