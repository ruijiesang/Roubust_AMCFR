<h1 align="center">
  <a href="https://openreview.net/pdf?id=DDIGCk25BO" target="_blank">
    VistaDPO: Video Hierarchical Spatial-Temporal Direct Preference Optimization for Large Video Models
  </a>
</h1>

<p align="center">
  <a href="https://xinyanliang.github.io/"><u>Xinyan Liang</u></a><sup>1</sup>, 
  <u>Ruijie Sang</u><sup>1</sup>, 
  <a href="https://dig.sxu.edu.cn/qyh/"><u>Yuhua Qian</u></a><sup>1</sup>, 
  <u>Qian Guo</u><sup>2</sup>, 
  <u>Feijiang Li</u><sup>1</sup>,
  <u>Liang Du</u><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Shanxi University, <sup>2</sup>Taiyuan University of Science and Technology
</p>

<p align="center">
  <a href="https://openreview.net/pdf?id=DDIGCk25BO">
    <img src="https://img.shields.io/badge/OpenReview-gray?style=flat&logo=OpenReview">
  </a>
</p>



# Citation
```
@ARTICLE{10458301,
  author={Zeng, Rui and Lu, Zhilin and Zhang, Xudong and Wang, Jintao and Wang, Jian},
  journal={IEEE Signal Processing Letters}, 
  title={Convolutional Neural Network Assisted Transformer for Automatic Modulation Recognition Under Large CFOs and SROs}, 
  year={2024},
  volume={31},
  number={},
  pages={741-745},
  keywords={Transformers;Convolution;Modulation;Feature extraction;Task analysis;Receivers;Frequency modulation;Automatic modulation recognition;carrier frequency offsets;sample rate offsets;transformer;group convolution},
  doi={10.1109/LSP.2024.3372770}}

```


# Requirements
```
pytorch
yacs
h5py
matplotlib
thop  
```

# Architecture
``` 
home
├── amr/
│   ├── dataloaders/
│   ├── models/
│   │   ├── losses/
│   │   ├── networks/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── draw.py
│   │   ├── init.py
│   │   ├── logger.py
│   │   ├── solver.py
│   │   ├── static.py
├── configs/  (hyperparameters)
│   ├── *.yaml
├── main.py
├── datasets/
├── results/
```

# Quick Start
If you want to train a network from scratch, please follow these steps:
1. preparing dataset: download the dataset with large-scale offsets [dataset.rar](https://drive.google.com/file/d/1xZa9GcZoIZXstkwNd4E68Wbq7DdFN-a5/view?usp=sharing), and form the file path like 'dataset/cfo.hdf5'

2. training and testing: run `python main.py --config xxx`. e.g.`python main.py --config configs/transgroupnet_cfo.yaml`

3. checking the results: check the well-trained models and the figures in `results/`

# Result Reproduction
1. preparing dataset: download the dataset with large-scale offsets [dataset.rar](https://drive.google.com/file/d/1xZa9GcZoIZXstkwNd4E68Wbq7DdFN-a5/view?usp=sharing), and form the file path like 'dataset/cfo.hdf5'

2. preparing models: download the prepared models and results in [results.rar](https://drive.google.com/file/d/1MiHnfB_F25c0yTIHt52JuXWQ27r4sFYH/view?usp=sharing), and extract to the current path

3. modifying settings: change the state of `train` From `True` to `False` in `configs/transgroupnet_cfo.yaml`, and run `python main.py --config configs/transgroupnet_cfo.yaml` to get the expermential results.

    e.g.
    ```yaml
    method: 'ours'
    train: False  # change the state: from True to False
    dataset: 'cfo'
    mod_type: ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM32", "QAM64", "QAM128", "QAM256", "GFSK", "WBFM", "AM-DSB", "AM-SSB", "OOK", "4ASK", "8ASK", "16PSK", "32PSK","8APSK","GMSK", "DQPSK","16APSK","32APSK","64APSK","128APSK"]
    workers: 8
    seed: 1
    gpu: 0
    cpu: False
    params:
        "network": "TransGroupNet"
        "loss": "loss_CE_test2"
        "batch_size": 1024
        "epochs": 200
        "lr": 5e-3
        "lr_decay": 0.8
        "weight_decay": 5e-2
        "early_stop": False
        "Xmode": [{"type":"APF","options":{"IQ_norm":False, "zero_mask":False}}]
    ```





