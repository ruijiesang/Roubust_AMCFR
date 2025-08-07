<h1 align="center">
  <a href="https://openreview.net/pdf?id=DDIGCk25BO" target="_blank">
    Robust Automatic Modulation Classification with Fuzzy Regularizationq
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



# Abstract
Automatic modulation classification (AMC) serves as a foundational pillar for cognitive radio systems, enabling critical functionalities including dynamic spectrum allocation, non-cooperative signal surveillance, and adaptive waveform optimization. However, practical deployment of
AMC faces a fundamental challenge: prediction ambiguity arising from intrinsic similarity among
modulation schemes and exacerbated under low signal-to-noise ratio (SNR) conditions. This phenomenon manifests as near-identical probability
distributions across confusable modulation types, significantly degrading classification reliability.
To address this, we propose Fuzzy Regularizationenhanced AMC (FR-AMC), a novel framework
that integrates uncertainty quantification into the
classification pipeline. The proposed FR has three
features: (1) Explicitly model prediction ambiguity during backpropagation, (2) dynamic sample
reweighting through adaptive loss scaling, (3) encourage margin maximization between confusable modulation clusters. Experimental results
on benchmark datasets demonstrate that the FR
achieves superior classification accuracy and robustness compared to compared methods, making
it a promising solution for real-world spectrum
management and communication applications.

# 🧱 Architecture
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
│   │   ├── log_train_info.py
│   │   ├── logger.py
│   │   ├── solver.py
│   │   ├── static.py
├── Data/
├── train_DAELSTM/
│   ├── DAELSTM_configs/
│   ├── results/
│   ├── DAELSTM_configs.py
│   ├── train.py
│   ├── train_Cen.py
├── train_FEAT/
├── train_MCLDNN/
├── train_ThreeStream/
```
# 🛠️ Previous Preparation
## 1.Clone this repository and navigate to source folder
``` 
cd VistaDPO
``` 

## 2.Build Environment
``` bash
echo "Creating conda environment"
conda create -n AMCFR python=3.10
conda activate AMCFR

echo "Installing dependencies"
pip install -r requirements.txt
``` 
## 3.Download the dataset
All of our datasets are public datasets, and you can obtain the [datasets](https://www.deepsig.ai/) you need at this location. 
Download the dataset to the "Data/" folder.


# 🚀 Quick Start
## 1.Modify the dataset address

## 2.Model configuration

## 3.Train

## 4.Checking the results


# 🙏 Acknowledgement

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





