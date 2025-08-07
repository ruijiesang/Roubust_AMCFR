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
``` bash
cd AMCFR
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

Navigate to the 'amr/dataloaders/' directory and select the corresponding dataset loader file (e.g., 'dataloader_2016aData.py').
Locate the 'data_path' variable and update its value to the path where your dataset is stored on your local machine.
```
class SignalDataLoader(object):
    def __init__(self, mod_type=[],snr_type=[],scal=0):
        mods = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16','QAM64', 'QPSK', 'WBFM']
        
        data_path=r'/home/sangruijie_qyh/Data/Signal/RML2016.10a_dict.pkl'
       
        data = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')
        data_keys=data.keys()
```
## 2.Model configuration
选择一个你想训练的模型，例如"DAELSTM"模型，进入模型对应的子文件夹"train_DAELSEM/DAELSTM_configs"进行参数配置。DAELSTM_train.yaml文件对应train.py文件；DAELSTM_train_Cen.yaml文件对应train_Cen.py文件。（此处我们保留两个训练脚本是因为FR损失涉及参数，而其余部分损失不涉及参数选择。）
### (1) 修改DAELSTM_config.py文件
将配置文件"DAELSTM_train.yaml"地址更换为该文件在您设备的实际地址。
```
def get_cfgs():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='/home/sangruijie_qyh/Code/TransGroupNet-FR/Roubust_AMCFR/train_DAELSTM/DAELSTM_configs/DAELSTM_train.yaml',
                        help='type of config file. e.g. resnet_cfo (Resnet_configs/resnet_cfo.yaml)')
    args = parser.parse_args()
    cfgs.merge_from_file(args.config)
    return cfgs
```
### (2) 修改"DAELSTM_train.yaml"文件
在该文件中methon对应所用模型架构（"./amr/models/networks/DAELSTM"）,network表示具体网络。（"DAELSTM.py" or 
"DAELSTM_1024.py");train表示进行训练或是只进行测试；scal控制是否添加噪声；
```yaml
method: 'DAELSTM'
train: True
dataset: '2016aData'
mod_type: ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16','QAM64', 'QPSK', 'WBFM']
snr_type: [0,2,4,6,8,10,12,14,16,18]
scal : 0    #0,0.2,0.4,0.6
workers: 8
seed: 1
gpu: 7
cpu: False
params:
    "network": "DAELSTM"
    "loss": "loss_FG"
    "loss_c": 2
    "loss_t": 2
    "loss_alp": 0.01
    "batch_size": 128
    "epochs": 200
    "lr": 5e-3
    "lr_decay": 0.
    "weight_decay": 0.
    "early_stop": False
    "Xmode": [{"type":"AP","options":{"IQ_norm":False, "zero_mask":True}}]
```
### (3)修改寻参范围
假如您是一个k分类问题，c的取值就是[1,k],t表示FR损失和交叉熵损失之间的权重。更多参数细节可见论文4.6节。
```
if __name__ == '__main__':
    c = [5]
    t = [ 0.0001 ]
    best_acc=0
    for c1 in c:
        for t1 in t:
            cfgs = get_cfgs()
            main(cfgs,c1,t1)
```

## 3.Train
### 运行带有参数的损失
```bash
cd './AMCFR/train_DAELSTM'
python train.py
```
### 运行无参数损失
```bash
cd './AMCFR/train_DAELSTM'
python train_Cen.py
```
## 4.Checking the results
在代码运行完毕后，程序会自动生成一个"results/"文件夹，保存最优模型参数、训练信息以及混淆矩阵。
```
├── train_DAELSTM/
│   ├── DAELSTM_configs/
│   ├── results/
│   │   ├── DAELSTM/
│   │   │   ├── DatasetName/
│   │   │   │   ├── lossName/
│   │   │   │   │   ├── paramers/
│   │   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   ├── draws/
│   │   │   │   │   │   ├── train_info.csv
│   │   │   │   │   ├── test_info.csv
```
# 🔄 Select other models or datasets
## Change the dataset
如果你要使用一个新的数据集

(1) 将数据集下载到指定文件夹中。

(2) 在'./Roubust_AMCFR/amr/dataloaders'中创建一个新的数据处理文件"dataloader_Newdataset",并将输出端口与已有输出端口保持一致，例如"dataloader_2016aData.py"文件。

(3) 对需要运行模型的config文件进行修改，例如"./Roubust_AMCFR/train_DAELSTM/DAELSTM_configs/DAELSTM_train.yaml"文件中，将"dataset:"设置为'Newdataset'.

## Change the model
如果你要使用一个新的模型

(1) 将模型保存在"./Roubust_AMCFR/amr/models/networks"文件夹中。

(2) 创建"train_Newmodel"子文件夹，类似"train_DAELSTM"文件夹。

(3) 修改"model_config.py"文件，以及"model_train.yaml"文件.(例如"DAELSTM_config.py"文件和"DAELSTM_train.yaml"文件)

# 📝 Citation
Please consider citing our paper if our code and benchmark are useful:

# 🙏 Acknowledgement

# 📪 Contact
For any question, feel free to email <span style="background-color:#f2f2f2; border-radius:6px; padding:2px 6px; font-family:monospace;">
  sangrj66@163.com
</span>




