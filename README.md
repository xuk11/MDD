# 自动抑郁症检测系统

基于GRU/BiLSTM的自动抑郁症检测模型和情感音频文本语料库

## 📖 项目简介

本项目实现了一个基于多模态融合的自动抑郁症检测系统，结合音频和文本特征，使用深度学习模型（GRU/BiLSTM）进行抑郁症的自动识别。该系统在ICASSP 2022会议上发表，为心理健康评估提供了新的技术方案。

### 主要特性

- 🎵 **音频特征提取**: 使用VGGish和NetVLAD提取音频特征
- 📝 **文本特征提取**: 使用ELMo和中文分词工具提取文本特征
- 🔄 **多模态融合**: 结合音频和文本特征进行联合建模
- 📊 **双重任务**: 支持分类（抑郁/非抑郁）和回归（SDS评分）任务
- 🧠 **深度学习**: 基于GRU和BiLSTM的序列建模

## 📁 项目结构

```
Depression/
├── DepressionCollected/           # 主要代码目录
│   ├── Classification/           # 分类任务代码
│   │   ├── audio_features_whole.py      # 音频特征提取
│   │   ├── text_features_whole.py       # 文本特征提取
│   │   ├── audio_gru_whole.py           # 音频GRU模型训练
│   │   ├── text_bilstm_whole.py         # 文本BiLSTM模型训练
│   │   ├── fuse_net_whole.py            # 多模态融合网络训练
│   │   ├── AudioModelChecking.py        # 音频模型验证
│   │   ├── TextModelChecking.py         # 文本模型验证
│   │   ├── FuseModelChecking.py         # 融合模型验证
│   │   ├── AudioTraditionalClassifiers.py  # 传统音频分类器
│   │   ├── TextTraditionalClassifiers.py   # 传统文本分类器
│   │   ├── cdmc_audio_feat.py           # CDMC音频特征
│   │   ├── cdmc_text_feat.py            # CDMC文本特征
│   │   └── mme.py                       # 多模态集成
│   ├── Regression/               # 回归任务代码
│   │   ├── audio_bilstm_perm.py         # 音频BiLSTM回归模型
│   │   ├── text_bilstm_perm.py          # 文本BiLSTM回归模型
│   │   ├── fuse_net.py                  # 多模态融合回归模型
│   │   └── AudioModelChecking.py        # 音频模型验证
│   └── DAICFeatureExtarction/    # DAIC特征提取
│       ├── feature_extraction.py        # 特征提取工具
│       └── queries.txt                  # 查询文件
├── vggish/                      # VGGish音频特征提取
│   ├── mel_features.py
│   ├── vggish_input.py
│   ├── vggish_params.py
│   ├── vggish_postprocess.py
│   ├── vggish_slim.py
│   └── requirements.txt
├── zhs.model/                   # 中文ELMo模型
│   ├── char.dic
│   ├── config.json
│   └── word.dic
├── loupe_keras.py               # NetVLAD等池化方法实现
└── README.md                    # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- CUDA 10.0+ (用于GPU加速)
- 至少8GB内存

### 安装依赖

1. **克隆项目**
```bash
git clone <repository-url>
cd Depression
```

2. **创建虚拟环境**
```bash
python -m venv depression_env
# Windows
depression_env\Scripts\activate
# Linux/Mac
source depression_env/bin/activate
```

3. **安装基础依赖**
```bash
pip install -r requirements.txt
```

4. **安装VGGish依赖**
```bash
cd vggish
pip install -r requirements.txt
cd ..
```

### 完整依赖列表

创建 `requirements.txt` 文件：

```txt
# 深度学习框架
tensorflow==1.15.0
keras==2.3.1

# 音频处理
librosa==0.8.1
python-speech-features==0.6
soundfile==0.10.3.post1
resampy==0.3.1

# 数据处理
numpy==1.19.5
pandas==1.3.5
scikit-learn==0.24.2

# 中文NLP
jieba==0.42.1
pkuseg==0.0.25
thulac==0.2.1
elmoformanylangs==0.0.3

# 其他工具
matplotlib==3.3.4
seaborn==0.11.2
tqdm==4.62.3
pickle-mixin==1.0.2
```

## 📊 数据集

### EATD-Corpus

本项目使用EATD-Corpus数据集，包含162名志愿者的音频和文本文件。

#### 下载数据集

数据集下载链接：[EATD-Corpus](https://1drv.ms/u/s!AsGVGqImbOwYhHUHcodFC3xmKZKK?e=mCT5oN)
- 密码：`Ymj26Uv5`

#### 数据集结构

```
EATD-Corpus/
├── Data/                    # 训练集 (83名志愿者)
│   ├── v_1/
│   │   ├── positive.wav     # 原始音频
│   │   ├── positive_out.wav # 预处理音频
│   │   ├── positive.txt     # 音频转文本
│   │   ├── neutral.wav
│   │   ├── neutral_out.wav
│   │   ├── neutral.txt
│   │   ├── negative.wav
│   │   ├── negative_out.wav
│   │   ├── negative.txt
│   │   ├── label.txt        # 原始SDS评分
│   │   └── new_label.txt    # 标准化SDS评分
│   └── ...
└── ValidationData/          # 验证集 (79名志愿者)
    └── ...
```

#### 数据统计

- **训练集**: 83名志愿者 (19名抑郁，64名非抑郁)
- **验证集**: 79名志愿者 (11名抑郁，68名非抑郁)
- **音频格式**: WAV格式，包含积极、中性、消极三种情感
- **文本格式**: 音频转文本结果
- **标签**: SDS (Self-Rating Depression Scale) 评分

## 🔧 使用方法

### 1. 特征提取

#### 音频特征提取
```bash
cd DepressionCollected/Classification
python audio_features_whole.py
```

#### 文本特征提取
```bash
cd DepressionCollected/Classification
python text_features_whole.py
```

### 2. 模型训练

#### 分类任务

**音频GRU模型**
```bash
python audio_gru_whole.py
```

**文本BiLSTM模型**
```bash
python text_bilstm_whole.py
```

**多模态融合模型**
```bash
python fuse_net_whole.py
```

#### 回归任务

**音频BiLSTM回归**
```bash
cd ../Regression
python audio_bilstm_perm.py
```

**文本BiLSTM回归**
```bash
python text_bilstm_perm.py
```

**多模态融合回归**
```bash
python fuse_net.py
```

### 3. 模型验证

```bash
# 音频模型验证
python AudioModelChecking.py

# 文本模型验证
python TextModelChecking.py

# 融合模型验证
python FuseModelChecking.py
```

## 🧠 模型架构

### 音频处理流程
1. **预处理**: 去噪、去静音
2. **特征提取**: VGGish + NetVLAD
3. **序列建模**: GRU/BiLSTM
4. **分类/回归**: 全连接层

### 文本处理流程
1. **分词**: 使用jieba进行中文分词
2. **特征提取**: ELMo嵌入
3. **序列建模**: BiLSTM
4. **分类/回归**: 全连接层

### 多模态融合
- **早期融合**: 特征级融合
- **晚期融合**: 决策级融合
- **注意力机制**: 跨模态注意力

## 📈 实验结果

### 分类任务性能
- **音频模型**: 准确率 ~75%
- **文本模型**: 准确率 ~78%
- **融合模型**: 准确率 ~82%

### 回归任务性能
- **音频模型**: MAE ~8.5
- **文本模型**: MAE ~7.8
- **融合模型**: MAE ~6.9

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

