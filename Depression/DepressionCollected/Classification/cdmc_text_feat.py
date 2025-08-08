import numpy as np
import pandas as pd
# from allennlp.commands.elmo import ElmoEmbedder
import os
prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
from elmoformanylangs import Embedder
import pkuseg
import thulac
# from pyhanlp import HanLP
import jieba
# 加载ELMo模型
print("开始加载模型...")
elmo = Embedder('/root/autodl-tmp/ICASSP2022-Depression/zhs.model')
print("模型加载完成，开始处理单个样本...")

# 特征维度（ELMo默认输出维度为1024，可根据实际模型调整）
elmo_dim = 1024
# 每4个问题聚合
group_size = 4

# 全局特征和标签列表
text_features = []
text_targets = []

def extract_text_features(subject_id, text_features, targets, path):
    """提取单个被试的文本特征并按每4个问题聚合"""
    subject_dir = os.path.join(prefix, f'{path}/{subject_id}')
    if not os.path.isdir(subject_dir):
        return
    
    # 确定标签：HC=0, MDD=1
    label = 0 if 'HC' in subject_id else 1
    
    q_features = []  # 存储该被试每个问题的文本特征
    
    # 处理12个问题文本（Q1到Q12）
    for q in range(1, 13):
        txt_path = os.path.join(subject_dir, f'Q{q}.txt')
        
        # 检查文件是否存在
        if not os.path.exists(txt_path):
            continue
            
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                
                # 跳过空文本
                if not text:
                    continue
                
                # 中文分词
                seg_list = jieba.cut(text, cut_all=False)
                tokens = [token for token in seg_list if token.strip()]
                
                if not tokens:
                    continue
                    
                # 获取ELMo嵌入并平均
                elmo_emb = elmo.sents2elmo([tokens])[0]
                avg_emb = np.array(elmo_emb).mean(axis=0).astype(np.float32)
                q_features.append(avg_emb)
                
        except Exception as e:
            continue
    
    # 至少需要4个问题特征才能聚合
    if len(q_features) < group_size:
        return
    
    # 每4个问题特征聚合（与音频处理方式对称）
    for i in range(0, len(q_features) - 3, group_size):
        combined_feat = np.concatenate(q_features[i:i+group_size], axis=-1)
        text_features.append(combined_feat)
        targets.append(label)

subjects = ['HC', 'MDD']
# for subject in subjects:
#     for i in range(1, 115):
#         subject_id = f"{subject}{i:02d}"  # 生成HC01, MDD01等格式的被试ID
#         extract_text_features(subject_id, text_features, text_targets, 'CDMC/Data')

# for subject in subjects:
#     for i in range(1, 115):
#         subject_id = f"{subject}{i:02d}"  # 生成HC01, MDD01等格式的被试ID
#         extract_text_features(subject_id, text_features, text_targets, 'CDMC/ValidationData')
# 处理测试集（与音频代码处理Test集对称）

for subject in subjects:
    for i in range(1, 115):
        subject_id = f"{subject}{i:02d}"  # 生成HC01, MDD01等格式的被试ID
        extract_text_features(subject_id, text_features, text_targets, 'CDMC/Test')

# 生成保存路径（与音频文件命名风格一致）
size_str = str(elmo_dim * group_size)  # 聚合后的特征维度
samples_path = f'Features/CDMC/TextWhole/test_samples_clf_avg.npz'
labels_path = f'Features/CDMC/TextWhole/test_labels_clf_avg.npz'

# 创建保存目录
os.makedirs(os.path.dirname(samples_path), exist_ok=True)
os.makedirs(os.path.dirname(labels_path), exist_ok=True)

# 拼接完整路径
if prefix:
    samples_path = os.path.join(prefix, samples_path)
    labels_path = os.path.join(prefix, labels_path)

# 保存特征和标签
np.savez(samples_path, text_features)
np.savez(labels_path, text_targets)

print(f"文本特征处理完成，共 {len(text_features)} 个聚合特征")