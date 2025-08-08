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
# seg = pkuseg.pkuseg()
# thu1 = thulac.thulac(seg_only=True)
print("开始加载模型...")
elmo = Embedder('/root/autodl-tmp/ICASSP2022-Depression/zhs.model')
print("模型加载完成，开始处理单个样本...")

topics = ['positive', 'neutral', 'negative']
answers = {}
text_features = []
text_targets = []

def extract_features(text_features, text_targets, path):
    for index in range(114):
        # print(os.path.join(prefix, path, str(index+1)))
        if os.path.isdir(os.path.join(prefix, path, f"v_{index+1}")):
            answers[index+1] = []
            for topic in topics:
                with open(os.path.join(prefix, path, f"v_{index+1}", '%s.txt'%(topic)) ,'r') as f:
                    lines = f.readlines()[0]
                    # seg_text = seg.cut(lines) 
                    # seg_text = thu1.cut(lines)
                    # seg_text_iter = HanLP.segment(lines) 
                    seg_text_iter = jieba.cut(lines, cut_all=False) 
                    answers[index+1].append([item for item in seg_text_iter])
                    print("Loaded text from %s/%s/%s.txt"%(path, f"v_{index+1}", topic))
                    # answers[dir].append(seg_text)
            with open(os.path.join(prefix, '{1}/v_{0}/new_label.txt'.format(index+1, path))) as fli:
                target = float(fli.readline())
            text_targets.append(1 if target >= 53 else 0)
            # text_targets.append(target)
            text_features.append([np.array(item).mean(axis=0) for item in elmo.sents2elmo(answers[index+1])])

# extract_features(text_features, text_targets, 'ETAD/Data')
# extract_features(text_features, text_targets, 'ETAD/ValidationData')
extract_features(text_features, text_targets, 'ETAD/Test')

print("Saving npz file locally...")

# Define the directory where you want to save the files
save_dir = os.path.join(prefix, 'Features/ETAD/TextWhole')

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Now save the files
np.savez(os.path.join(save_dir, 'test_samples_clf_avg.npz'), text_features)
np.savez(os.path.join(save_dir, 'test_labels_clf_avg.npz'), text_targets)

print(f"Files saved successfully to {save_dir}")
    