import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
import itertools
from torch.utils.data import Dataset, DataLoader

prefix = os.path.abspath(os.path.join(os.getcwd(), "./"))

# 读取EATD和CMDC两个数据集的特征
def load_dataset_features(dataset_name, is_test=False):
    """加载指定数据集的特征，处理标签不一致问题"""
    data_type = "test" if is_test else "whole"
    
    # 构建文件路径
    text_path = os.path.join(prefix, f'Features/{dataset_name}/TextWhole/{data_type}_samples_clf_avg.npz')
    audio_path = os.path.join(prefix, f'Features/{dataset_name}/AudioWhole/{data_type}_samples_clf_256.npz')
    text_label_path = os.path.join(prefix, f'Features/{dataset_name}/TextWhole/{data_type}_labels_clf_avg.npz')
    audio_label_path = os.path.join(prefix, f'Features/{dataset_name}/AudioWhole/{data_type}_labels_clf_256.npz')
    
    # 验证文件是否存在
    for path in [text_path, audio_path, text_label_path, audio_label_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征文件不存在: {path}")
    
    # 加载文本特征
    with np.load(text_path) as data:
        text_features = data['arr_0']
    
    # 加载音频特征（自动处理维度）
    with np.load(audio_path) as data:
        audio_features = data['arr_0']
        # 打印音频特征原始形状用于调试
        print(f"{dataset_name} {data_type} 音频特征原始形状: {audio_features.shape}")
        print(f"{dataset_name} {data_type} 文本特征原始形状: {text_features.shape}")
        
        # 处理音频特征维度 - 自动移除所有大小为1的维度
        audio_features = np.squeeze(audio_features)
        print(f"移除单维度后音频特征形状: {audio_features.shape}")
        
        # 如果处理后仍多于2维，尝试合并维度（假设前两维为样本和时间步）
        if audio_features.ndim > 2:
            # 计算新的特征维度（合并除第一维外的所有维度）
            new_feature_dim = np.prod(audio_features.shape[1:])
            audio_features = audio_features.reshape(audio_features.shape[0], new_feature_dim)
            print(f"合并维度后音频特征形状: {audio_features.shape}")
        
        # 确保最终是2维形状 (样本数, 特征数)
        if audio_features.ndim != 2:
            raise ValueError(f"音频特征维度处理后仍不符合要求: {audio_features.ndim}维，形状为{audio_features.shape}")
    
    # 加载标签
    with np.load(text_label_path) as data:
        text_targets = data['arr_0']
    with np.load(audio_label_path) as data:
        audio_targets = data['arr_0']
    
        # 处理样本数量不一致问题
    text_count = len(text_features)
    audio_count = len(audio_features)
    label_count_text = len(text_targets)
    label_count_audio = len(audio_targets)
        # 确保标签数量与对应特征数量一致
    if text_count != label_count_text:
        print(f"警告: 文本特征与文本标签数量不一致，截断为较小值 ({min(text_count, label_count_text)})")
        text_features = text_features[:min(text_count, label_count_text)]
        text_targets = text_targets[:min(text_count, label_count_text)]
        text_count = len(text_features)
    
    if audio_count != label_count_audio:
        print(f"警告: 音频特征与音频标签数量不一致，截断为较小值 ({min(audio_count, label_count_audio)})")
        audio_features = audio_features[:min(audio_count, label_count_audio)]
        audio_targets = audio_targets[:min(audio_count, label_count_audio)]
        audio_count = len(audio_features)
    
    # 找出样本数量差异并截断到最小数量
    if text_count != audio_count:
        min_count = min(text_count, audio_count)
        print(f"警告: {dataset_name} 数据集中文本和音频样本数量不一致，截断为较小值 ({min_count})")
        text_features = text_features[:min_count]
        audio_features = audio_features[:min_count]
        text_targets = text_targets[:min_count]
        audio_targets = audio_targets[:min_count]
    
    # 检查标签是否一致，若不一致则找出差异并使用文本标签
    if not np.array_equal(text_targets, audio_targets):
        # 找出不一致的索引
        diff_indices = np.where(text_targets != audio_targets)[0]
        print(f"警告: {dataset_name} 数据集中有{len(diff_indices)}个样本标签不一致，将使用文本标签")
        # 使用文本标签作为基准
        audio_targets = text_targets.copy()
    
    # 处理文本特征维度（如果需要）
    if text_features.ndim > 2:
        text_features = text_features.reshape(text_features.shape[0], -1)
        print(f"文本特征维度调整后形状: {text_features.shape}")
    
    # 组合特征（确保样本数量一致）
    min_samples = min(len(text_features), len(audio_features))
    fuse_features = [[audio_features[i], text_features[i]] for i in range(min_samples)]
    fuse_targets = text_targets[:min_samples]
    
    return fuse_features, fuse_targets

# 加载EATD和CMDC数据集的训练和测试特征
eatd_features, eatd_targets = load_dataset_features('ETAD', is_test=False)
cmdc_features, cmdc_targets = load_dataset_features('CDMC', is_test=False)

# 加载测试集特征 (将whole改为test)
eatd_test_features, eatd_test_targets = load_dataset_features('ETAD', is_test=True)
cmdctest_features, cmdctest_targets = load_dataset_features('CDMC', is_test=True)

# 合并两个数据集作为训练数据
all_features = eatd_features + cmdc_features
all_targets = np.concatenate([eatd_targets, cmdc_targets])

# 分别获取两个数据集的索引
eatd_idxs = np.arange(len(eatd_features))
cmdct_idxs = np.arange(len(eatd_features), len(eatd_features) + len(cmdc_features))

# 划分有标签和无标签数据（半监督）
labeled_ratio = 0.3  # 使用30%的数据作为有标签数据
np.random.seed(42)
total_samples = len(all_features)
labeled_idxs = np.random.choice(total_samples, int(total_samples * labeled_ratio), replace=False)
unlabeled_idxs = np.setdiff1d(np.arange(total_samples), labeled_idxs)

# 按数据集划分有标签和无标签索引
eatd_labeled_idxs = [idx for idx in labeled_idxs if idx in eatd_idxs]
cmdct_labeled_idxs = [idx for idx in labeled_idxs if idx in cmdct_idxs]
eatd_unlabeled_idxs = [idx for idx in unlabeled_idxs if idx in eatd_idxs]
cmdct_unlabeled_idxs = [idx for idx in unlabeled_idxs if idx in cmdct_idxs]

# 按类别划分索引
dep_idxs = np.where(all_targets == 1)[0]
non_dep_idxs = np.where(all_targets == 0)[0]

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
    
def standard_confusion_matrix(y_test, y_test_pred):
    """创建标准混淆矩阵"""
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])

def model_performance(y_test, y_test_pred_proba):
    """评估模型性能"""
    y_test_pred = y_test_pred_proba

    # 计算混淆矩阵
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_test_pred, conf_matrix

# 定义特征提取器
class TextFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(TextFeatureExtractor, self).__init__()
        self.num_classes = config['num_classes']
        self.dropout = config['dropout']
        self.hidden_dims = config['text_hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['text_embed_size']
        
        # 注意力层
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )

        # 双向LSTM
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=True)
        
        # 特征提取FC层
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
    def attention_net_with_w(self, lstm_out, lstm_hidden):
        """注意力机制"""
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        lstm_hidden = lstm_hidden.unsqueeze(1)
        atten_w = self.attention_layer(lstm_hidden)
        m = nn.Tanh()(h)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        softmax_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, h)
        return context.squeeze(1)
    
    def forward(self, x):
        x = x.permute(1, 0, 2)  # [len_seq, batch_size, embedding_dim]
        output, (final_hidden_state, _) = self.lstm_net(x)
        output = output.permute(1, 0, 2)  # [batch_size, len_seq, n_hidden * 2]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        return self.fc_out(atten_out)

class AudioFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(AudioFeatureExtractor, self).__init__()
        self.num_classes = config['num_classes']
        self.dropout = config['dropout']
        self.hidden_dims = config['audio_hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['audio_embed_size']
        
        # 注意力层
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )

        # GRU网络
        self.gru_net = nn.GRU(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=False, batch_first=True)
        
        self.ln = nn.LayerNorm(self.embedding_size)
        
        # 特征提取FC层
        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, x):
        x = self.ln(x)
        x, _ = self.gru_net(x)
        x = x.sum(dim=1)
        return self.fc_audio(x)

# 定义融合网络
class FusionNet(nn.Module):
    def __init__(self, config):
        super(FusionNet, self).__init__()
        self.config = config
        
        # 特征提取器
        self.text_extractor = TextFeatureExtractor(config)
        self.audio_extractor = AudioFeatureExtractor(config)
        
        # 模态注意力
        self.modal_attn = nn.Linear(
            config['text_hidden_dims'] + config['audio_hidden_dims'],
            2,  # 两个模态
            bias=False
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config['text_hidden_dims'] + config['audio_hidden_dims'], 
                      config['num_classes']),
            nn.Softmax(dim=1)
        )
        
        # 域鉴别器 (用于跨域适应)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(config['text_hidden_dims'] + config['audio_hidden_dims'], 
                      config['text_hidden_dims']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['text_hidden_dims'], 2),  # 两个域
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, extract_features=False):
        # x是包含音频和文本特征的元组
        audio_feature = self.audio_extractor(x[0])
        text_feature = self.text_extractor(x[1])
        
        # 模态注意力
        concat_features = torch.cat((text_feature, audio_feature), dim=1)
        modal_weights = F.softmax(self.modal_attn(concat_features), dim=1)
        text_weight = modal_weights[:, 0].unsqueeze(1)
        audio_weight = modal_weights[:, 1].unsqueeze(1)
        
        weighted_text = text_feature * text_weight
        weighted_audio = audio_feature * audio_weight
        fused_features = torch.cat((weighted_text, weighted_audio), dim=1)
        
        if extract_features:
            return fused_features
        
        # 分类输出
        class_output = self.classifier(fused_features)
        # 域输出
        domain_output = self.domain_discriminator(fused_features)
        
        return class_output, domain_output, fused_features

# 定义MME(极大极小熵)损失函数
class MMELoss(nn.Module):
    def __init__(self, config):
        super(MMELoss, self).__init__()
        self.config = config
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
    def forward(self, class_output, domain_output, features, labels=None, domains=None, is_labeled=True):
        loss = 0.0
        
        # 有标签数据的分类损失
        if is_labeled and labels is not None:
            loss += self.class_criterion(class_output, labels)
        
        # 域分类损失 (对抗损失)
        if domains is not None:
            loss += self.config['domain_lambda'] * self.domain_criterion(domain_output, domains)
        
        # MME损失 - 最大化无标签数据的熵
        if not is_labeled:
            # 计算预测分布的熵
            entropy = -torch.sum(class_output * torch.log(class_output + 1e-8), dim=1).mean()
            # 最大化无标签数据的熵 (鼓励不确定性)
            loss += self.config['mme_lambda'] * (-entropy)
        
        # 特征分布一致性损失
        if features is not None and self.config['consistency_lambda'] > 0:
            # 计算特征的均值和方差
            mean = torch.mean(features, dim=0)
            var = torch.var(features, dim=0)
            # 鼓励特征分布接近标准正态分布
            loss += self.config['consistency_lambda'] * (torch.mean(torch.square(mean)) + torch.mean(torch.square(var - 1)))
        
        return loss

# 定义数据集加载器
class MultimodalDataset(Dataset):
    def __init__(self, features, targets, domains=None, indices=None):
        self.features = [features[i] for i in indices] if indices is not None else features
        self.targets = targets[indices] if indices is not None else targets
        self.domains = domains[indices] if domains is not None and indices is not None else domains
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        audio = torch.FloatTensor(self.features[idx][0])
        text = torch.FloatTensor(self.features[idx][1])
        target = self.targets[idx]
        domain = self.domains[idx] if self.domains is not None else 0
        
        return (audio, text), target, domain

# 配置参数
config = {
    'num_classes': 2,
    'dropout': 0.3,
    'rnn_layers': 2,
    'audio_embed_size': 256,
    'text_embed_size': 1024,
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 8e-6,
    'audio_hidden_dims': 256,
    'text_hidden_dims': 128,
    'cuda': torch.cuda.is_available(),
    'lambda': 1e-5,
    'domain_lambda': 0.1,  # 域适应损失权重
    'mme_lambda': 0.01,    # MME损失权重
    'consistency_lambda': 0.001,  # 特征一致性损失权重
    'labeled_ratio': 0.3   # 有标签数据比例
}

# 创建域标签（0表示EATD，1表示CMDC）
domain_labels = np.concatenate([
    np.zeros(len(eatd_features)),  # EATD域
    np.ones(len(cmdc_features))   # CMDC域
])

# 创建数据加载器
labeled_dataset = MultimodalDataset(
    all_features, all_targets, domain_labels, labeled_idxs
)
unlabeled_dataset = MultimodalDataset(
    all_features, all_targets, domain_labels, unlabeled_idxs
)

# 测试集
test_features = eatd_test_features + cmdctest_features
test_targets = np.concatenate([eatd_test_targets, cmdctest_targets])
test_domains = np.concatenate([
    np.zeros(len(eatd_test_features)),
    np.ones(len(cmdctest_features))
])
test_dataset = MultimodalDataset(test_features, test_targets, test_domains)

# 数据加载器
labeled_loader = DataLoader(labeled_dataset, batch_size=config['batch_size'], shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# 初始化模型、优化器和损失函数
model = FusionNet(config)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = MMELoss(config)

if config['cuda']:
    model = model.cuda()
    criterion = criterion.cuda()

def train(epoch):
    model.train()
    total_loss = 0
    labeled_correct = 0
    domain_correct = 0
    total_samples = 0
    
    # 同时迭代有标签和无标签数据
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    
    for batch_idx in range(max(len(labeled_loader), len(unlabeled_loader))):
        # 处理有标签数据
        try:
            (audio, text), labels, domains = next(labeled_iter)
            if config['cuda']:
                audio, text, labels, domains = audio.cuda(), text.cuda(), labels.cuda(), domains.cuda()
            
            optimizer.zero_grad()
            class_output, domain_output, _ = model((audio, text))
            
            # 计算分类准确率
            pred = class_output.data.max(1, keepdim=True)[1]
            labeled_correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            
            # 计算域分类准确率
            domain_pred = domain_output.data.max(1, keepdim=True)[1]
            domain_correct += domain_pred.eq(domains.data.view_as(domain_pred)).cpu().sum()
            
            # 计算损失
            loss = criterion(class_output, domain_output, None, labels, domains, is_labeled=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_samples += len(labels)
            
        except StopIteration:
            pass
        
        # 处理无标签数据
        try:
            (audio, text), _, domains = next(unlabeled_iter)
            if config['cuda']:
                audio, text, domains = audio.cuda(), text.cuda(), domains.cuda()
            
            optimizer.zero_grad()
            class_output, domain_output, features = model((audio, text))
            
            # 计算域分类准确率
            domain_pred = domain_output.data.max(1, keepdim=True)[1]
            domain_correct += domain_pred.eq(domains.data.view_as(domain_pred)).cpu().sum()
            
            # 计算MME损失（无标签）
            loss = criterion(class_output, domain_output, features, domains=domains, is_labeled=False)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_samples += len(domains)
            
        except StopIteration:
            pass
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    labeled_acc = labeled_correct / len(labeled_dataset) if len(labeled_dataset) > 0 else 0
    domain_acc = domain_correct / (len(labeled_dataset) + len(unlabeled_dataset)) if (len(labeled_dataset) + len(unlabeled_dataset)) > 0 else 0
    
    print(f'Train Epoch: {epoch}\tLoss: {avg_loss:.6f}\tLabeled Acc: {labeled_acc:.4f}\tDomain Acc: {domain_acc:.4f}')
    return labeled_acc

def evaluate(model, test_loader, fold):
    model.eval()
    total_loss = 0
    correct = 0
    y_preds = []
    y_trues = []
    
    with torch.no_grad():
        for (audio, text), labels, domains in test_loader:
            if config['cuda']:
                audio, text, labels, domains = audio.cuda(), text.cuda(), labels.cuda(), domains.cuda()
            
            class_output, _, _ = model((audio, text))
            loss = criterion(class_output, None, None, labels, is_labeled=True)
            total_loss += loss.item()
            
            pred = class_output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            
            y_preds.extend(pred.cpu().numpy().flatten())
            y_trues.extend(labels.cpu().numpy().flatten())
    
    # 计算性能指标
    y_test_pred, conf_matrix = model_performance(y_trues, y_preds)
    
    print(f'\nTest set: Average loss: {total_loss/len(test_loader.dataset):.4f}')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1] + 1e-8)
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0] + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}\n")
    print('='*89)
    
    return accuracy, f1_score

if __name__ == '__main__':
    max_f1 = -1
    max_acc = -1
    max_train_acc = -1
    
    for epoch in range(1, config['epochs'] + 1):
        train_acc = train(epoch)
        test_acc, test_f1 = evaluate(model, test_loader, 0)
        
        # 保存最佳模型
        if test_f1 > max_f1 and train_acc > 0.7:
            max_f1 = test_f1
            max_acc = test_acc
            save(model, os.path.join(prefix, f'Model/ClassificationTest/Fuse/fuse_mme_{max_f1:.4f}'))
            print(f'*'*64)
            print(f'Model saved: F1: {max_f1:.4f}, Acc: {max_acc:.4f}')
            print(f'*'*64)
