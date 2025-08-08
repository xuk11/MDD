import os
import numpy as np
import wave
import librosa
from python_speech_features import *
import sys
import pickle
sys.path.append('/root/autodl-tmp/ICASSP2022-Depression')
import tensorflow.compat.v1 as tf

import vggish.vggish_input as vggish_input
import vggish.vggish_params as vggish_params
import vggish.vggish_postprocess as vggish_postprocess
import vggish.vggish_slim as vggish_slim

import loupe_keras as lpk

tf.enable_eager_execution()

from elmoformanylangs import Embedder

elmo = Embedder('/root/autodl-tmp/ICASSP2022-Depression/zhs.model')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

# Paths to downloaded VGGish files.
checkpoint_path = os.path.join(os.getcwd(), 'vggish/vggish_model.ckpt')
pca_params_path = os.path.join(os.getcwd(), 'vggish/vggish_pca_params.npz')

cluster_size = 16

min_len = 100
max_len = -1

def to_vggish_embedds(x, sr):
    # x为输入的音频，sr为sample_rate
    input_batch = vggish_input.waveform_to_examples(x, sr)
    with tf.Graph().as_default(), tf.Session() as sess:
      vggish_slim.define_vggish_slim()
      vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

      features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
      embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
      [embedding_batch] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: input_batch})

    # Postprocess the results to produce whitened quantized embeddings.
    pproc = vggish_postprocess.Postprocessor(pca_params_path)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    
    return tf.cast(postprocessed_batch, dtype='float32')

def wav2vlad(wave_data, sr):
    global cluster_size
    signal = wave_data
    melspec = librosa.feature.melspectrogram(y=signal, n_mels=80, sr=sr).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    feature_size = melspec.shape[1]
    max_samples = melspec.shape[0]
    output_dim = cluster_size * 16
    feat = lpk.NetVLAD(feature_size=feature_size, max_samples=max_samples, \
                            cluster_size=cluster_size, output_dim=output_dim) \
                                (tf.convert_to_tensor(melspec))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        r = feat.numpy()
    return r
        
def extract_features(subject_id, audio_features, targets, path):
    global max_len, min_len
    wave_files = [f'Q{i}.wav' for i in range(1, 13)]
    waves = []
    srs = []

    for file in wave_files:
        file_path = os.path.join(prefix, f'{path}/{subject_id}', file)
        if not os.path.exists(file_path):
            continue
        wave_file = wave.open(file_path)
        sr = wave_file.getframerate()
        nframes = wave_file.getnframes()
        wave_data = np.frombuffer(wave_file.readframes(nframes), dtype=np.short).astype(float)
        length = nframes / sr
        waves.append(wave_data)
        srs.append(sr)

        if length > max_len:
            max_len = length
        if length < min_len:
            min_len = length

    if len(waves) < 4:
        return
    
    vlads = [wav2vlad(wave_data, sr) for wave_data, sr in zip(waves, srs)]

    for i in range(0, len(vlads) - 3, 4):  # 确保i+4 <= len(vlads)
        combined_vlad = np.concatenate(vlads[i:i+4], axis=-1)
        audio_features.append(combined_vlad)
        label = 0 if 'HC' in path else 1
        targets.append(label)

audio_features = []
audio_targets = []



subjects = ['HC', 'MDD']
# for subject in subjects:
#     for i in range(1, 115):
#         subject_id = f"{subject}{i:02d}"  # HC01, MDD01, ...
#         extract_features(subject_id, audio_features, audio_targets, 'CDMC/Data')

# for subject in subjects:
#     for i in range(1, 115):
#         subject_id = f"{subject}{i:02d}"  # HC01, MDD01, ...
#         extract_features(subject_id, audio_features, audio_targets, 'CDMC/ValidationData')

for subject in subjects:
    for i in range(1, 115):
        subject_id = f"{subject}{i:02d}"  # HC01, MDD01, ...
        extract_features(subject_id, audio_features, audio_targets, 'CDMC/Test')

size_str = str(cluster_size * 16)

# 创建目录（先格式化完整路径）
samples_path = f'Features/CDMC/AudioWhole/test_samples_clf_{size_str}.npz'
labels_path = f'Features/CDMC/AudioWhole/test_labels_clf_{size_str}.npz'

os.makedirs(os.path.dirname(samples_path), exist_ok=True)
os.makedirs(os.path.dirname(labels_path), exist_ok=True)

# 保存文件（使用os.path.join正确拼接路径）
if prefix:  # 如果prefix存在
    samples_path = os.path.join(prefix, samples_path)
    labels_path = os.path.join(prefix, labels_path)

np.savez(samples_path, audio_features)
np.savez(labels_path, audio_targets)

print(max_len, min_len)



