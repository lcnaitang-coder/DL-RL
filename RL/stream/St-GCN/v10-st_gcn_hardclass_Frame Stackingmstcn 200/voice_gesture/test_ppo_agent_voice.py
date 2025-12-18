# -*- coding: utf-8 -*-
"""
PPO 模型验证脚本 (Voice Integration Version)
================================================
适配：v13.6 训练脚本 + 语音指令增强
功能：加载 RL Agent (带语音输入) 进行测试，自动处理 Scaler，输出详细分析。
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import random

# === 1. 全局配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../../Data1"))
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test")

# [路径] 指向 v13.6 预训练模型文件夹 (ST-GCN 部分不变)
PRETRAIN_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models_stgcn_v13_6_mstcn_193fps/attention"))
FE_WEIGHTS_PATH = os.path.join(PRETRAIN_PATH, "feature_extractor_stgcn.weights.h5")
CH_MODEL_PATH = os.path.join(PRETRAIN_PATH, "classifier_head_stgcn.h5")

# [路径] 指向 Voice RL 训练结果文件夹
MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models_rl_v13_6_mstcn_193fps/voice"))
PPO_ACTOR_PATH = os.path.join(MODEL_PATH, "ppo_actor_policy_v13_6.weights.h5") 

# Scaler 缓存路径 (使用 Voice 目录下的缓存)
SCALER_CACHE_PATH = os.path.join(MODEL_PATH, "scaler_params_v13_6.json")

L2_REG_VALUE = 1e-4

# 动作死线
CLASS_TIME_LIMITS = { 
    0: 80, 1: 85, 2: 150, 3: 130, 4: 150,
    5: 125, 6: 135, 7: 60, 8: 120, 9: 80,
    10: 120, 11: 120, 12: 110 
}
CLASS_TRIGGER_START = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 
    6: 1, 7: 1, 8: 1, 9: 1, 
    10: 1, 11: 1, 12: 1
}

# === 2. 模型组件 (与 v13.6 训练脚本一致) ===

class LeapMotionAdapter(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs); self.num_nodes = 21; self.out_channels = 18 
    def call(self, inputs):
        palm_feat = inputs[..., 0:18]; bone_feat = inputs[..., 18:]
        shape_back = tf.shape(bone_feat)[:-1]; bone_feat_reshaped = tf.reshape(bone_feat, tf.concat([shape_back, [20, 10]], axis=0))
        palm_node = tf.expand_dims(palm_feat, axis=-2); bone_padding = tf.zeros(tf.concat([tf.shape(bone_feat_reshaped)[:-1], [8]], axis=0))
        bone_nodes = tf.concat([bone_feat_reshaped, bone_padding], axis=-1); return tf.concat([palm_node, bone_nodes], axis=-2)

def get_adapter_adjacency_matrix():
    num_nodes = 21; edges = []
    for root in [1, 5, 9, 13, 17]: edges.append((0, root))
    for fs in [1, 5, 9, 13, 17]: edges.append((fs, fs+1)); edges.append((fs+1, fs+2)); edges.append((fs+2, fs+3))
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges: A[i, j] = 1.0; A[j, i] = 1.0 
    A = A + np.eye(num_nodes); D = np.sum(A, axis=1); D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.; D_mat_inv_sqrt = np.diag(D_inv_sqrt)
    return tf.constant(np.dot(np.dot(D_mat_inv_sqrt, A), D_mat_inv_sqrt), dtype=tf.float32)

class GraphConv(layers.Layer):
    def __init__(self, out_channels, adjacency_matrix, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs); self.out_channels = out_channels; self.A = adjacency_matrix; self.l2_reg = keras.regularizers.l2(l2_reg)
    def build(self, input_shape): self.W = self.add_weight(shape=(input_shape[-1], self.out_channels), initializer='glorot_uniform', regularizer=self.l2_reg, trainable=True)
    def call(self, inputs): x = tf.matmul(inputs, self.W); return tf.einsum('vw,bwc->bvc', self.A, x)

class StreamingMultiScaleTCN(layers.Layer):
    def __init__(self, filters, stride=1, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.l2_reg = keras.regularizers.l2(l2_reg)

        c_branch = filters // 5
        c_last = filters - (c_branch * 4)
        
        self.conv_b1 = layers.Conv1D(c_branch, 3, strides=stride, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        self.conv_b2 = layers.Conv1D(c_branch, 3, strides=stride, dilation_rate=2, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        self.conv_b3 = layers.Conv1D(c_branch, 5, strides=stride, dilation_rate=4, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        self.conv_b4 = layers.Conv1D(c_branch, 9, strides=stride, dilation_rate=8, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        self.conv_b5 = layers.Conv1D(c_last, 9, strides=stride, dilation_rate=12, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)

        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=self.l2_reg)

    def build(self, input_shape):
        super().build(input_shape)
        self.buf_b1 = self.add_weight(name='buf_b1', shape=(1, 2, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b2 = self.add_weight(name='buf_b2', shape=(1, 4, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b3 = self.add_weight(name='buf_b3', shape=(1, 16, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b4 = self.add_weight(name='buf_b4', shape=(1, 64, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b5 = self.add_weight(name='buf_b5', shape=(1, 96, input_shape[-1]), initializer='zeros', trainable=False)

        self.att_dense1 = layers.Dense(self.filters // 4, activation='relu', kernel_regularizer=self.l2_reg)
        self.att_dense2 = layers.Dense(self.filters, activation='sigmoid', kernel_regularizer=self.l2_reg)

    def reset_states(self):
        for buf in [self.buf_b1, self.buf_b2, self.buf_b3, self.buf_b4, self.buf_b5]:
            buf.assign(tf.zeros_like(buf))

    def call(self, inputs, training=None):
        seq_len = tf.shape(inputs)[1]
        is_sequence_processing = training or (seq_len > 1)

        if is_sequence_processing:
            x1 = self.conv_b1(inputs); x2 = self.conv_b2(inputs); x3 = self.conv_b3(inputs); x4 = self.conv_b4(inputs); x5 = self.conv_b5(inputs)
        else:
            c1 = self.buf_b1.value(); combo1 = tf.concat([c1, inputs], axis=1)
            x1 = self.conv_b1(combo1); self.buf_b1.assign(combo1[:, 1:, :]); x1 = x1[:, -1:, :]
            c2 = self.buf_b2.value(); combo2 = tf.concat([c2, inputs], axis=1)
            x2 = self.conv_b2(combo2); self.buf_b2.assign(combo2[:, 1:, :]); x2 = x2[:, -1:, :]
            c3 = self.buf_b3.value(); combo3 = tf.concat([c3, inputs], axis=1)
            x3 = self.conv_b3(combo3); self.buf_b3.assign(combo3[:, 1:, :]); x3 = x3[:, -1:, :]
            c4 = self.buf_b4.value(); combo4 = tf.concat([c4, inputs], axis=1)
            x4 = self.conv_b4(combo4); self.buf_b4.assign(combo4[:, 1:, :]); x4 = x4[:, -1:, :]
            c5 = self.buf_b5.value(); combo5 = tf.concat([c5, inputs], axis=1)
            x5 = self.conv_b5(combo5); self.buf_b5.assign(combo5[:, 1:, :]); x5 = x5[:, -1:, :]

        x_concat = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        attention_score = self.att_dense1(x_concat)
        attention_score = self.att_dense2(attention_score) 
        x_attended = x_concat * attention_score
        return self.conv_out(x_attended)

class StreamingSTGCN_Model(keras.Model):
    def __init__(self, hidden_dim=64, output_dim=32, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.adapter = LeapMotionAdapter(); self.A = get_adapter_adjacency_matrix()
        self.gcn1 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn1 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) 
        self.drop1 = layers.Dropout(0.3)
        self.gcn2 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn2 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) 
        self.drop2 = layers.Dropout(0.3)
        self.dense_out = layers.Dense(output_dim, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))
    
    def reset_states(self):
        self.tcn1.reset_states(); self.tcn2.reset_states()

    def call(self, inputs, training=False):
        x = self.adapter(inputs); B = tf.shape(x)[0]; T = tf.shape(x)[1]; V = 21; C = 18 
        x = tf.reshape(x, (B * T, V, C)); h1 = self.gcn1(x); h1 = tf.nn.relu(h1); h1 = tf.reshape(h1, (B, T, V, 64))
        h1_pool = tf.reduce_mean(h1, axis=2); t1 = self.tcn1(h1_pool, training=training); t1 = self.drop1(t1, training=training)
        t2 = self.tcn2(t1, training=training); t2 = self.drop2(t2, training=training); out = self.dense_out(t2)
        if training or (inputs.shape[1] is not None and inputs.shape[1] > 1): return tf.reduce_mean(out, axis=1)
        return tf.squeeze(out, axis=1)

# === 3. PPO Actor ===
def build_ppo_actor(state_dim, action_dim):
    inp = layers.Input((state_dim,))
    d1 = layers.Dense(128, activation="relu")(inp)
    d2 = layers.Dense(128, activation="relu")(d1)
    out = layers.Dense(action_dim, activation="sigmoid")(d2)
    return keras.Model(inp, out)

# === 4. Scaler 工具函数 ===
def get_scaler(train_path, cache_path):
    if os.path.exists(cache_path):
        print(f"✅ Loading Scaler from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            data = json.load(f)
            mean = np.array(data['mean'])
            scale = np.array(data['scale'])
            scale[scale==0] = 1.0
            return mean, scale
    else:
        print(f"⚠️ Scaler cache not found. Computing from TRAIN data at {train_path}...")
        sequences = []
        for root, _, files in os.walk(train_path):
            for file in files:
                if file.endswith(".csv"):
                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        if not df.empty:
                            sequences.append(df.select_dtypes(include=np.number).values)
                    except: pass
        
        all_data = np.vstack([s.reshape(-1, s.shape[-1]) for s in sequences])
        scaler = StandardScaler().fit(all_data)
        mean = scaler.mean_
        scale = scaler.scale_
        with open(cache_path, 'w') as f:
            json.dump({'mean': mean.tolist(), 'scale': scale.tolist()}, f)
        print(f"✅ Scaler computed and saved to {cache_path}")
        return mean, scale

# === 5. 绘图函数 ===
def plot_detailed_analysis(results_df, cm, classes, save_path):
    plt.rcParams['axes.unicode_minus'] = False
    try: plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    except: pass
    fig = plt.figure(figsize=(20, 12)); gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, :]); error_counts = results_df.groupby(['True Class', 'Result']).size().unstack(fill_value=0)
    desired_order = ['CORRECT', 'WRONG', 'LATE', 'EARLY', 'MISSED']
    for col in desired_order: 
        if col not in error_counts.columns: error_counts[col] = 0
    error_counts = error_counts[desired_order]
    colors = ['#4CAF50', '#F44336', '#FF9800', '#FFEB3B', '#9E9E9E'] 
    error_counts.plot(kind='bar', stacked=True, ax=ax1, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title('Per-Class Performance (Voice Integrated)', fontsize=16); ax1.legend(title='Outcome', bbox_to_anchor=(1.0, 1.0))
    
    ax2 = fig.add_subplot(gs[1, 0]); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Confusion Matrix')
    
    ax3 = fig.add_subplot(gs[1, 1]); total = results_df['Result'].value_counts()
    if len(total) > 0:
        ax3.pie(total, labels=total.index, autopct='%1.1f%%', startangle=140, colors=['#4CAF50' if x=='CORRECT' else '#FF9800' if x=='LATE' else '#F44336' if x=='WRONG' else '#9E9E9E' for x in total.index])
    ax3.set_title('Overall Performance Distribution')
    
    plt.tight_layout(); plt.savefig(save_path, dpi=150); print(f"Plot saved to: {save_path}")

# === 6. 主程序 ===
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    # 1. Scaler
    mean, scale = get_scaler(TRAIN_PATH, SCALER_CACHE_PATH)
    
    classes = sorted([d for d in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    print(f"Detected {len(classes)} classes: {classes}")
    
    test_files = []
    for c in classes:
        cls_path = os.path.join(TEST_PATH, c)
        for f in os.listdir(cls_path):
            if f.endswith(".csv"): test_files.append({"path": os.path.join(cls_path, f), "label": class_map[c], "class_name": c})
    
    print(f"Loading Models for Voice Agent Evaluation...")
    
    # 2. ST-GCN
    fe = StreamingSTGCN_Model(output_dim=32, l2_reg=L2_REG_VALUE)
    fe(tf.zeros((1, 1, 218)), training=False) 
    fe.load_weights(FE_WEIGHTS_PATH)
    
    ch = keras.models.load_model(CH_MODEL_PATH, compile=False)
    
    # 3. PPO Agent (Voice Enhanced)
    STACK_SIZE = 5
    NUM_CLASSES = len(classes)
    AUDIO_DIM = NUM_CLASSES + 1
    
    # Obs Dim = (13 * 5 + 2) + 14 = 81
    obs_dim = (NUM_CLASSES * STACK_SIZE + 2) + AUDIO_DIM
    
    actor = build_ppo_actor(obs_dim, 1)
    
    try:
        actor.load_weights(PPO_ACTOR_PATH)
        print(f"✅ Loaded Voice RL weights from: {PPO_ACTOR_PATH}")
    except Exception as e:
        print(f"❌ Error loading RL weights: {e}")
        exit()

    results_log = []; y_true_all, y_pred_all = [], []
    
    # === DEBUG 配置 ===
    debug_samples = 3
    sample_counters = {c: 0 for c in classes}

    print(f"\n🚀 Running Validation with Voice Simulation...")
    
    for item in tqdm(test_files):
        try: 
            df = pd.read_csv(item["path"])
            if df.empty: continue
            data = df.select_dtypes(include=np.number).values
        except: continue
        
        fe.reset_states(); prob_history = deque(maxlen=10)
        stack_buffer = deque(maxlen=STACK_SIZE)
        
        # [新增] Voice Simulation State
        simulated_audio_state = np.zeros(AUDIO_DIM)
        audio_decay = 0.95
        
        # 模拟 70% 概率出现语音
        has_voice_command = (random.random() < 0.7)
        true_label = item["label"]
        trigger_start = CLASS_TRIGGER_START.get(true_label, 1)
        
        if has_voice_command:
            voice_onset_step = random.randint(5, max(10, trigger_start + 15))
        else:
            voice_onset_step = -1
            
        triggered = False
        result_type = "MISSED"
        final_pred_name = "None"
        offset_limit = CLASS_TIME_LIMITS.get(true_label, 120)
        
        is_debug_sample = sample_counters[item['class_name']] < debug_samples
        if is_debug_sample:
            print(f"\n--- Debug Sample: {item['class_name']} (Voice: {has_voice_command}, Onset: {voice_onset_step}) ---")
        
        for i in range(len(data)):
            # 1. Feature Extraction
            raw = data[i]; norm = np.zeros_like(raw); msk = np.abs(scale) > 1e-6
            norm[msk] = (raw[msk]-mean[msk])/scale[msk]
            inp = tf.convert_to_tensor(norm.reshape(1, 1, -1), dtype=tf.float32)
            
            feat = fe(inp, training=False)
            probs = ch(feat, training=False).numpy()[0]
            prob_history.append(np.max(probs))
            
            stack_buffer.append(probs)
            while len(stack_buffer) < STACK_SIZE: stack_buffer.append(probs)
            stacked_probs = np.array(stack_buffer).flatten()
            
            norm_step = min(i / 200.0, 1.0)
            prob_std = np.std(prob_history) if len(prob_history) >= 2 else 0.0
            
            gesture_obs = np.concatenate([stacked_probs, [norm_step], [prob_std]])
            
            # 2. Voice State Update
            if has_voice_command and i == voice_onset_step:
                if 0 <= true_label < NUM_CLASSES:
                    simulated_audio_state[true_label] = 1.0
                    simulated_audio_state[true_label] *= np.random.uniform(0.8, 1.0)
            
            simulated_audio_state *= audio_decay
            if np.max(simulated_audio_state[:-1]) < 0.1:
                simulated_audio_state[-1] = 1.0
            else:
                simulated_audio_state[-1] = 0.0
                
            # 3. Combine Obs
            state = np.concatenate([gesture_obs, simulated_audio_state])
            
            # 4. RL Inference
            action_prob = actor(state.reshape(1, -1), training=False).numpy()[0][0]
            
            if is_debug_sample and (i % 20 == 0 or i == offset_limit):
                max_conf = np.max(probs)
                pred_cls = classes[np.argmax(probs)]
                print(f"Step {i:3d} | ST-GCN: {max_conf:.2f} ({pred_cls}) | VoiceActive: {simulated_audio_state[-1]<0.5} | RL: {action_prob:.4f}")

            if action_prob > 0.5:
                pred_idx = np.argmax(probs)
                final_pred_name = classes[pred_idx]
                triggered = True
                
                if i < trigger_start: result_type = "EARLY"
                elif pred_idx != true_label: result_type = "WRONG"
                elif i > offset_limit: result_type = "LATE"
                else: result_type = "CORRECT"
                
                if is_debug_sample:
                    print(f"🔥 TRIGGERED at Step {i} | Result: {result_type}")
                
                y_true_all.append(true_label); y_pred_all.append(pred_idx)
                break
        
        if not triggered:
            if is_debug_sample:
                print(f"❄️ FAILED to trigger. Ended as MISSED.")

        results_log.append({"True Class": item["class_name"], "Result": result_type, "Pred Class": final_pred_name})
        sample_counters[item['class_name']] += 1

    df_res = pd.DataFrame(results_log)
    
    if len(df_res) > 0:
        count_correct = len(df_res[df_res['Result'] == 'CORRECT'])
        count_late = len(df_res[df_res['Result'] == 'LATE'])
        strict_acc = count_correct / len(df_res) * 100
        loose_acc = (count_correct + count_late) / len(df_res) * 100
        print(f"\n📊 Accuracy Summary (Voice Integrated):")
        print(f"   Strict (On Time): {strict_acc:.2f}%")
        print(f"   Loose (Inc. Late): {loose_acc:.2f}%")
        
        save_path = os.path.join(MODEL_PATH, "PPO_Validation_Voice_Analysis.png")
        
        if len(y_true_all) > 0:
            cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(classes)))
            plot_detailed_analysis(df_res, cm, classes, save_path)
    else:
        print("No samples processed.")
