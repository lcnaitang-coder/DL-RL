# -*- coding: utf-8 -*-
"""
PPO 模型验证脚本 (v13.5 Super Large Field 适配版)
================================================
适配：同步 v13.5 训练脚本中的 4分支 MS-TCN 结构 (Max Dilation=8)。
功能：加载 RL Agent 进行测试，输出详细的触发时机和准确率分析。
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

# === 1. 全局配置 (请根据你的本地路径修改) ===
DATA_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/Data1"

# [修改] 指向 v13.5 预训练模型文件夹
PRETRAIN_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/St-GCN/models_stgcn_v13_5_mstcn_65fps" 

# [修改] 指向 v13.5 RL 训练结果文件夹
MODEL_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/St-GCN/models_rl_v13_5_mstcn_65fps"
PPO_ACTOR_PATH = os.path.join(MODEL_PATH, "ppo_actor_policy_v13_5.weights.h5") 

TEST_PATH = os.path.join(DATA_PATH, "test") 
SCALER_PATH = os.path.join(DATA_PATH, "scaler_params3.json") # 确保这个文件存在，或者是 scaler_params.json

FE_WEIGHTS_PATH = os.path.join(PRETRAIN_PATH, "feature_extractor_stgcn.weights.h5")
CH_MODEL_PATH = os.path.join(PRETRAIN_PATH, "classifier_head_stgcn.h5")

L2_REG_VALUE = 1e-4

# 动作死线 (根据 v13.5 训练时的设定)
CLASS_TIME_LIMITS = {
    0: 80, 1: 85, 2: 135, 3: 100, 4: 130,
    5: 125, 6: 135, 7: 55, 8: 70, 9: 95
}
# 最早触发
CLASS_TRIGGER_START = {
    0: 1, 1: 1, 2: 1, 3: 1, 
    4: 1, 5: 1, 
    6: 1, 7: 1, 8: 1, 9: 1
}

# === 2. 模型组件 (必须与 v13.5 训练脚本完全一致) ===

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

# 【关键修改】同步 v13.5 的 4分支 TCN 结构
class StreamingMultiScaleTCN(layers.Layer):
    def __init__(self, filters, stride=1, l2_reg=1e-4, **kwargs):
        """
        MS-TCN v3 (Super Large Field - 4 Branches):
        Branch 1: k=3, d=1 (RF=3)
        Branch 2: k=5, d=2 (RF=9)
        Branch 3: k=9, d=4 (RF=33)
        Branch 4: k=9, d=8 (RF=65) -> Extra Large
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.l2_reg = keras.regularizers.l2(l2_reg)

        c_branch = filters // 4
        c_last = filters - (c_branch * 3)
        
        self.conv_s = layers.Conv1D(c_branch, 3, strides=stride, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        self.conv_m = layers.Conv1D(c_branch, 5, strides=stride, dilation_rate=2, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        self.conv_l = layers.Conv1D(c_branch, 9, strides=stride, dilation_rate=4, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        self.conv_xl = layers.Conv1D(c_last, 9, strides=stride, dilation_rate=8, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)

        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=self.l2_reg)

    def build(self, input_shape):
        super().build(input_shape)
        # Buffer initialization
        self.buf_s = self.add_weight(name='buf_s', shape=(1, 2, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_m = self.add_weight(name='buf_m', shape=(1, 8, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_l = self.add_weight(name='buf_l', shape=(1, 32, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_xl = self.add_weight(name='buf_xl', shape=(1, 64, input_shape[-1]), initializer='zeros', trainable=False)

    def reset_states(self):
        for buf in [self.buf_s, self.buf_m, self.buf_l, self.buf_xl]:
            buf.assign(tf.zeros_like(buf))

    def call(self, inputs, training=None):
        seq_len = tf.shape(inputs)[1]
        is_sequence_processing = training or (seq_len > 1)

        if is_sequence_processing:
            x1 = self.conv_s(inputs)
            x2 = self.conv_m(inputs)
            x3 = self.conv_l(inputs)
            x4 = self.conv_xl(inputs)
        else:
            # Streaming Logic with 4 branches
            curr_s = self.buf_s.value(); comb_s = tf.concat([curr_s, inputs], axis=1)
            x1 = self.conv_s(comb_s); self.buf_s.assign(comb_s[:, 1:, :]); x1 = x1[:, -1:, :]

            curr_m = self.buf_m.value(); comb_m = tf.concat([curr_m, inputs], axis=1)
            x2 = self.conv_m(comb_m); self.buf_m.assign(comb_m[:, 1:, :]); x2 = x2[:, -1:, :]

            curr_l = self.buf_l.value(); comb_l = tf.concat([curr_l, inputs], axis=1)
            x3 = self.conv_l(comb_l); self.buf_l.assign(comb_l[:, 1:, :]); x3 = x3[:, -1:, :]

            curr_xl = self.buf_xl.value(); comb_xl = tf.concat([curr_xl, inputs], axis=1)
            x4 = self.conv_xl(comb_xl); self.buf_xl.assign(comb_xl[:, 1:, :]); x4 = x4[:, -1:, :]

        x = tf.concat([x1, x2, x3, x4], axis=-1)
        return self.conv_out(x)

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

# === 4. 绘图函数 ===
def plot_detailed_analysis(results_df, cm, classes, save_path):
    plt.rcParams['axes.unicode_minus'] = False
    try: plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    except: pass
    fig = plt.figure(figsize=(20, 12)); gs = fig.add_gridspec(2, 2)
    
    # Bar Chart
    ax1 = fig.add_subplot(gs[0, :]); error_counts = results_df.groupby(['True Class', 'Result']).size().unstack(fill_value=0)
    desired_order = ['CORRECT', 'WRONG', 'LATE', 'EARLY', 'MISSED']
    for col in desired_order: 
        if col not in error_counts.columns: error_counts[col] = 0
    error_counts = error_counts[desired_order]
    colors = ['#4CAF50', '#F44336', '#FF9800', '#FFEB3B', '#9E9E9E'] 
    error_counts.plot(kind='bar', stacked=True, ax=ax1, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title('Per-Class Performance (v13.5 Super Large Field)', fontsize=16); ax1.legend(title='Outcome', bbox_to_anchor=(1.0, 1.0))
    
    # Confusion Matrix
    ax2 = fig.add_subplot(gs[1, 0]); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Confusion Matrix')
    
    # Pie Chart
    ax3 = fig.add_subplot(gs[1, 1]); total = results_df['Result'].value_counts()
    if len(total) > 0:
        ax3.pie(total, labels=total.index, autopct='%1.1f%%', startangle=140, colors=['#4CAF50' if x=='CORRECT' else '#FF9800' if x=='LATE' else '#F44336' if x=='WRONG' else '#9E9E9E' for x in total.index])
    ax3.set_title('Overall Performance Distribution')
    
    plt.tight_layout(); plt.savefig(save_path, dpi=150); print(f"Plot saved to: {save_path}")

# === 5. 主程序 ===
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    try:
        with open(SCALER_PATH, 'r') as f: scaler_data = json.load(f)
        mean = np.array(scaler_data['mean']); scale = np.array(scaler_data['scale']); scale[scale==0] = 1.0
    except FileNotFoundError:
        print(f"❌ Scaler file not found at {SCALER_PATH}. Please check the path.")
        exit()
    
    classes = sorted([d for d in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    
    test_files = []
    for c in classes:
        cls_path = os.path.join(TEST_PATH, c)
        for f in os.listdir(cls_path):
            if f.endswith(".csv"): test_files.append({"path": os.path.join(cls_path, f), "label": class_map[c], "class_name": c})
    
    print(f"Loading Models for v13.5 (Super Large Field) Evaluation...")
    
    # 初始化模型结构
    fe = StreamingSTGCN_Model(output_dim=32, l2_reg=L2_REG_VALUE)
    fe(tf.zeros((1, 1, 218)), training=False) # Dummy Forward
    
    # 加载权重
    print(f"Loading ST-GCN weights: {FE_WEIGHTS_PATH}")
    try:
        fe.load_weights(FE_WEIGHTS_PATH)
    except ValueError as e:
        print(f"❌ Error loading ST-GCN weights: {e}")
        print("💡 可能是 TCN 结构不匹配。请确认 FE_WEIGHTS_PATH 指向的是 v13.5 的权重文件。")
        exit()
    
    print(f"Loading Classifier Head: {CH_MODEL_PATH}")
    ch = keras.models.load_model(CH_MODEL_PATH, compile=False)
    
    STACK_SIZE = 5
    obs_dim = len(classes) * STACK_SIZE + 2 
    actor = build_ppo_actor(obs_dim, 1)
    
    try:
        actor.load_weights(PPO_ACTOR_PATH)
        print(f"✅ Loaded RL weights from: {PPO_ACTOR_PATH}")
    except Exception as e:
        print(f"❌ Error loading RL weights: {e}")
        exit()

    results_log = []; y_true_all, y_pred_all = [], []
    
    # === DEBUG 变量 ===
    debug_samples = 5  # 每个类别打印前 n 个样本的详细日志
    sample_counters = {c: 0 for c in classes}

    print(f"\n🚀 Running Validation with DEBUG MODE (v13.5)...")
    
    for item in tqdm(test_files):
        try: 
            df = pd.read_csv(item["path"])
            if df.empty: continue
            data = df.select_dtypes(include=np.number).values
        except: continue
        
        fe.reset_states(); prob_history = deque(maxlen=10)
        stack_buffer = deque(maxlen=STACK_SIZE)
        
        triggered = False
        result_type = "MISSED"
        final_pred_name = "None"
        offset_limit = CLASS_TIME_LIMITS.get(item["label"], 95)
        trigger_start = CLASS_TRIGGER_START.get(item["label"], 12)
        
        # 是否打印当前样本的 debug 信息
        is_debug_sample = sample_counters[item['class_name']] < debug_samples
        if is_debug_sample:
            print(f"\n--- Debug Sample: {item['class_name']} (File: {os.path.basename(item['path'])}) ---")
        
        for i in range(len(data)):
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
            
            state = np.concatenate([stacked_probs, [norm_step], [prob_std]])
            
            # === 获取 RL 动作概率 ===
            action_prob = actor(state.reshape(1, -1), training=False).numpy()[0][0]
            
            # [DEBUG PRINT]
            if is_debug_sample and (i % 15 == 0 or i == offset_limit):
                max_conf = np.max(probs)
                pred_cls = classes[np.argmax(probs)]
                print(f"Step {i:3d} | ST-GCN: {max_conf:.2f} ({pred_cls}) | RL: {action_prob:.4f} | Limit: {offset_limit}")

            if action_prob > 0.5:
                pred_idx = np.argmax(probs)
                final_pred_name = classes[pred_idx]
                triggered = True
                if i < trigger_start: result_type = "EARLY"
                elif pred_idx != item["label"]: result_type = "WRONG"
                elif i > offset_limit: result_type = "LATE"
                else: result_type = "CORRECT"
                
                if is_debug_sample:
                    print(f"🔥 TRIGGERED at Step {i} | Result: {result_type}")
                
                y_true_all.append(item["label"]); y_pred_all.append(pred_idx)
                break
        
        if not triggered:
            if is_debug_sample:
                print(f"❄️ FAILED to trigger. Ended as MISSED. (Final Step: {len(data)})")
            # 如果 Missed，我们通常认为预测为 "None" 或者不计入混淆矩阵，或者记为错误类别
            # 这里为了绘图，暂时不加入 y_pred_all，只在 result_type 记录

        results_log.append({"True Class": item["class_name"], "Result": result_type, "Pred Class": final_pred_name})
        sample_counters[item['class_name']] += 1

    df_res = pd.DataFrame(results_log)
    
    if len(df_res) > 0:
        count_correct = len(df_res[df_res['Result'] == 'CORRECT'])
        count_late = len(df_res[df_res['Result'] == 'LATE'])
        strict_acc = count_correct / len(df_res) * 100
        loose_acc = (count_correct + count_late) / len(df_res) * 100
        print(f"\n📊 Accuracy Summary (v13.5):")
        print(f"   Strict (On Time): {strict_acc:.2f}%")
        print(f"   Loose (Inc. Late): {loose_acc:.2f}%")
        
        save_path = os.path.join(MODEL_PATH, "PPO_Validation_v13_5_Analysis.png")
        
        # 注意：为了混淆矩阵不出错，只统计触发了的情况，或者需要处理 Missed
        # 这里仅基于触发了的样本绘制混淆矩阵
        if len(y_true_all) > 0:
            cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(classes)))
            plot_detailed_analysis(df_res, cm, classes, save_path)
        else:
            print("No triggers occurred, skipping plots.")
    else:
        print("No samples processed.")