# -*- coding: utf-8 -*-
"""
PPO 模型验证脚本 (v13.3 修正版 - 适配 MS-TCN)
================================================
修正：同步了 v13.3 训练脚本中的 StreamingMultiScaleTCN 结构。
功能：增加实时调试信息，用于诊断 "100% Missed" 问题。
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import json
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === 1. 全局配置 ===
DATA_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/Data1"
PRETRAIN_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/models_stgcn_v13_mstcn_33fps" # 注意：这里要指向包含 feature_extractor 的文件夹

# 指向 v13.3 PPO 训练结果
MODEL_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/models_rl_v13_mstcn_33fps"
PPO_ACTOR_PATH = os.path.join(MODEL_PATH, "ppo_actor_policy_v13_mstcn.weights.h5") 

TEST_PATH = os.path.join(DATA_PATH, "test") # 或者 "test"
SCALER_PATH = os.path.join(DATA_PATH, "scaler_params3.json")

FE_WEIGHTS_PATH = os.path.join(PRETRAIN_PATH, "feature_extractor_stgcn.weights.h5")
CH_MODEL_PATH = os.path.join(PRETRAIN_PATH, "classifier_head_stgcn.h5")

L2_REG_VALUE = 1e-4

# 动作死线
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

# === 2. 模型组件 (必须与 v13.3 训练脚本完全一致) ===
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

# 【修正】使用 StreamingMultiScaleTCN 替换旧的 StreamingTCN
# 【必须更新】与训练脚本 v13.3 保持完全一致的 MS-TCN
class StreamingMultiScaleTCN(layers.Layer):
    def __init__(self, filters, stride=1, l2_reg=1e-4, **kwargs):
        """
        MS-TCN v2 (Large Field) - 同步 v13.3 训练脚本结构:
        Branch 1: k=3 (细节)
        Branch 2: k=5, d=2 (中等视野) -> Buffer=8
        Branch 3: k=9, d=4 (大视野) -> Buffer=32
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.l2_reg = keras.regularizers.l2(l2_reg)

        c_small = filters // 4
        c_mid = filters // 4
        c_large = filters - c_small - c_mid
        
        # Branch 1: Small (k=3)
        self.conv_small = layers.Conv1D(c_small, 3, strides=stride, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        
        # Branch 2: Mid (k=5, d=2)
        self.conv_mid = layers.Conv1D(c_mid, 5, strides=stride, dilation_rate=2, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        
        # Branch 3: Large (k=9, d=4)
        self.conv_large = layers.Conv1D(c_large, 9, strides=stride, dilation_rate=4, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)

        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=self.l2_reg)

    def build(self, input_shape):
        super().build(input_shape)
        
        # k=3, d=1 -> buf=2
        self.buf_small = self.add_weight(name='buf_s', shape=(1, 2, input_shape[-1]), initializer='zeros', trainable=False)
        
        # k=5, d=2 -> buf=(5-1)*2 = 8
        self.buf_mid   = self.add_weight(name='buf_m', shape=(1, 8, input_shape[-1]), initializer='zeros', trainable=False)
        
        # k=9, d=4 -> buf=(9-1)*4 = 32
        self.buf_large = self.add_weight(name='buf_l', shape=(1, 32, input_shape[-1]), initializer='zeros', trainable=False)

    def reset_states(self):
        for buf in [self.buf_small, self.buf_mid, self.buf_large]:
            buf.assign(tf.zeros_like(buf))

    def call(self, inputs, training=None):
        # 验证模式下通常 training=False，走缓存逻辑
        # 但为了兼容性，保留全序列处理逻辑
        seq_len = tf.shape(inputs)[1]
        is_sequence_processing = training or (seq_len > 1)

        if is_sequence_processing:
            x1 = self.conv_small(inputs)
            x2 = self.conv_mid(inputs)
            x3 = self.conv_large(inputs)
        else:
            # Branch 1
            curr_s = self.buf_small.value(); comb_s = tf.concat([curr_s, inputs], axis=1)
            x1 = self.conv_small(comb_s); self.buf_small.assign(comb_s[:, 1:, :]); x1 = x1[:, -1:, :]

            # Branch 2
            curr_m = self.buf_mid.value(); comb_m = tf.concat([curr_m, inputs], axis=1)
            x2 = self.conv_mid(comb_m); self.buf_mid.assign(comb_m[:, 1:, :]); x2 = x2[:, -1:, :]

            # Branch 3
            curr_l = self.buf_large.value(); comb_l = tf.concat([curr_l, inputs], axis=1)
            x3 = self.conv_large(comb_l); self.buf_large.assign(comb_l[:, 1:, :]); x3 = x3[:, -1:, :]

        x = tf.concat([x1, x2, x3], axis=-1)
        return self.conv_out(x)

class StreamingSTGCN_Model(keras.Model):
    def __init__(self, hidden_dim=64, output_dim=32, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.adapter = LeapMotionAdapter(); self.A = get_adapter_adjacency_matrix()
        self.gcn1 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn1 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) # 【修正】这里用 MS-TCN
        self.drop1 = layers.Dropout(0.3)
        self.gcn2 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn2 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) # 【修正】这里用 MS-TCN
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

# === 4. 绘图函数 (保持不变) ===
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
    ax1.set_title('Per-Class Performance Breakdown (v13.3 MS-TCN)', fontsize=16); ax1.legend(title='Outcome', bbox_to_anchor=(1.0, 1.0))
    ax2 = fig.add_subplot(gs[1, 0]); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Confusion Matrix'); ax3 = fig.add_subplot(gs[1, 1]); total = results_df['Result'].value_counts()
    if len(total) > 0:
        ax3.pie(total, labels=total.index, autopct='%1.1f%%', startangle=140, colors=['#4CAF50' if x=='CORRECT' else '#FF9800' if x=='LATE' else '#F44336' if x=='WRONG' else '#9E9E9E' for x in total.index])
    ax3.set_title('Overall Performance Distribution')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); print(f"Plot saved to: {save_path}")

# === 5. 主程序 ===
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    with open(SCALER_PATH, 'r') as f: scaler_data = json.load(f)
    mean = np.array(scaler_data['mean']); scale = np.array(scaler_data['scale']); scale[scale==0] = 1.0
    
    classes = sorted([d for d in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    
    test_files = []
    for c in classes:
        cls_path = os.path.join(TEST_PATH, c)
        for f in os.listdir(cls_path):
            if f.endswith(".csv"): test_files.append({"path": os.path.join(cls_path, f), "label": class_map[c], "class_name": c})
    
    print(f"Loading Models for v13.3 Evaluation...")
    # 初始化模型结构
    fe = StreamingSTGCN_Model(output_dim=32, l2_reg=L2_REG_VALUE)
    # Build 模型 (Dummy Forward)
    fe(tf.zeros((1, 1, 218)), training=False) 
    
    # 加载权重
    print(f"Loading ST-GCN weights: {FE_WEIGHTS_PATH}")
    fe.load_weights(FE_WEIGHTS_PATH)
    
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
        print("💡 确保路径指向了 v13.3 训练生成的权重文件！")
        exit()

    results_log = []; y_true_all, y_pred_all = [], []
    
    # === DEBUG 变量 ===
    debug_samples = 5  # 打印前 5 个样本
    debug_counter = 0

    print(f"\n🚀 Running Validation with DEBUG MODE...")
    
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
        is_debug_sample = debug_counter < debug_samples
        if is_debug_sample:
            print(f"\n--- Debug Sample: {item['class_name']} ---")
        
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
            if is_debug_sample and (i % 10 == 0 or i == offset_limit):
                max_conf = np.max(probs)
                pred_cls = classes[np.argmax(probs)]
                print(f"Step {i:3d} | ST-GCN Conf: {max_conf:.2f} ({pred_cls}) | RL Prob: {action_prob:.4f} | Limit: {offset_limit}")

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
        
        if not triggered and is_debug_sample:
            print(f"❄️ FAILED to trigger. Ended as MISSED. (Final Step: {len(data)})")

        results_log.append({"True Class": item["class_name"], "Result": result_type, "Pred Class": final_pred_name})
        debug_counter += 1

    df_res = pd.DataFrame(results_log)
    
    if len(df_res) > 0:
        count_correct = len(df_res[df_res['Result'] == 'CORRECT'])
        count_late = len(df_res[df_res['Result'] == 'LATE'])
        strict_acc = count_correct / len(df_res) * 100
        loose_acc = (count_correct + count_late) / len(df_res) * 100
        print(f"\n📊 Accuracy: Strict={strict_acc:.2f}% | Loose={loose_acc:.2f}%")
        
        save_path = os.path.join(DATA_PATH, "PPO_Validation_v13_4_Analysis.png")
        cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(classes)))
        plot_detailed_analysis(df_res, cm, classes, save_path)
    else:
        print("No samples processed.")