# -*- coding: utf-8 -*-
"""
MS-STGCN (v13.5 Super Large Field) 批量性能诊断工具
================================================
适配：同步了 v13.5 的 4分支 TCN 结构 (Max Dilation=8)。
功能：加载 Phase 1 训练的模型，进行纯监督学习性能测试。
"""

import os
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 1. 全局配置 ===
DATA_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/Data1"
PICTURE_SAVE_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/Data1/MS_STGCN_v13_5_Analysis"
TEST_PATH = os.path.join(DATA_PATH, "test")
SCALER_PATH = os.path.join(DATA_PATH, "scaler_params3.json") 

os.makedirs(PICTURE_SAVE_PATH, exist_ok=True)

# 【关键】指向 v13.5 训练出的模型路径
MODEL_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/St-GCN/models_stgcn_v13_5_mstcn_65fps"
FE_PATH = os.path.join(MODEL_PATH, "feature_extractor_stgcn.weights.h5")
CH_PATH = os.path.join(MODEL_PATH, "classifier_head_stgcn.h5")

CONFIDENCE_THRESHOLD = 0.92 
L2_REG_VALUE = 1e-4

# ============================================================================
# 2. MS-STGCN 模型定义 (必须完全同步 v13.5 训练脚本)
# ============================================================================

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

# === 【核心修正】 v13.5 的 4分支 MS-TCN ===
class StreamingMultiScaleTCN(layers.Layer):
    def __init__(self, filters, stride=1, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.l2_reg = keras.regularizers.l2(l2_reg)

        c_branch = filters // 4
        c_last = filters - (c_branch * 3)
        
        # Branch 1: k=3
        self.conv_s = layers.Conv1D(c_branch, 3, strides=stride, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        # Branch 2: k=5, d=2
        self.conv_m = layers.Conv1D(c_branch, 5, strides=stride, dilation_rate=2, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        # Branch 3: k=9, d=4
        self.conv_l = layers.Conv1D(c_branch, 9, strides=stride, dilation_rate=4, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        # Branch 4: k=9, d=8 (v13.5 新增)
        self.conv_xl = layers.Conv1D(c_last, 9, strides=stride, dilation_rate=8, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)

        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=self.l2_reg)

    def build(self, input_shape):
        super().build(input_shape)
        # Buffer 初始化
        self.buf_s = self.add_weight(name='buf_s', shape=(1, 2, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_m = self.add_weight(name='buf_m', shape=(1, 8, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_l = self.add_weight(name='buf_l', shape=(1, 32, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_xl = self.add_weight(name='buf_xl', shape=(1, 64, input_shape[-1]), initializer='zeros', trainable=False)

    def reset_states(self):
        for buf in [self.buf_s, self.buf_m, self.buf_l, self.buf_xl]:
            buf.assign(tf.zeros_like(buf))

    def call(self, inputs, training=None):
        # 这里的逻辑只用于流式推理 (Testing)
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
        return tf.squeeze(out, axis=1)

# ============================================================================
# 3. 增强可视化函数 (保持不变)
# ============================================================================
def plot_dual_analysis(rec_frames, peak_frames, stats, classes, global_rec_p80, global_peak_avg, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

    fig = plt.figure(figsize=(16, 14))
    
    # --- A. 首次识别帧 ---
    ax1 = plt.subplot2grid((12, 1), (0, 0), rowspan=4)
    ax1.hist(rec_frames, bins=30, color='#4CAF50', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(rec_frames), color='red', linestyle='--', label=f'Mean: {np.mean(rec_frames):.1f}')
    ax1.axvline(global_rec_p80, color='orange', linestyle='--', linewidth=2, label=f'P80: {global_rec_p80:.1f}')
    ax1.set_title(f'1. MS-TCN (65 Frames / v13.5) 首次识别帧分布 - Total: {len(rec_frames)}', fontweight='bold')
    ax1.set_ylabel('Count'); ax1.legend(loc='upper right'); ax1.grid(axis='y', alpha=0.3)

    # --- B. 波峰帧 ---
    ax2 = plt.subplot2grid((12, 1), (5, 0), rowspan=4)
    ax2.hist(peak_frames, bins=30, color='#2196F3', edgecolor='white', alpha=0.8)
    ax2.axvline(global_peak_avg, color='purple', linestyle='--', linewidth=2, label=f'Mean Peak: {global_peak_avg:.1f}')
    ax2.set_title(f'2. MS-TCN 波峰置信度帧分布 (Peak Confidence)', fontweight='bold')
    ax2.set_xlabel('Frame Index'); ax2.set_ylabel('Count'); ax2.legend(loc='upper right'); ax2.grid(axis='y', alpha=0.3)

    # --- C. 表格 ---
    table_data = []
    col_labels = ['Class', 'First Rec (Mean)', 'First Rec (P80)', 'Peak Frame (Mean)', 'Avg Peak Conf']
    
    for cls in classes:
        rec_list = stats['rec'][cls]
        peak_list = stats['peak'][cls]
        conf_list = stats['peak_val'][cls]
        
        if rec_list:
            row = [cls, f"{np.mean(rec_list):.1f}", f"{np.percentile(rec_list, 80):.1f}",
                   f"{np.mean(peak_list):.1f}", f"{np.mean(conf_list):.4f}"]
        else:
            row = [cls, "-", "-", "-", "-"]
        table_data.append(row)
    
    table_data.append(["🌍 GLOBAL", f"{np.mean(rec_frames):.1f}", f"{global_rec_p80:.1f}", f"{global_peak_avg:.1f}", "-"])

    ax_table = plt.subplot2grid((12, 1), (10, 0), rowspan=2)
    ax_table.axis('off')
    the_table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colColours=['#eeeeee']*5)
    the_table.auto_set_font_size(False); the_table.set_fontsize(10); the_table.scale(1, 1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📈 MS-STGCN (v13.5) 分析图表已保存至: {save_path}")

# ============================================================================
# 4. 主程序
# ============================================================================
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    # 1. 加载配置
    with open(SCALER_PATH, 'r') as f:
        scaler_data = json.load(f)
    mean = np.array(scaler_data['mean']); scale = np.array(scaler_data['scale'])
    scale[scale == 0] = 1.0
    print("✅ Scaler Loaded.")

    classes = sorted([d for d in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    test_files = []
    for c in classes:
        for f in os.listdir(os.path.join(TEST_PATH, c)):
            if f.endswith(".csv"):
                test_files.append({"path": os.path.join(TEST_PATH, c, f), "label": class_to_idx[c], "class": c})

    # 2. 加载模型
    print("Building MS-STGCN Model (v13.5 Super Large Field / 65 Frames)...")
    fe = StreamingSTGCN_Model(output_dim=32, l2_reg=L2_REG_VALUE)
    
    # Build Dummy Input (1, 1, 218)
    dummy_input = tf.zeros((1, 1, 218))
    fe(dummy_input) 
    
    # Load Weights
    print(f"Loading weights from: {FE_PATH}")
    fe.load_weights(FE_PATH)
    print(f"✅ Feature Extractor Loaded.")

    ch = keras.models.load_model(CH_PATH, compile=False)
    print(f"✅ Classifier Head Loaded.")
    
    print(f"🚀 开始 MS-STGCN (v13.5) 全序列流式扫描 (共 {len(test_files)} 个样本)...")
    print("⚠️  注意：视野为 65 帧，前 1 秒内的预测可能不稳定。")

    # 3. 统计容器
    stats = {
        'rec': {c: [] for c in classes}, 
        'peak': {c: [] for c in classes}, 
        'peak_val': {c: [] for c in classes}
    }
    all_rec_frames = []
    all_peak_frames = []
    failed_count = 0

    # 4. 推理循环
    for item in tqdm(test_files, desc="Processing"):
        try:
            df = pd.read_csv(item["path"])
            data = df.select_dtypes(include=np.number).values
        except: continue
        if len(data) == 0: continue

        fe.reset_states() # 重置记忆
        
        target_label = item["label"]
        first_rec_frame = -1
        max_conf_val = 0.0
        peak_frame_idx = -1
        frame_preds = []

        for i in range(len(data)):
            # 预处理
            raw = data[i]
            norm = np.zeros_like(raw)
            msk = np.abs(scale) > 1e-6
            norm[msk] = (raw[msk] - mean[msk]) / scale[msk]
            
            inp = tf.convert_to_tensor(norm.reshape(1, 1, -1), dtype=tf.float32)

            # 推理
            feat = fe(inp, training=False)
            probs = ch(feat, training=False).numpy()[0]
            
            curr_pred = np.argmax(probs)
            curr_conf = probs[target_label]
            frame_preds.append(curr_conf)

            # A. 首次识别
            if first_rec_frame == -1:
                if curr_pred == target_label and curr_conf >= CONFIDENCE_THRESHOLD:
                    first_rec_frame = i + 1

            # B. 波峰
            if curr_pred == target_label:
                if curr_conf > max_conf_val:
                    max_conf_val = curr_conf
                    peak_frame_idx = i + 1
        
        # === 判定 ===
        if max_conf_val >= CONFIDENCE_THRESHOLD: 
            if first_rec_frame == -1: first_rec_frame = peak_frame_idx 
            
            stats['rec'][item["class"]].append(first_rec_frame)
            stats['peak'][item["class"]].append(peak_frame_idx)
            stats['peak_val'][item["class"]].append(max_conf_val)
            
            all_rec_frames.append(first_rec_frame)
            all_peak_frames.append(peak_frame_idx)
        else:
            failed_count += 1
            print(f"\n❌ 失败样本: {item['class']} / {os.path.basename(item['path'])}")
            print(f"   - 最大置信度: {max_conf_val:.4f}")
            print("-" * 30)

    # 5. 结果汇总
    print("\n" + "="*50)
    print(f"样本总数: {len(test_files)}")
    print(f"成功识别: {len(test_files) - failed_count}")
    print(f"完全失败: {failed_count}")
    
    if all_rec_frames:
        g_rec_p80 = np.percentile(all_rec_frames, 80)
        g_peak_avg = np.mean(all_peak_frames)
        
        print(f"🌍 全局首次识别 P80: {g_rec_p80:.1f} 帧")
        print(f"🌍 全局波峰位置 Avg: {g_peak_avg:.1f} 帧")
        
        save_path = os.path.join(PICTURE_SAVE_PATH, "MS_STGCN_v13_5_Performance.png")
        plot_dual_analysis(all_rec_frames, all_peak_frames, stats, classes, g_rec_p80, g_peak_avg, save_path)
    else:
        print("❌ 无有效数据生成图表")