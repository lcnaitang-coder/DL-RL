# -*- coding: utf-8 -*-
"""
ST-GCN 架构自动搜索工具 (Architecture Search)
================================================
功能：
1. 自动测试不同的 分支数量(Branches) 和 膨胀率(Dilation) 组合。
2. 寻找 Accuracy 和 Receptive Field (视野) 之间的最佳平衡点。
3. 这里的 TCN 是动态构建的，只用于训练评估性能。
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import time

# --- 1. 全局配置 ---
DATA_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/Data1"
SAVE_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/St-GCN/arch_search_results"
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test")

# 搜索参数
SEARCH_EPOCHS = 35  # 每个架构跑多少轮 (不用跑满150轮，30-40轮足够看趋势)
BATCH_SIZE = 32
L2_REG = 1e-4

# --- 2. 动态模型组件 ---

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

# === 动态 TCN 构建器 ===
class DynamicMultiScaleTCN(layers.Layer):
    def __init__(self, filters, branch_config, l2_reg=1e-4, **kwargs):
        """
        branch_config: List of tuples [(kernel_size, dilation_rate), ...]
        例如: [(3,1), (5,2), (9,4)]
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.config = branch_config
        self.branches = []
        
        # 自动分配通道数
        n_branches = len(branch_config)
        c_per_branch = filters // n_branches
        c_last = filters - c_per_branch * (n_branches - 1)
        
        for i, (k, d) in enumerate(branch_config):
            ch = c_last if i == n_branches - 1 else c_per_branch
            # 只有训练用的 Conv1D，无需 Buffer (搜索阶段只看准确率)
            self.branches.append(
                layers.Conv1D(ch, k, dilation_rate=d, padding='causal', activation='relu', 
                              kernel_regularizer=keras.regularizers.l2(l2_reg))
            )
            
        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=keras.regularizers.l2(l2_reg))

    def call(self, inputs):
        branch_outs = [conv(inputs) for conv in self.branches]
        x = tf.concat(branch_outs, axis=-1)
        return self.conv_out(x)

class DynamicSTGCN_Model(keras.Model):
    def __init__(self, hidden_dim, output_dim, branch_config, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.adapter = LeapMotionAdapter()
        self.A = get_adapter_adjacency_matrix()
        
        self.gcn1 = GraphConv(hidden_dim, self.A, l2_reg)
        # 使用动态 TCN
        self.tcn1 = DynamicMultiScaleTCN(hidden_dim, branch_config, l2_reg)
        self.drop1 = layers.Dropout(0.3)
        
        self.gcn2 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn2 = DynamicMultiScaleTCN(hidden_dim, branch_config, l2_reg)
        self.drop2 = layers.Dropout(0.3)
        
        self.dense_out = layers.Dense(output_dim, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.head = layers.Dense(output_dim, activation='softmax') # 临时的分类头，用于评估

    def call(self, inputs, training=False):
        x = self.adapter(inputs)
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; V = 21; C = 18 
        x = tf.reshape(x, (B * T, V, C))
        h1 = self.gcn1(x); h1 = tf.nn.relu(h1); h1 = tf.reshape(h1, (B, T, V, 64))
        h1_pool = tf.reduce_mean(h1, axis=2)
        
        t1 = self.tcn1(h1_pool); t1 = self.drop1(t1, training=training)
        t2 = self.tcn2(t1); t2 = self.drop2(t2, training=training)
        
        return self.dense_out(t2)

def build_search_model(hidden_dim, num_classes, branch_config, input_shape):
    inp = layers.Input(shape=input_shape)
    feat = DynamicSTGCN_Model(hidden_dim, 32, branch_config)(inp)
    # Global Pooling for Sequence Classification
    pool = layers.GlobalAveragePooling1D()(feat) 
    out = layers.Dense(num_classes, activation='softmax')(pool)
    return keras.Model(inp, out)

# --- 3. 辅助函数 ---
def calculate_receptive_field(config):
    # TCN 感受野公式: 1 + (kernel-1) * dilation
    # 因为分支是并行的，总感受野 = Max(各分支感受野)
    # 因为有两层 TCN 堆叠，总感受野 ≈ Layer1_RF + Layer2_RF - 1
    
    max_rf_single_layer = 0
    for k, d in config:
        rf = 1 + (k - 1) * d
        if rf > max_rf_single_layer:
            max_rf_single_layer = rf
            
    # 两层堆叠的近似总视野
    total_rf = max_rf_single_layer * 2 - 1 
    return max_rf_single_layer, total_rf

def load_data(base_path, class_map, max_len=150):
    sequences, labels = [], []
    for class_name, label in class_map.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path): continue
        for file_name in os.listdir(class_path):
            if file_name.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(class_path, file_name))
                    if df.empty: continue
                    sequences.append(df.select_dtypes(include=np.number).values)
                    labels.append(label)
                except: continue
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post'), np.array(labels)

def augment_data(data, noise_level=0.02):
    augmented_data = np.copy(data); mask = (data.sum(axis=2, keepdims=True) != 0)
    augmented_data += np.random.normal(0, noise_level, data.shape) * mask
    return augmented_data

# --- 4. 搜索主程序 ---
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    # === 定义搜索空间 (Candidates) ===
    # 格式: List of (kernel_size, dilation_rate)
    candidates = {
        "A (v13.4 Baseline)": [(3,1), (5,2), (9,4)], 
        "B (v13.5 Current)":  [(3,1), (5,2), (9,4), (9,8)],
        "C (Balanced 4)":     [(3,1), (5,2), (7,4), (9,6)],
        "D (Deep 5)":         [(3,1), (3,2), (5,4), (7,8), (9,16)],
        "E (Wide & Shallow)": [(3,1), (5,1), (7,1), (9,1)]
    }

    # 加载数据
    classes = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    
    print("Loading Data...")
    X_train_raw, y_train_raw = load_data(TRAIN_PATH, class_map)
    X_test_raw, y_test_raw = load_data(TEST_PATH, class_map)
    
    scaler = StandardScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[2]))
    X_train = scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[2])).reshape(X_train_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[2])).reshape(X_test_raw.shape)
    
    X_train_aug = augment_data(X_train)
    y_train = to_categorical(y_train_raw, len(classes))
    y_test = to_categorical(y_test_raw, len(classes))

    results = []

    print(f"\n🚀 开始架构搜索 (Total Candidates: {len(candidates)})")
    print(f"每轮训练 Epochs: {SEARCH_EPOCHS}")
    print("="*60)

    for name, config in candidates.items():
        tf.keras.backend.clear_session() # 清理显存
        
        single_rf, total_rf = calculate_receptive_field(config)
        print(f"\n🧪 Testing Candidate: {name}")
        print(f"   Config: {config}")
        print(f"   Receptive Field (Single Layer): {single_rf} frames")
        print(f"   Total Network RF (Approx): {total_rf} frames (~{total_rf/60:.2f}s)")
        
        model = build_search_model(64, len(classes), config, (None, X_train.shape[2]))
        
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time = time.time()
        
        # 使用 Early Stopping 防止过拟合，也可以节省时间
        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        
        hist = model.fit(
            X_train_aug, y_train,
            validation_data=(X_test, y_test),
            epochs=SEARCH_EPOCHS,
            batch_size=32,
            verbose=0, # 静默模式，只打印结果
            callbacks=[es]
        )
        
        duration = time.time() - start_time
        best_val_acc = max(hist.history['val_accuracy'])
        final_train_acc = hist.history['accuracy'][-1]
        
        print(f"   ✅ Result: Val Acc = {best_val_acc:.2%} | Time = {duration:.1f}s")
        
        results.append({
            "Name": name,
            "Branches": len(config),
            "Max Dilation": max([c[1] for c in config]),
            "Receptive Field (Frames)": total_rf,
            "Time (s)": round(duration, 1),
            "Val Accuracy": round(best_val_acc * 100, 2),
            "Train Accuracy": round(final_train_acc * 100, 2)
        })

    # --- 5. 结果汇总 ---
    print("\n" + "="*60)
    print("🏆 ARCHITECTURE LEADERBOARD 🏆")
    print("="*60)
    
    df_res = pd.DataFrame(results).sort_values(by="Val Accuracy", ascending=False)
    print(df_res.to_string(index=False))
    # 2. 【新增】保存到文件 (保存在 SAVE_PATH 目录下)
    csv_path = os.path.join(SAVE_PATH, "arch_search_results.csv")
    # 【新增这一行】 强制创建文件夹，如果不存在的话
    os.makedirs(SAVE_PATH, exist_ok=True)
    df_res.to_csv(csv_path, index=False)
    print(f"\n💾 详细结果已保存至: {csv_path}")
    best_arch = df_res.iloc[0]
    print("\n💡 Recommendation:")
    print(f"Based on the search, the best architecture is: [{best_arch['Name']}]")
    print(f"It achieved {best_arch['Val Accuracy']}% accuracy with a Receptive Field of {best_arch['Receptive Field (Frames)']} frames.")