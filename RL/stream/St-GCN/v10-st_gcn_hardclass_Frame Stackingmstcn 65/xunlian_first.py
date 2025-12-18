# -*- coding: utf-8 -*-
"""
Leap Motion ST-GCN 监督学习训练框架 (Phase 1 - Pretrain - v13.5 Super Large Field)
======================================================================
【v13.5 关键升级：比肩 LSTM 的时序能力】
1. [架构升级]：MS-TCN 升级为 4 分支结构。
2. [超大视野]：新增 Extra Large 分支 (k=9, dilation=8)。
   - 理论感受野：(9-1)*8 + 1 = 65 帧。
   - 时间跨度：约 1.1 秒 (60FPS下)。
   - 作用：能够捕捉长距离依赖（如完整的画圆、慢速手势），解决 "长时动作" 识别难点。

【包含组件】
1. LeapMotionAdapter (拓扑构建)
2. StreamingMultiScaleTCN (4分支多尺度时序卷积 - Max Dilation=8)
3. StreamingSTGCN_Model (核心特征提取)
4. ClassifierHead (分类层)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- 1. 全局配置 ---
DATA_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/Data1"
# 更新保存路径，以免覆盖旧模型
SAVE_MODEL_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/St-GCN/models_stgcn_v13_5_mstcn_65fps" 

TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test") 

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# 输出文件命名
FE_SAVE_NAME = os.path.join(SAVE_MODEL_PATH, "feature_extractor_stgcn.weights.h5")
CH_SAVE_NAME = os.path.join(SAVE_MODEL_PATH, "classifier_head_stgcn.h5")

L2_REG_VALUE = 1e-4

# --- 2. ST-GCN 核心组件 ---

class LeapMotionAdapter(layers.Layer):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.num_nodes = 21
        self.out_channels = 18 

    def call(self, inputs):
        palm_feat = inputs[..., 0:18] 
        bone_feat = inputs[..., 18:]  
        shape_back = tf.shape(bone_feat)[:-1]
        bone_feat_reshaped = tf.reshape(bone_feat, tf.concat([shape_back, [20, 10]], axis=0))
        palm_node = tf.expand_dims(palm_feat, axis=-2)
        bone_padding = tf.zeros(tf.concat([tf.shape(bone_feat_reshaped)[:-1], [8]], axis=0))
        bone_nodes = tf.concat([bone_feat_reshaped, bone_padding], axis=-1)
        return tf.concat([palm_node, bone_nodes], axis=-2)

def get_adapter_adjacency_matrix():
    num_nodes = 21
    edges = []
    for root in [1, 5, 9, 13, 17]: edges.append((0, root))
    for fs in [1, 5, 9, 13, 17]: 
        edges.append((fs, fs+1)); edges.append((fs+1, fs+2)); edges.append((fs+2, fs+3))
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges: A[i, j] = 1.0; A[j, i] = 1.0 
    A = A + np.eye(num_nodes)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)
    return tf.constant(np.dot(np.dot(D_mat_inv_sqrt, A), D_mat_inv_sqrt), dtype=tf.float32)

class GraphConv(layers.Layer):
    def __init__(self, out_channels, adjacency_matrix, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels; self.A = adjacency_matrix; self.l2_reg = keras.regularizers.l2(l2_reg)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.out_channels), initializer='glorot_uniform', regularizer=self.l2_reg, trainable=True)
    def call(self, inputs):
        x = tf.matmul(inputs, self.W)
        return tf.einsum('vw,bwc->bvc', self.A, x)

# === 【关键修改】 MS-TCN (65帧视野 - Super Large Field) ===
class StreamingMultiScaleTCN(layers.Layer):
    def __init__(self, filters, stride=1, l2_reg=1e-4, **kwargs):
        """
        MS-TCN v3 (Super Large Field - 4 Branches):
        Branch 1: k=3, d=1 (RF=3)  - 瞬时细节
        Branch 2: k=5, d=2 (RF=9)  - 短时特征
        Branch 3: k=9, d=4 (RF=33) - 中长时特征
        Branch 4: k=9, d=8 (RF=65) - 超长时特征 (约1.1秒，比肩LSTM)
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.l2_reg = keras.regularizers.l2(l2_reg)

        # 将 filters 分给 4 个分支
        c_branch = filters // 4
        # 确保总通道数匹配 (余数给最后一个分支)
        c_last = filters - (c_branch * 3)
        
        # Branch 1: Small (k=3, d=1) -> RF = 3
        self.conv_s = layers.Conv1D(c_branch, 3, strides=stride, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        
        # Branch 2: Mid (k=5, d=2) -> RF = 9
        self.conv_m = layers.Conv1D(c_branch, 5, strides=stride, dilation_rate=2, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        
        # Branch 3: Large (k=9, d=4) -> RF = 33
        self.conv_l = layers.Conv1D(c_branch, 9, strides=stride, dilation_rate=4, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)

        # Branch 4: Extra Large (k=9, d=8) -> RF = (9-1)*8 + 1 = 65
        self.conv_xl = layers.Conv1D(c_last, 9, strides=stride, dilation_rate=8, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)

        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=self.l2_reg)

    def build(self, input_shape):
        super().build(input_shape)
        # Buffer Size = (kernel_size - 1) * dilation_rate
        
        # k=3, d=1 -> buf=2
        self.buf_s = self.add_weight(name='buf_s', shape=(1, 2, input_shape[-1]), initializer='zeros', trainable=False)
        
        # k=5, d=2 -> buf=(5-1)*2 = 8
        self.buf_m = self.add_weight(name='buf_m', shape=(1, 8, input_shape[-1]), initializer='zeros', trainable=False)
        
        # k=9, d=4 -> buf=(9-1)*4 = 32
        self.buf_l = self.add_weight(name='buf_l', shape=(1, 32, input_shape[-1]), initializer='zeros', trainable=False)

        # k=9, d=8 -> buf=(9-1)*8 = 64
        self.buf_xl = self.add_weight(name='buf_xl', shape=(1, 64, input_shape[-1]), initializer='zeros', trainable=False)

    def reset_states(self):
        for buf in [self.buf_s, self.buf_m, self.buf_l, self.buf_xl]:
            buf.assign(tf.zeros_like(buf))

    def call(self, inputs, training=None):
        seq_len = tf.shape(inputs)[1]
        # 如果是训练(training=True) 或者 输入长度大于1(离线测试)，则使用序列模式
        is_sequence_processing = training or (seq_len > 1)

        if is_sequence_processing:
            x1 = self.conv_s(inputs)
            x2 = self.conv_m(inputs)
            x3 = self.conv_l(inputs)
            x4 = self.conv_xl(inputs)
        else:
            # Streaming Inference Mode (Online)
            # Branch 1
            curr_s = self.buf_s.value(); comb_s = tf.concat([curr_s, inputs], axis=1)
            x1 = self.conv_s(comb_s); self.buf_s.assign(comb_s[:, 1:, :]); x1 = x1[:, -1:, :]

            # Branch 2
            curr_m = self.buf_m.value(); comb_m = tf.concat([curr_m, inputs], axis=1)
            x2 = self.conv_m(comb_m); self.buf_m.assign(comb_m[:, 1:, :]); x2 = x2[:, -1:, :]

            # Branch 3
            curr_l = self.buf_l.value(); comb_l = tf.concat([curr_l, inputs], axis=1)
            x3 = self.conv_l(comb_l); self.buf_l.assign(comb_l[:, 1:, :]); x3 = x3[:, -1:, :]

            # Branch 4 (Extra Large)
            curr_xl = self.buf_xl.value(); comb_xl = tf.concat([curr_xl, inputs], axis=1)
            x4 = self.conv_xl(comb_xl); self.buf_xl.assign(comb_xl[:, 1:, :]); x4 = x4[:, -1:, :]

        x = tf.concat([x1, x2, x3, x4], axis=-1)
        return self.conv_out(x)

class StreamingSTGCN_Model(keras.Model):
    def __init__(self, hidden_dim=64, output_dim=32, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.adapter = LeapMotionAdapter()
        self.A = get_adapter_adjacency_matrix()
        
        self.gcn1 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn1 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) 
        self.drop1 = layers.Dropout(0.3)
        
        self.gcn2 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn2 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) 
        self.drop2 = layers.Dropout(0.3)
        
        self.dense_out = layers.Dense(output_dim, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))
    
    def reset_states(self):
        self.tcn1.reset_states()
        self.tcn2.reset_states()

    def call(self, inputs, training=False):
        x = self.adapter(inputs)
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; V = 21; C = 18 
        
        x = tf.reshape(x, (B * T, V, C))
        h1 = self.gcn1(x); h1 = tf.nn.relu(h1); h1 = tf.reshape(h1, (B, T, V, 64))
        h1_pool = tf.reduce_mean(h1, axis=2)
        
        t1 = self.tcn1(h1_pool, training=training); t1 = self.drop1(t1, training=training)
        t2 = self.tcn2(t1, training=training); t2 = self.drop2(t2, training=training)
        
        out = self.dense_out(t2)
        
        seq_len = tf.shape(inputs)[1]
        if training or seq_len > 1: return tf.reduce_mean(out, axis=1)
        return tf.squeeze(out, axis=1)

# --- 3. 辅助函数 ---
def load_data(base_path, class_map, max_len=200):
    sequences, labels = [], []
    print(f"Loading data from {base_path}...")
    
    if not os.path.exists(base_path):
        print(f"Warning: Path {base_path} not found!")
        return np.array([]), np.array([])

    for class_name, label in class_map.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path): continue
        
        files = [f for f in os.listdir(class_path) if f.endswith(".csv")]
        for file_name in files:
            try:
                df = pd.read_csv(os.path.join(class_path, file_name))
                if df.empty: continue
                seq = df.select_dtypes(include=np.number).values
                sequences.append(seq)
                labels.append(label)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue
                
    print(f"Found {len(sequences)} samples in {base_path}.")
    if len(sequences) == 0: return np.array([]), np.array([])
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post'), np.array(labels)

def augment_data(data, noise_level=0.02):
    if len(data) == 0: return data
    augmented_data = np.copy(data)
    mask = (data.sum(axis=2, keepdims=True) != 0)
    augmented_data += np.random.normal(0, noise_level, data.shape) * mask
    return augmented_data

def build_classifier_head(input_shape, num_classes):
    l2 = keras.regularizers.l2(L2_REG_VALUE)
    inp = layers.Input(shape=input_shape)
    d = layers.Dropout(0.5)(inp)
    d = layers.Dense(32, activation='relu', kernel_regularizer=l2)(d)
    out = layers.Dense(num_classes, activation='softmax')(d)
    return keras.Model(inputs=inp, outputs=out, name="ClassifierHead")

# --- 4. 主程序 ---
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    if not os.path.exists(TRAIN_PATH): raise FileNotFoundError(f"Train path not found: {TRAIN_PATH}")

    classes = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes: {classes}")

    X_train_raw, y_train_raw = load_data(TRAIN_PATH, class_map)
    X_test_raw, y_test_raw = load_data(TEST_PATH, class_map)

    if len(X_train_raw) > 0:
        scaler = StandardScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[2]))
        
        X_train = scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[2])).reshape(X_train_raw.shape)
        X_train_aug = augment_data(X_train)
        y_train = to_categorical(y_train_raw, num_classes)
        
        if len(X_test_raw) > 0:
            X_test = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[2])).reshape(X_test_raw.shape)
            y_test = to_categorical(y_test_raw, num_classes)
            validation_data = (X_test, y_test)
            print(f"Validation data ready: {X_test.shape}")
        else:
            validation_data = None
            print("Warning: No test data found. Training without validation.")
    else:
        raise ValueError("No training data found!")

    print("\nBuilding MS-STGCN Model (Super Large Field: 65 Frames / ~1.1s)...")
    
    stgcn_fe = StreamingSTGCN_Model(output_dim=32, l2_reg=L2_REG_VALUE)
    classifier_head = build_classifier_head((32,), num_classes)
    
    inputs = layers.Input(shape=(None, X_train.shape[2]))
    feats = stgcn_fe(inputs, training=True) 
    outputs = classifier_head(feats)
    
    full_model = keras.Model(inputs=inputs, outputs=outputs, name="Full_STGCN_Training")
    
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n[Phase 1] Starting Supervised Training (Super Large Field)...")
    
    callbacks = [
        EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=8, verbose=1),
        ModelCheckpoint(filepath=os.path.join(SAVE_MODEL_PATH, "best_model.weights.h5"), 
                        monitor='val_accuracy' if validation_data else 'accuracy',
                        save_best_only=True,
                        save_weights_only=True)
    ]
    
    full_model.fit(
        X_train_aug, y_train,
        validation_data=validation_data, 
        epochs=150, 
        batch_size=32,
        shuffle=True,
        callbacks=callbacks
    )
    
    print(f"\nSaving models to {SAVE_MODEL_PATH}...")
    stgcn_fe.save_weights(FE_SAVE_NAME)
    print(f"Feature Extractor weights saved: {FE_SAVE_NAME}")
    
    classifier_head.save(CH_SAVE_NAME)
    print(f"Classifier Head model saved: {CH_SAVE_NAME}")
    
    print("Phase 1 Complete. Ready for RL.")