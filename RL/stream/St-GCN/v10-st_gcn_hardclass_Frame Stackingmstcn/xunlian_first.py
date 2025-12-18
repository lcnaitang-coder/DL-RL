# -*- coding: utf-8 -*-
"""
Leap Motion ST-GCN 监督学习训练框架 (Phase 1 - Pretrain)
======================================================================
【v13.0 更新说明】
1. 引入 MS-TCN (Multi-Scale TCN)：并行使用 k=3, k=5, k=9(dilated) 卷积核。
   - 解决动态手势（如画圆）与静态手势（如握拳）混淆的问题。
   - 视野从 9 帧扩大到 17 帧。
2. 加入验证集：加载 test 文件夹数据用于监控过拟合。

【包含组件】
1. LeapMotionAdapter (拓扑构建)
2. StreamingMultiScaleTCN (多尺度时序卷积 - NEW)
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
SAVE_MODEL_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/models_stgcn_v13_mstcn" # 建议升级版本号

TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test") # === 修改点：新增测试集路径 ===

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# 输出文件命名
FE_SAVE_NAME = os.path.join(SAVE_MODEL_PATH, "feature_extractor_stgcn.weights.h5")
CH_SAVE_NAME = os.path.join(SAVE_MODEL_PATH, "classifier_head_stgcn.h5")

L2_REG_VALUE = 1e-4

# --- 2. ST-GCN 核心组件 ---

class LeapMotionAdapter(layers.Layer):
    """
    将 Leap Motion 的 218 维平铺数据转换为 (21, 18) 的图结构数据
    """
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.num_nodes = 21
        self.out_channels = 18 

    def call(self, inputs):
        # inputs shape: (B, T, 218)
        palm_feat = inputs[..., 0:18] # (B, T, 18)
        bone_feat = inputs[..., 18:]  # (B, T, 200)
        
        # 重塑骨骼特征: 200 -> 20个手指节点 * 10维特征
        shape_back = tf.shape(bone_feat)[:-1] # (B, T)
        bone_feat_reshaped = tf.reshape(bone_feat, tf.concat([shape_back, [20, 10]], axis=0))
        
        # 扩展手掌节点: (B, T, 18) -> (B, T, 1, 18)
        palm_node = tf.expand_dims(palm_feat, axis=-2)
        
        # 填充骨骼节点: 10维 -> 18维 (补8个0)
        # === 注意：如果有速度特征优化，建议在这里填入计算出的速度，目前保持补0 ===
        bone_padding = tf.zeros(tf.concat([tf.shape(bone_feat_reshaped)[:-1], [8]], axis=0))
        bone_nodes = tf.concat([bone_feat_reshaped, bone_padding], axis=-1)
        
        # 拼接: 手掌(1) + 手指(20) = 21个节点
        return tf.concat([palm_node, bone_nodes], axis=-2)

def get_adapter_adjacency_matrix():
    num_nodes = 21
    edges = []
    # 掌心(0) 到 手指根部(1, 5, 9, 13, 17)
    for root in [1, 5, 9, 13, 17]: edges.append((0, root))
    # 手指内部连接
    for fs in [1, 5, 9, 13, 17]: 
        edges.append((fs, fs+1))
        edges.append((fs+1, fs+2))
        edges.append((fs+2, fs+3))
    
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges: 
        A[i, j] = 1.0
        A[j, i] = 1.0 
    
    # 归一化邻接矩阵 (D^-0.5 * (A+I) * D^-0.5)
    A = A + np.eye(num_nodes)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)
    return tf.constant(np.dot(np.dot(D_mat_inv_sqrt, A), D_mat_inv_sqrt), dtype=tf.float32)

class GraphConv(layers.Layer):
    def __init__(self, out_channels, adjacency_matrix, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.A = adjacency_matrix
        self.l2_reg = keras.regularizers.l2(l2_reg)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.out_channels), 
                                 initializer='glorot_uniform', 
                                 regularizer=self.l2_reg, 
                                 trainable=True)
    
    def call(self, inputs):
        # inputs: (B, T, V, C)
        # Graph Convolution formula: A * X * W
        x = tf.matmul(inputs, self.W) # (B, T, V, out_channels)
        return tf.einsum('vw,bwc->bvc', self.A, x) # (B, T, V, out_channels)

# === 修改点：全新的 Multi-Scale TCN 类 ===
class StreamingMultiScaleTCN(layers.Layer):
    def __init__(self, filters, stride=1, l2_reg=1e-4, **kwargs):
        """
        MS-TCN: 多尺度时序卷积
        Branch 1: k=3 (细节)
        Branch 2: k=5 (短时序)
        Branch 3: k=9, dilation=2 (长时序, 感受野=17)
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.l2_reg = keras.regularizers.l2(l2_reg)

        # 通道分配：将总 filters 分配给三个分支
        c_small = filters // 4
        c_mid = filters // 4
        c_large = filters - c_small - c_mid
        
        # 定义三个并行的 1D 卷积
        self.conv_small = layers.Conv1D(c_small, 3, strides=stride, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        self.conv_mid = layers.Conv1D(c_mid, 5, strides=stride, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)
        # 注意：这里使用了 dilation_rate=2，感受野扩大
        self.conv_large = layers.Conv1D(c_large, 9, strides=stride, dilation_rate=2, padding='causal', activation='relu', kernel_regularizer=self.l2_reg)

        # 1x1 卷积用于特征融合
        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=self.l2_reg)

    def build(self, input_shape):
        super().build(input_shape)
        # 为流式推理创建 Buffer
        # Buffer Size = (kernel_size - 1) * dilation_rate
        # k=3, d=1 -> buf=2
        self.buf_small = self.add_weight(name='buf_s', shape=(1, 2, input_shape[-1]), initializer='zeros', trainable=False)
        # k=5, d=1 -> buf=4
        self.buf_mid   = self.add_weight(name='buf_m', shape=(1, 4, input_shape[-1]), initializer='zeros', trainable=False)
        # k=9, d=2 -> buf=(9-1)*2=16
        self.buf_large = self.add_weight(name='buf_l', shape=(1, 16, input_shape[-1]), initializer='zeros', trainable=False)

    def reset_states(self):
        for buf in [self.buf_small, self.buf_mid, self.buf_large]:
            buf.assign(tf.zeros_like(buf))

    def call(self, inputs, training=None):
        # 判断模式：
        # 1. 如果 training=True -> 训练模式 (处理整段序列)
        # 2. 如果 training=False 但 inputs 长度 > 1 -> 验证模式 (处理整段序列)
        # 3. 否则 -> 流式推理模式 (处理单帧)
        
        seq_len = tf.shape(inputs)[1]
        is_sequence_processing = training or (seq_len > 1)

        if is_sequence_processing:
            # --- 批处理模式 (Train/Val) ---
            x1 = self.conv_small(inputs)
            x2 = self.conv_mid(inputs)
            x3 = self.conv_large(inputs)
        else:
            # --- 流式模式 (RL Inference) ---
            # Branch 1
            curr_s = self.buf_small.value()
            comb_s = tf.concat([curr_s, inputs], axis=1)
            x1 = self.conv_small(comb_s)
            self.buf_small.assign(comb_s[:, 1:, :])
            x1 = x1[:, -1:, :]

            # Branch 2
            curr_m = self.buf_mid.value()
            comb_m = tf.concat([curr_m, inputs], axis=1)
            x2 = self.conv_mid(comb_m)
            self.buf_mid.assign(comb_m[:, 1:, :])
            x2 = x2[:, -1:, :]

            # Branch 3
            curr_l = self.buf_large.value()
            comb_l = tf.concat([curr_l, inputs], axis=1)
            x3 = self.conv_large(comb_l)
            self.buf_large.assign(comb_l[:, 1:, :])
            x3 = x3[:, -1:, :]

        # 融合
        x = tf.concat([x1, x2, x3], axis=-1)
        return self.conv_out(x)

class StreamingSTGCN_Model(keras.Model):
    def __init__(self, hidden_dim=64, output_dim=32, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.adapter = LeapMotionAdapter()
        self.A = get_adapter_adjacency_matrix()
        
        self.gcn1 = GraphConv(hidden_dim, self.A, l2_reg)
        # === 修改点：替换为 MS-TCN ===
        self.tcn1 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) 
        self.drop1 = layers.Dropout(0.3)
        
        self.gcn2 = GraphConv(hidden_dim, self.A, l2_reg)
        # === 修改点：替换为 MS-TCN ===
        self.tcn2 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) 
        self.drop2 = layers.Dropout(0.3)
        
        self.dense_out = layers.Dense(output_dim, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))
    
    def reset_states(self):
        self.tcn1.reset_states()
        self.tcn2.reset_states()

    def call(self, inputs, training=False):
        x = self.adapter(inputs)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        V = 21
        C = 18 
        
        # 1. GCN Block 1
        x = tf.reshape(x, (B * T, V, C))
        h1 = self.gcn1(x)
        h1 = tf.nn.relu(h1)
        h1 = tf.reshape(h1, (B, T, V, 64))
        h1_pool = tf.reduce_mean(h1, axis=2) # Pooling Nodes -> (B, T, 64)
        
        # 2. TCN Block 1 (MS-TCN)
        t1 = self.tcn1(h1_pool, training=training)
        t1 = self.drop1(t1, training=training)
        
        # 3. GCN Block 2 (Logic reused GCN1 output logic, simplifying to TCN-only structure for 2nd block as per original)
        # 注意：原代码 tcn2 接收的是 t1。这里保持原逻辑不变，即堆叠 TCN。
        # 如果需要更深的 GCN，可以在这里加 GCN 层。目前保持你原来的结构：GCN->Pool->TCN->TCN
        
        # 4. TCN Block 2 (MS-TCN)
        t2 = self.tcn2(t1, training=training)
        t2 = self.drop2(t2, training=training)
        
        out = self.dense_out(t2)
        
        # 判断是否需要 reduce_mean
        # 如果是训练(training=True) 或者 输入长度 > 1 (验证集)，则返回序列平均
        seq_len = tf.shape(inputs)[1]
        if training or seq_len > 1:
            return tf.reduce_mean(out, axis=1)
        
        # 否则是流式推理，返回单帧结果
        return tf.squeeze(out, axis=1)

# --- 3. 辅助函数 ---
def load_data(base_path, class_map, max_len=200):
    sequences, labels = [], []
    print(f"Loading data from {base_path}...")
    
    # 检查路径是否存在
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
                # 仅选择数值列
                seq = df.select_dtypes(include=np.number).values
                sequences.append(seq)
                labels.append(label)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue
                
    print(f"Found {len(sequences)} samples in {base_path}.")
    if len(sequences) == 0:
        return np.array([]), np.array([])
        
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post'), np.array(labels)

def augment_data(data, noise_level=0.02):
    if len(data) == 0: return data
    augmented_data = np.copy(data)
    # 只对非填充部分加噪声 (假设全是0的是填充)
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

    # 1. 确定类别映射
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train path not found: {TRAIN_PATH}")

    classes = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes: {classes}")

    # 2. 加载训练集
    X_train_raw, y_train_raw = load_data(TRAIN_PATH, class_map)
    
    # 3. 加载测试集 (验证集) === 修改点 ===
    X_test_raw, y_test_raw = load_data(TEST_PATH, class_map)

    # 4. 数据预处理
    if len(X_train_raw) > 0:
        # Fit Scaler ONLY on Training data
        scaler = StandardScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[2]))
        
        # Transform Train
        X_train = scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[2])).reshape(X_train_raw.shape)
        X_train_aug = augment_data(X_train)
        y_train = to_categorical(y_train_raw, num_classes)
        
        # Transform Test (Apply same scaler) === 修改点 ===
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

    # 5. 模型构建
    print("\nBuilding MS-STGCN Model (Multi-Scale)...")
    
    # 5.1 特征提取器
    stgcn_fe = StreamingSTGCN_Model(output_dim=32, l2_reg=L2_REG_VALUE)
    
    # 5.2 分类头
    classifier_head = build_classifier_head((32,), num_classes)
    
    # 5.3 组合模型
    inputs = layers.Input(shape=(None, X_train.shape[2]))
    # training=True 确保 Dropout 开启且 MS-TCN 使用并行模式
    feats = stgcn_fe(inputs, training=True) 
    outputs = classifier_head(feats)
    
    full_model = keras.Model(inputs=inputs, outputs=outputs, name="Full_STGCN_Training")
    
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n[Phase 1] Starting Supervised Training (MS-TCN Enhanced)...")
    
    # 回调函数
    callbacks = [
        # 如果有验证集，监控 val_loss，否则监控 loss
        EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=8, verbose=1),
        # 可选：保存最佳模型
        ModelCheckpoint(filepath=os.path.join(SAVE_MODEL_PATH, "best_model.weights.h5"), 
                        monitor='val_accuracy' if validation_data else 'accuracy',
                        save_best_only=True,
                        save_weights_only=True)
    ]
    
    full_model.fit(
        X_train_aug, y_train,
        validation_data=validation_data, # === 修改点：传入验证集 ===
        epochs=150, # 适当增加 Epoch，因为模型变复杂了
        batch_size=32,
        shuffle=True,
        callbacks=callbacks
    )
    
    # 6. 保存权重
    print(f"\nSaving models to {SAVE_MODEL_PATH}...")
    
    # 保存特征提取器权重
    stgcn_fe.save_weights(FE_SAVE_NAME)
    print(f"Feature Extractor weights saved: {FE_SAVE_NAME}")
    
    # 保存分类头模型
    classifier_head.save(CH_SAVE_NAME)
    print(f"Classifier Head model saved: {CH_SAVE_NAME}")
    
    print("Phase 1 Complete. Ready for RL.")