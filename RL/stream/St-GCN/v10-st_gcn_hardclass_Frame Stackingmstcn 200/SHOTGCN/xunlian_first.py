# -*- coding: utf-8 -*-
"""
Leap Motion ST-GCN 监督学习训练框架 (Phase 1 - v14.3 "Nuclear" Anti-Overfit)
======================================================================
【v14.3 终极抗过拟合更新】
1. [关键] Label Smoothing (标签平滑): loss计算时引入 0.1 的平滑，禁止模型由100%的自信。
2. [关键] L2 正则化暴增: 从 1e-3 提升到 0.01 (1e-2)，强力限制权重。
3. [架构] 极致瘦身: hidden_dim 降至 32。
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
SAVE_MODEL_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/St-GCN/models_stgcn_v14_3_nuclear" 

TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test") 

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

FE_SAVE_NAME = os.path.join(SAVE_MODEL_PATH, "feature_extractor_stgcn.weights.h5")
CH_SAVE_NAME = os.path.join(SAVE_MODEL_PATH, "classifier_head_stgcn.h5")
BEST_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, "best_model.weights.h5")

# [UPDATE] L2 正则化强度提升 10 倍 (0.001 -> 0.01)
# 这会强迫模型权重非常小，极难过拟合
L2_REG_VALUE = 0.01

# --- 2. EMA 回调 ---
class CustomEMACallback(keras.callbacks.Callback):
    def __init__(self, decay=0.999):
        super(CustomEMACallback, self).__init__()
        self.decay = decay
        self.ema_weights = []

    def on_train_begin(self, logs=None):
        self.ema_weights = [tf.Variable(w, trainable=False) for w in self.model.trainable_variables]

    def on_train_batch_end(self, batch, logs=None):
        for weight, ema_weight in zip(self.model.trainable_variables, self.ema_weights):
            ema_weight.assign(self.decay * ema_weight + (1.0 - self.decay) * weight)

    def on_train_end(self, logs=None):
        print("\n[EMA] Applying EMA weights to model before saving final state...")
        for weight, ema_weight in zip(self.model.trainable_variables, self.ema_weights):
            weight.assign(ema_weight)

# --- 3. ST-GCN 核心组件 ---
class LeapMotionAdapter(layers.Layer):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
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

class RepGatedConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.l2_reg = keras.regularizers.l2(l2_reg)
        glu_filters = filters * 2
        self.conv_main = layers.Conv1D(glu_filters, kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_regularizer=self.l2_reg)
        self.bn_main = layers.BatchNormalization()
        self.conv_1x1 = layers.Conv1D(glu_filters, 1, padding='valid', kernel_regularizer=self.l2_reg)
        self.bn_1x1 = layers.BatchNormalization()

    def call(self, inputs, training=None):
        x_main = self.bn_main(self.conv_main(inputs), training=training)
        x_1x1 = self.bn_1x1(self.conv_1x1(inputs), training=training)
        x_sum = x_main + x_1x1
        a, b = tf.split(x_sum, 2, axis=-1)
        return a * tf.sigmoid(b)

class StreamingMultiScaleTCN(layers.Layer):
    def __init__(self, filters, stride=1, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.l2_reg = keras.regularizers.l2(l2_reg)
        c_branch = filters // 5
        c_last = filters - (c_branch * 4)
        self.rep_b1 = RepGatedConv1D(c_branch, 3, dilation_rate=1, l2_reg=l2_reg)
        self.rep_b2 = RepGatedConv1D(c_branch, 3, dilation_rate=2, l2_reg=l2_reg)
        self.rep_b3 = RepGatedConv1D(c_branch, 5, dilation_rate=4, l2_reg=l2_reg)
        self.rep_b4 = RepGatedConv1D(c_branch, 9, dilation_rate=8, l2_reg=l2_reg)
        self.rep_b5 = RepGatedConv1D(c_last, 9, dilation_rate=12, l2_reg=l2_reg)
        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=self.l2_reg)
        self.att_dense1 = layers.Dense(self.filters // 4, activation='relu', kernel_regularizer=self.l2_reg)
        self.att_dense2 = layers.Dense(self.filters, activation='sigmoid', kernel_regularizer=self.l2_reg)

    def build(self, input_shape):
        super().build(input_shape)
        self.buf_b1 = self.add_weight(name='buf_b1', shape=(1, 2, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b2 = self.add_weight(name='buf_b2', shape=(1, 4, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b3 = self.add_weight(name='buf_b3', shape=(1, 16, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b4 = self.add_weight(name='buf_b4', shape=(1, 64, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b5 = self.add_weight(name='buf_b5', shape=(1, 96, input_shape[-1]), initializer='zeros', trainable=False)

    def reset_states(self):
        for buf in [self.buf_b1, self.buf_b2, self.buf_b3, self.buf_b4, self.buf_b5]: buf.assign(tf.zeros_like(buf))

    def call(self, inputs, training=None):
        seq_len = tf.shape(inputs)[1]
        is_sequence_processing = training or (seq_len > 1)
        if is_sequence_processing:
            x1 = self.rep_b1(inputs, training=training)
            x2 = self.rep_b2(inputs, training=training)
            x3 = self.rep_b3(inputs, training=training)
            x4 = self.rep_b4(inputs, training=training)
            x5 = self.rep_b5(inputs, training=training)
        else:
            c1 = self.buf_b1.value(); combo1 = tf.concat([c1, inputs], axis=1)
            x1 = self.rep_b1(combo1, training=training); self.buf_b1.assign(combo1[:, 1:, :]); x1 = x1[:, -1:, :]
            c2 = self.buf_b2.value(); combo2 = tf.concat([c2, inputs], axis=1)
            x2 = self.rep_b2(combo2, training=training); self.buf_b2.assign(combo2[:, 1:, :]); x2 = x2[:, -1:, :]
            c3 = self.buf_b3.value(); combo3 = tf.concat([c3, inputs], axis=1)
            x3 = self.rep_b3(combo3, training=training); self.buf_b3.assign(combo3[:, 1:, :]); x3 = x3[:, -1:, :]
            c4 = self.buf_b4.value(); combo4 = tf.concat([c4, inputs], axis=1)
            x4 = self.rep_b4(combo4, training=training); self.buf_b4.assign(combo4[:, 1:, :]); x4 = x4[:, -1:, :]
            c5 = self.buf_b5.value(); combo5 = tf.concat([c5, inputs], axis=1)
            x5 = self.rep_b5(combo5, training=training); self.buf_b5.assign(combo5[:, 1:, :]); x5 = x5[:, -1:, :]
        x_concat = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        attention_score = self.att_dense2(self.att_dense1(x_concat))
        return self.conv_out(x_concat * attention_score)

class StreamingSTGCN_Model(keras.Model):
    def __init__(self, hidden_dim=64, output_dim=32, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.adapter = LeapMotionAdapter()
        self.A = get_adapter_adjacency_matrix()
        self.gcn1 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn1 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) 
        self.drop1 = layers.Dropout(0.5)
        self.gcn2 = GraphConv(hidden_dim, self.A, l2_reg)
        self.tcn2 = StreamingMultiScaleTCN(hidden_dim, l2_reg=l2_reg) 
        self.drop2 = layers.Dropout(0.5)
        self.dense_out = layers.Dense(output_dim, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))
    
    def reset_states(self):
        self.tcn1.reset_states(); self.tcn2.reset_states()

    def call(self, inputs, training=False):
        x = self.adapter(inputs); B = tf.shape(x)[0]; T = tf.shape(x)[1]; V = 21; C = 18 
        x = tf.reshape(x, (B * T, V, C))
        
        # GCN Block 1
        h1 = self.gcn1(x)
        h1 = tf.nn.relu(h1)
        # 动态 reshape，避免 64 写死的问题
        h1 = tf.reshape(h1, (B, T, V, self.hidden_dim))
        
        h1_pool = tf.reduce_mean(h1, axis=2)
        
        # TCN Blocks
        t1 = self.tcn1(h1_pool, training=training)
        t1 = self.drop1(t1, training=training)
        t2 = self.tcn2(t1, training=training)
        t2 = self.drop2(t2, training=training)
        
        out = self.dense_out(t2)
        
        seq_len = tf.shape(inputs)[1]
        if training or seq_len > 1: return tf.reduce_mean(out, axis=1)
        return tf.squeeze(out, axis=1)

# --- 4. 辅助函数 ---
def load_data(base_path, class_map, max_len=200):
    sequences, labels = [], []
    print(f"Loading data from {base_path}...")
    if not os.path.exists(base_path): return np.array([]), np.array([])
    for class_name, label in class_map.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path): continue
        for file_name in [f for f in os.listdir(class_path) if f.endswith(".csv")]:
            try:
                df = pd.read_csv(os.path.join(class_path, file_name))
                if df.empty: continue
                sequences.append(df.select_dtypes(include=np.number).values)
                labels.append(label)
            except: continue
    if len(sequences) == 0: return np.array([]), np.array([])
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post'), np.array(labels)

def augment_data_enhanced(data, labels, noise_level=0.02):
    """数据增强 (x4)"""
    if len(data) == 0: return data, labels
    aug_X, aug_Y = [], []
    for i in range(len(data)):
        seq = data[i]
        lbl = labels[i]
        aug_X.append(seq); aug_Y.append(lbl)
        
        mask = (seq.sum(axis=1, keepdims=True) != 0)
        noise = np.random.normal(0, noise_level, seq.shape)
        aug_X.append(seq + noise); aug_Y.append(lbl)
        
        scale = np.random.uniform(0.9, 1.1)
        aug_X.append(seq * scale); aug_Y.append(lbl)
        
        shift = np.random.randint(-3, 4)
        shifted = np.roll(seq, shift, axis=0)
        aug_X.append(shifted); aug_Y.append(lbl)
    return np.array(aug_X), np.array(aug_Y)

def build_classifier_head(input_shape, num_classes):
    l2 = keras.regularizers.l2(L2_REG_VALUE)
    inp = layers.Input(shape=input_shape)
    d = layers.Dropout(0.5)(inp)
    d = layers.Dense(32, activation='relu', kernel_regularizer=l2)(d)
    out = layers.Dense(num_classes, activation='softmax')(d)
    return keras.Model(inputs=inp, outputs=out, name="ClassifierHead")

# --- 5. 主程序 ---
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    if not os.path.exists(TRAIN_PATH): raise FileNotFoundError(f"Train path not found: {TRAIN_PATH}")

    classes = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    num_classes = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}
    print(f"Detected {num_classes} classes: {classes}")

    # 清理旧权重
    if os.path.exists(CH_SAVE_NAME):
        try:
            old_model = keras.models.load_model(CH_SAVE_NAME, compile=False)
            if old_model.output_shape[-1] != num_classes:
                os.remove(CH_SAVE_NAME); os.remove(FE_SAVE_NAME); os.remove(BEST_MODEL_PATH)
        except: pass

    X_train_raw, y_train_raw = load_data(TRAIN_PATH, class_map)
    X_test_raw, y_test_raw = load_data(TEST_PATH, class_map)

    if len(X_train_raw) > 0:
        scaler = StandardScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[2]))
        X_train = scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[2])).reshape(X_train_raw.shape)
        
        print("Applying Enhanced Data Augmentation (x4)...")
        X_train_aug, y_train_aug = augment_data_enhanced(X_train, y_train_raw)
        
        y_train_final = to_categorical(y_train_aug, num_classes)
        print(f"Training data shape after augmentation: {X_train_aug.shape}")
        
        if len(X_test_raw) > 0:
            X_test = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[2])).reshape(X_test_raw.shape)
            y_test = to_categorical(y_test_raw, num_classes)
            validation_data = (X_test, y_test)
        else:
            validation_data = None
    else:
        raise ValueError("No training data found!")

    print(f"\nBuilding MS-STGCN Model (v14.3 'Nuclear' Anti-Overfit)...")
    
    # [UPDATE] 极致瘦身: hidden_dim 降至 32
    stgcn_fe = StreamingSTGCN_Model(hidden_dim=32, output_dim=32, l2_reg=L2_REG_VALUE)
    classifier_head = build_classifier_head((32,), num_classes)
    
    inputs = layers.Input(shape=(None, X_train.shape[2]))
    feats = stgcn_fe(inputs, training=True) 
    outputs = classifier_head(feats)
    
    full_model = keras.Model(inputs=inputs, outputs=outputs, name="Full_STGCN_Training")
    
    # [UPDATE] 启用 label_smoothing = 0.1
    # 这会阻止 Training Accuracy 轻易达到 100%，从而强制模型学习通用特征
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn, 
        metrics=['accuracy']
    )

    print(f"\n[Phase 1] Starting Supervised Training...")
    
    callbacks = [
        EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=10, verbose=1),
        ModelCheckpoint(filepath=BEST_MODEL_PATH, 
                        monitor='val_accuracy' if validation_data else 'accuracy',
                        save_best_only=True,
                        save_weights_only=True),
        CustomEMACallback(decay=0.999)
    ]
    
    full_model.fit(
        X_train_aug, y_train_final, 
        validation_data=validation_data, 
        epochs=150, 
        batch_size=32, # 保持 32 以防 OOM
        shuffle=True,
        callbacks=callbacks
    )
    
    print(f"\nSaving models to {SAVE_MODEL_PATH}...")
    stgcn_fe.save_weights(FE_SAVE_NAME)
    classifier_head.save(CH_SAVE_NAME)
    print("Phase 1 Complete (v14.3).")