# -*- coding: utf-8 -*-
"""
ST-GCN v13.7 Voice Enhanced 模型转换脚本 (Keras -> ONNX)
==============================================================
适配: 
1. 带有 Point-wise Channel Attention 的新模型架构 (v13.7)
2. 带有 Voice Input 的 PPO Agent (Voice Version)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx
import os
import numpy as np

# --- 1. 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
STGCN_PATH = os.path.join(BASE_PATH, "models_stgcn_v13_6_mstcn_193fps/attention")
RL_PATH = os.path.join(BASE_PATH, "models_rl_v13_6_mstcn_193fps/voice") # Voice 目录
SAVE_PATH = RL_PATH 

FE_WEIGHTS_PATH = os.path.join(STGCN_PATH, "feature_extractor_stgcn.weights.h5")
CH_PATH = os.path.join(STGCN_PATH, "classifier_head_stgcn.h5")
RL_AGENT_PATH = os.path.join(RL_PATH, "ppo_actor_policy_v13_6.weights.h5") 

ONNX_FE_PATH = os.path.join(SAVE_PATH, "feature_extractor_v13_7_att.onnx")
ONNX_CH_PATH = os.path.join(SAVE_PATH, "classifier_head_v13_7.onnx")
ONNX_AGENT_PATH = os.path.join(SAVE_PATH, "ppo_actor_voice_v13_7.onnx") # Voice 专属名称

L2_REG_VALUE = 1e-4

# ============================================================================
# A. 原始训练模型定义 (必须与 xunlian_first.py 一致)
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

        # 这里的 conv_out 将处理经过 Attention 加权后的特征
        self.conv_out = layers.Conv1D(filters, 1, kernel_regularizer=self.l2_reg)

    def build(self, input_shape):
        super().build(input_shape)
        # Buffer 状态定义 (保持不变)
        self.buf_b1 = self.add_weight(name='buf_b1', shape=(1, 2, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b2 = self.add_weight(name='buf_b2', shape=(1, 4, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b3 = self.add_weight(name='buf_b3', shape=(1, 16, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b4 = self.add_weight(name='buf_b4', shape=(1, 64, input_shape[-1]), initializer='zeros', trainable=False)
        self.buf_b5 = self.add_weight(name='buf_b5', shape=(1, 96, input_shape[-1]), initializer='zeros', trainable=False)

        # [新增] 流式注意力机制权重
        # Point-wise Channel Attention: 独立计算每一帧的通道权重，不依赖未来信息
        self.att_dense1 = layers.Dense(self.filters // 4, activation='relu', kernel_regularizer=self.l2_reg)
        self.att_dense2 = layers.Dense(self.filters, activation='sigmoid', kernel_regularizer=self.l2_reg)

    def call(self, inputs, training=None):
        # 简化版 call (用于 build)
        x1 = self.conv_b1(inputs); x2 = self.conv_b2(inputs); x3 = self.conv_b3(inputs); x4 = self.conv_b4(inputs); x5 = self.conv_b5(inputs)
        x = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        att = self.att_dense1(x)
        att = self.att_dense2(att)
        return self.conv_out(x * att)

class StreamingSTGCN_Model_Train(keras.Model):
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
    
    def call(self, inputs, training=False):
        x = self.adapter(inputs); B = tf.shape(x)[0]; T = tf.shape(x)[1]; V = 21; C = 18 
        x = tf.reshape(x, (B * T, V, C)); h1 = self.gcn1(x); h1 = tf.nn.relu(h1); h1 = tf.reshape(h1, (B, T, V, 64))
        h1_pool = tf.reduce_mean(h1, axis=2); t1 = self.tcn1(h1_pool, training=training); t1 = self.drop1(t1, training=training)
        t2 = self.tcn2(t1, training=training); t2 = self.drop2(t2, training=training); out = self.dense_out(t2)
        if inputs.shape[1] is not None and inputs.shape[1] > 1: return tf.reduce_mean(out, axis=1)
        return tf.squeeze(out, axis=1)

# ============================================================================
# B. 导出专用模型结构 (加入 Attention)
# ============================================================================

class LeapMotionAdapter_Static(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, inputs):
        palm_feat = inputs[..., 0:18]; bone_feat = inputs[..., 18:]
        bone_feat_reshaped = tf.reshape(bone_feat, (1, 1, 20, 10))
        palm_node = tf.expand_dims(palm_feat, axis=-2)
        bone_padding = tf.zeros((1, 1, 20, 8))
        bone_nodes = tf.concat([bone_feat_reshaped, bone_padding], axis=-1)
        return tf.concat([palm_node, bone_nodes], axis=-2)

class GraphConv_Static(layers.Layer):
    def __init__(self, out_channels, adjacency_matrix, **kwargs):
        super().__init__(**kwargs); self.out_channels = out_channels; self.A_init = adjacency_matrix 
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.out_channels), initializer='glorot_uniform', trainable=True)
        A_reshaped_np = tf.reshape(self.A_init, (1, 21, 21)).numpy()
        self.A = self.add_weight(name="adj", shape=(1, 21, 21), initializer=tf.constant_initializer(A_reshaped_np), trainable=False)
    def call(self, inputs):
        x = tf.matmul(inputs, self.W); return tf.matmul(self.A, x)

class GraphPooling_Static(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        num_nodes = 21; val = 1.0 / float(num_nodes)
        init_val = np.full((1, 1, num_nodes), val, dtype=np.float32)
        self.pool_matrix = self.add_weight(name="pool_mat", shape=(1, 1, num_nodes), initializer=tf.constant_initializer(init_val), trainable=False)
    def call(self, inputs): return tf.matmul(self.pool_matrix, inputs)

class StreamingMSTCN_Export(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        c_branch = filters // 5; c_last = filters - (c_branch * 4)
        
        self.conv_b1 = layers.Conv1D(c_branch, 3, strides=stride, padding='valid', activation='relu')
        self.conv_b2 = layers.Conv1D(c_branch, 3, strides=stride, dilation_rate=2, padding='valid', activation='relu')
        self.conv_b3 = layers.Conv1D(c_branch, 5, strides=stride, dilation_rate=4, padding='valid', activation='relu')
        self.conv_b4 = layers.Conv1D(c_branch, 9, strides=stride, dilation_rate=8, padding='valid', activation='relu')
        self.conv_b5 = layers.Conv1D(c_last, 9, strides=stride, dilation_rate=12, padding='valid', activation='relu')
        
        self.att_dense1 = layers.Dense(self.filters // 4, activation='relu')
        self.att_dense2 = layers.Dense(self.filters, activation='sigmoid')
        self.conv_out = layers.Conv1D(filters, 1)

    def call(self, inputs, buffers):
        comb1 = tf.concat([buffers[0], inputs], axis=1); x1 = self.conv_b1(comb1); new_b1 = comb1[:, 1:, :]
        comb2 = tf.concat([buffers[1], inputs], axis=1); x2 = self.conv_b2(comb2); new_b2 = comb2[:, 1:, :]
        comb3 = tf.concat([buffers[2], inputs], axis=1); x3 = self.conv_b3(comb3); new_b3 = comb3[:, 1:, :]
        comb4 = tf.concat([buffers[3], inputs], axis=1); x4 = self.conv_b4(comb4); new_b4 = comb4[:, 1:, :]
        comb5 = tf.concat([buffers[4], inputs], axis=1); x5 = self.conv_b5(comb5); new_b5 = comb5[:, 1:, :]
        
        merged = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        att = self.att_dense1(merged)
        att = self.att_dense2(att)
        merged_att = merged * att
        final_out = self.conv_out(merged_att)
        
        return final_out, [new_b1, new_b2, new_b3, new_b4, new_b5]

class STGCN_Export_Model_v13(keras.Model):
    def __init__(self, hidden_dim=64, output_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.adapter = LeapMotionAdapter_Static()
        self.A = get_adapter_adjacency_matrix()
        self.gcn1 = GraphConv_Static(hidden_dim, self.A)
        self.pooling = GraphPooling_Static() 
        self.tcn1 = StreamingMSTCN_Export(hidden_dim)
        self.tcn2 = StreamingMSTCN_Export(hidden_dim)
        self.dense_out = layers.Dense(output_dim, activation='tanh')

    def call(self, input_list):
        input_frame = input_list[0]
        l1_bufs = input_list[1:6]
        l2_bufs = input_list[6:11]
        
        x = self.adapter(input_frame)
        x_reshaped = tf.reshape(x, (1, 21, 18))
        h1 = self.gcn1(x_reshaped)
        h1 = tf.nn.relu(h1)
        h1_pooled = self.pooling(h1) 
        
        t1, l1_bufs_new = self.tcn1(h1_pooled, l1_bufs)
        t2, l2_bufs_new = self.tcn2(t1, l2_bufs)
        
        out = self.dense_out(t2)
        out = tf.squeeze(out, axis=1)
        
        return [out] + l1_bufs_new + l2_bufs_new

# --- 4. 执行转换 ---

def convert_fe_v13():
    print("\n[1/3] 转换 ST-GCN v13.7 (Attention Enhanced)...")
    if not os.path.exists(FE_WEIGHTS_PATH): print(f"❌ 找不到权重文件: {FE_WEIGHTS_PATH}"); return

    train_model = StreamingSTGCN_Model_Train(output_dim=32)
    train_model(tf.zeros((1, 200, 218))) 
    try: train_model.load_weights(FE_WEIGHTS_PATH); print("  - 训练权重加载成功。")
    except Exception as e: print(f"  x 权重加载失败: {e}"); return

    export_model = STGCN_Export_Model_v13(output_dim=32)
    buf_sizes = [2, 4, 16, 64, 96]
    dummy_bufs = [tf.zeros((1, s, 64)) for s in buf_sizes]
    export_model([tf.zeros((1, 1, 218))] + dummy_bufs + dummy_bufs)
    
    def transfer_mstcn(src_layer, dest_layer):
        dest_layer.conv_b1.set_weights(src_layer.conv_b1.get_weights())
        dest_layer.conv_b2.set_weights(src_layer.conv_b2.get_weights())
        dest_layer.conv_b3.set_weights(src_layer.conv_b3.get_weights())
        dest_layer.conv_b4.set_weights(src_layer.conv_b4.get_weights())
        dest_layer.conv_b5.set_weights(src_layer.conv_b5.get_weights())
        dest_layer.att_dense1.set_weights(src_layer.att_dense1.get_weights())
        dest_layer.att_dense2.set_weights(src_layer.att_dense2.get_weights())
        dest_layer.conv_out.set_weights(src_layer.conv_out.get_weights())

    def transfer_gcn(src_layer, dest_layer):
        dest_layer.W.assign(src_layer.get_weights()[0])

    transfer_gcn(train_model.gcn1, export_model.gcn1)
    transfer_mstcn(train_model.tcn1, export_model.tcn1)
    transfer_mstcn(train_model.tcn2, export_model.tcn2)
    export_model.dense_out.set_weights(train_model.dense_out.get_weights())

    spec = [tf.TensorSpec((1, 1, 218), tf.float32, name="input_frame")]
    for i, size in enumerate(buf_sizes): spec.append(tf.TensorSpec((1, size, 64), tf.float32, name=f"l1_b{i+1}_in"))
    for i, size in enumerate(buf_sizes): spec.append(tf.TensorSpec((1, size, 64), tf.float32, name=f"l2_b{i+1}_in"))
        
    onnx_model, _ = tf2onnx.convert.from_keras(export_model, input_signature=[spec], opset=11)
    with open(ONNX_FE_PATH, "wb") as f: f.write(onnx_model.SerializeToString())
    print(f"✅ [成功] ST-GCN v13.7 已保存: {ONNX_FE_PATH}")

def convert_ch():
    print("\n[2/3] 转换分类头 (13 Classes)...")
    if not os.path.exists(CH_PATH): print(f"❌ 找不到文件: {CH_PATH}"); return
    model = keras.models.load_model(CH_PATH, compile=False)
    spec = (tf.TensorSpec((1, 32), tf.float32, name="input_feat"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11)
    with open(ONNX_CH_PATH, "wb") as f: f.write(onnx_model.SerializeToString())
    print(f"✅ [成功] 分类头已保存: {ONNX_CH_PATH}")

def convert_actor():
    print("\n[3/3] 转换 PPO Voice Actor (81-dim Input)...")
    if not os.path.exists(RL_AGENT_PATH): print(f"❌ 找不到文件: {RL_AGENT_PATH}"); return
    
    # [更新] Voice Agent Input Dim
    # Gesture: 13 * 5 + 2 = 67
    # Audio: 13 + 1 = 14
    # Total: 81
    state_dim = 81 
    
    inp = layers.Input((state_dim,)); d1 = layers.Dense(128, activation="relu")(inp); d2 = layers.Dense(128, activation="relu")(d1); out = layers.Dense(1, activation="sigmoid")(d2); actor_model = keras.Model(inp, out)
    try: actor_model.load_weights(RL_AGENT_PATH); print("  - Actor 权重加载成功。")
    except Exception as e: print(f"  x Actor 权重失败: {e}"); return
    spec = (tf.TensorSpec((1, state_dim), tf.float32, name="input_state"),)
    onnx_model, _ = tf2onnx.convert.from_keras(actor_model, input_signature=spec, opset=11)
    with open(ONNX_AGENT_PATH, "wb") as f: f.write(onnx_model.SerializeToString())
    print(f"✅ [成功] PPO Voice Actor 已保存: {ONNX_AGENT_PATH}")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]
    
    convert_fe_v13()
    convert_ch()
    convert_actor()
    print("\n全部转换完成。ONNX 文件已保存在 Voice RL 训练结果目录中。")
