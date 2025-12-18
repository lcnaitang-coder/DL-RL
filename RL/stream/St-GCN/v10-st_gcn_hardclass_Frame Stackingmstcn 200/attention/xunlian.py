# -*- coding: utf-8 -*-
"""
Leap Motion 动态手势识别 RL 训练框架 (v13.6 Deep Balanced - 13 Classes - Original Config)
======================================================================
【配置说明】
1. [模型架构]：v13.6 (5分支 MS-TCN, Max Dilation=12, RF=193帧)。
2. [类别适配]：已添加对 13 个手势类别的支持。
3. [训练参数]：保持原版奖惩 (Penalty -15.0) 和轮次 (100 Updates)。
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import random
from collections import deque

# --- 1. 全局配置 ---
DATA_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/Data1"

# 指向 v13.6 预训练模型路径
PRETRAIN_MODEL_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/St-GCN/models_stgcn_v13_6_mstcn_193fps/attention" 
# RL 模型保存路径
SAVE_MODEL_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/St-GCN/models_rl_v13_6_mstcn_193fps/attention"

TRAIN_PATH = os.path.join(DATA_PATH, "train")
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# 权重路径
FE_WEIGHTS_PATH = os.path.join(PRETRAIN_MODEL_PATH, "feature_extractor_stgcn.weights.h5")
CH_MODEL_PATH = os.path.join(PRETRAIN_MODEL_PATH, "classifier_head_stgcn.h5")
PPO_ACTOR_PATH = os.path.join(SAVE_MODEL_PATH, "ppo_actor_policy_v13_6.weights.h5") 

L2_REG_VALUE = 1e-4

# 动作死线 (适配 13 类)
# 为新增的 10, 11, 12 号手势预留了 120 帧的默认值
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

# --- 2. ST-GCN 核心组件 (v13.6 5分支结构) ---

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

# === MS-TCN v13.6 (5 Branches) ===
# === [修正版] MS-TCN v13.7 with Attention (用于 RL 脚本) ===
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

        # [新增] 流式注意力机制权重 (这就是多出来的 4 个权重)
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

        # 1. 拼接
        x_concat = tf.concat([x1, x2, x3, x4, x5], axis=-1)

        # 2. [新增] 应用 Point-wise Channel Attention
        attention_score = self.att_dense1(x_concat)
        attention_score = self.att_dense2(attention_score) 
        
        # 3. 加权
        x_attended = x_concat * attention_score

        # 4. 输出
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

# --- 3. 辅助函数 ---
def load_data(base_path, class_map, max_len=200):
    sequences, labels = [], []
    print(f"Loading data from {base_path}...")
    for class_name, label in class_map.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path): continue
        for file_name in os.listdir(class_path):
            if file_name.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(class_path, file_name))
                    if df.empty: continue
                    sequences.append(df.select_dtypes(include=np.number).values); labels.append(label)
                except: continue
    print(f"Found {len(sequences)} samples.")
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post'), np.array(labels)

def augment_data(data, noise_level=0.02):
    augmented_data = np.copy(data); mask = (data.sum(axis=2, keepdims=True) != 0)
    augmented_data += np.random.normal(0, noise_level, data.shape) * mask
    return augmented_data

def build_classifier_head(input_shape, num_classes):
    l2 = keras.regularizers.l2(L2_REG_VALUE); inp = layers.Input(shape=input_shape); d = layers.Dropout(0.5)(inp)
    d = layers.Dense(32, activation='relu', kernel_regularizer=l2)(d); out = layers.Dense(num_classes, activation='softmax')(d)
    return keras.Model(inputs=inp, outputs=out, name="ClassifierHead")

# --- 4. RL 环境 (v13.3 Logic - 13 Classes) ---
class StreamingGestureEnv:
    def __init__(self, stgcn_feature_extractor, classifier_head, data, labels, global_max_len=200):
        self.feature_extractor = stgcn_feature_extractor; self.classifier_head = classifier_head
        self.data = data; self.labels = labels
        self.global_max_len = global_max_len; self.HISTORY_LEN = 10
        self.prob_history = deque(maxlen=self.HISTORY_LEN)
        
        self.STACK_SIZE = 5
        self.stack_buffer = deque(maxlen=self.STACK_SIZE)
        self.obs_dim = self.classifier_head.output.shape[1] * self.STACK_SIZE + 2 
        self.zero_obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        self.class_limits = CLASS_TIME_LIMITS
        self.class_triggers = CLASS_TRIGGER_START
        self.CONFIDENCE_THRESHOLD = 0.82 
        
        self.MAX_PERFECT_REWARD = 15.0; self.MIN_PERFECT_REWARD = 5.0
        self.MAX_LATE_REWARD = 2.0; self.MIN_LATE_REWARD = 0.5
        self.LATE_WAIT_PENALTY = -0.1; self.NORMAL_WAIT_REWARD = -0.01
        self.REWARD_MISSED = -20.0          
        
        self.curriculum_progress = 0.0; self.CURRICULUM_EPISODES = 6000 
        
        # [保持原设置]
        self.wrong_penalty_range = [-1.0, -15.0]; self.current_wrong_penalty = self.wrong_penalty_range[0]

        self.epoch_stats = {}; self.sample_indices_by_class = {}
        unique_labels = np.unique(labels)
        for label in unique_labels: self.sample_indices_by_class[label] = np.where(labels == label)[0]
        self._reset_internal_state()

    def update_curriculum(self, current_episode):
        progress = min(current_episode / self.CURRICULUM_EPISODES, 1.0)
        self.curriculum_progress = progress
        start, end = self.wrong_penalty_range
        self.current_wrong_penalty = start + (end - start) * progress
        if current_episode % 1000 == 0:
            print(f"[Curriculum] Ep:{current_episode} | Wrong Penalty: {self.current_wrong_penalty:.2f} | Progress: {progress:.1%}")

    def _reset_internal_state(self):
        self.current_idx = -1; self._current_sequence = None; self._true_label = -1
        self._current_step = 0; self._max_steps = 0; self._last_probs = None; self.prob_history.clear()
        self.stack_buffer.clear()

    def reset_epoch_stats(self):
        self.epoch_stats = {i: [0, 0] for i in self.sample_indices_by_class.keys()}

    def _process_frame(self):
        frame = self._current_sequence[self._current_step]; frame_input = frame.reshape(1, 1, -1)
        feat = self.feature_extractor(frame_input, training=False)
        probs = self.classifier_head(feat, training=False)
        self._last_probs = probs; self.prob_history.append(np.max(probs.numpy()))

    def _get_obs(self):
        current_probs = self._last_probs.numpy().flatten()
        self.stack_buffer.append(current_probs)
        while len(self.stack_buffer) < self.STACK_SIZE: self.stack_buffer.append(current_probs)
        stacked_probs = np.array(self.stack_buffer).flatten()
        norm_step = min(self._current_step / self.global_max_len, 1.0)
        prob_std = np.std(self.prob_history) if len(self.prob_history) >= 2 else 0.0
        return np.concatenate([stacked_probs, [norm_step], [prob_std]])

    def reset(self):
        self.current_idx = random.randint(0, len(self.data)-1)
        full_seq = self.data[self.current_idx]
        self._true_label = self.labels[self.current_idx]
        self.current_offset_limit = self.class_limits.get(self._true_label, 120) 
        self.current_trigger_start = self.class_triggers.get(self._true_label, 1)
        
        lens = np.where(full_seq.sum(axis=1)==0)[0]
        self._max_steps = lens[0] if len(lens)>0 else len(full_seq)
        if self._max_steps <= 12: return self.reset()
        
        self._current_sequence = full_seq[:self._max_steps]; self._current_step = 0
        self.prob_history.clear(); self.feature_extractor.reset_states()
        self.stack_buffer.clear() 
        self._process_frame()
        return self._get_obs()

    def step(self, action):
        probs = self._last_probs.numpy().flatten()
        pred = np.argmax(probs); conf = np.max(probs)
        is_too_early = self._current_step < self.current_trigger_start
        is_post_deadline = self._current_step > self.current_offset_limit

        done = False; reward = 0.0; is_success = False

        if action == 1: # TRIGGER
            done = True 
            if is_too_early:
                reward = -2.0 
            elif pred == self._true_label:
                if not is_post_deadline:
                    total_window = max(1, self.current_offset_limit - self.current_trigger_start)
                    time_left = max(0, self.current_offset_limit - self._current_step)
                    factor = time_left / total_window
                    reward = self.MIN_PERFECT_REWARD + (self.MAX_PERFECT_REWARD - self.MIN_PERFECT_REWARD) * factor
                    is_success = True
                else:
                    overdue_frames = self._current_step - self.current_offset_limit
                    late_factor = max(0.0, 1.0 - (overdue_frames / 20.0)) 
                    reward = self.MIN_LATE_REWARD + (self.MAX_LATE_REWARD - self.MIN_LATE_REWARD) * late_factor
                    is_success = True 
            else:
                reward = self.current_wrong_penalty * (1.0 + conf)
            self._update_stats(is_success)
            return self.zero_obs, reward, done, {}

        else: # WAIT
            self._current_step += 1
            if self._current_step >= self._max_steps - 1:
                done = True; self._process_frame()
                reward = self.REWARD_MISSED # -10
                self._update_stats(False)
                return self._get_obs(), reward, done, {}
            else:
                if is_post_deadline: reward = self.LATE_WAIT_PENALTY 
                else: reward = self.NORMAL_WAIT_REWARD 
                self._process_frame()
                return self._get_obs(), reward, done, {}

    def _update_stats(self, is_success):
        cls = self._true_label
        if cls in self.epoch_stats:
            self.epoch_stats[cls][1] += 1
            if is_success: self.epoch_stats[cls][0] += 1

# --- 5. PPO Agent (v13.3 NaN Protected) ---
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99; self.gae_lambda = 0.95; self.clip_epsilon = 0.2
        self.actor_optimizer = keras.optimizers.Adam(1e-4, clipnorm=0.5)
        self.critic_optimizer = keras.optimizers.Adam(3e-4, clipnorm=0.5)
        
        inp = layers.Input((state_dim,)); d1 = layers.Dense(128, activation="relu")(inp); d2 = layers.Dense(128, activation="relu")(d1)
        self.actor = keras.Model(inp, layers.Dense(action_dim, activation="sigmoid")(d2))
        
        inp_c = layers.Input((state_dim,)); cd1 = layers.Dense(128, activation="relu")(inp_c); cd2 = layers.Dense(128, activation="relu")(cd1)
        self.critic = keras.Model(inp_c, layers.Dense(1)(cd2))

    def get_action_and_value(self, state):
        state = tf.expand_dims(state, 0); probs = self.actor(state)
        probs = tf.clip_by_value(probs, 1e-7, 1.0 - 1e-7)
        val = self.critic(state)
        dist = tfp.distributions.Bernoulli(probs=probs); action = dist.sample()
        return action.numpy()[0][0], dist.log_prob(action).numpy()[0][0], val.numpy()[0][0]

    def train(self, s, a, r, logp, v, d):
        adv = np.zeros_like(r, dtype=np.float32); last_gae = 0; next_v = np.append(v[1:], 0)
        for t in reversed(range(len(r))):
            delta = r[t] + self.gamma * next_v[t] * (1-d[t]) - v[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1-d[t]) * last_gae; adv[t] = last_gae
        ret = adv + v[:-1]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        ds = tf.data.Dataset.from_tensor_slices((s, a, adv, ret, logp)).shuffle(len(s)).batch(64)
        al, cl = 0, 0; huber = keras.losses.Huber(delta=1.0)
        entropy_coef = 0.05
        update_count = 0 
        
        for _ in range(4):
            for bs, ba, badv, bret, blogp in ds:
                with tf.GradientTape() as tape:
                    probs = self.actor(bs)
                    probs = tf.clip_by_value(probs, 1e-7, 1.0 - 1e-7)
                    dist = tfp.distributions.Bernoulli(probs=probs)
                    new_logp = tf.reduce_sum(dist.log_prob(tf.reshape(ba, (-1,1))), axis=1); ratio = tf.exp(new_logp - blogp)
                    surr1 = ratio * badv; surr2 = tf.clip_by_value(ratio, 0.8, 1.2) * badv
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    entropy = tf.reduce_mean(dist.entropy())
                    loss = policy_loss - entropy_coef * entropy 
                
                if tf.math.is_nan(loss): continue
                grads = tape.gradient(loss, self.actor.trainable_variables)
                if any([tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None]): continue
                self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables)); al += loss.numpy()
                
                with tf.GradientTape() as tape: loss_c = huber(bret, self.critic(bs))
                if tf.math.is_nan(loss_c): continue 
                grads_c = tape.gradient(loss_c, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads_c, self.critic.trainable_variables)); cl += loss_c.numpy()
                update_count += 1
        return al/(update_count + 1e-8), cl/(update_count + 1e-8)

# --- 6. 主程序 ---
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    classes = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_map.items()}
    
    print(f"Detected {len(classes)} classes: {classes}")
    
    # 【关键修正】max_len 必须设为 200，否则数据会被截断
    print("Loading data with max_len=200...")
    X_train_raw, y_train_raw = load_data(TRAIN_PATH, class_map, max_len=200)
    y_train = to_categorical(y_train_raw, len(classes))
    
    scaler = StandardScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[2]))
    X_train = scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[2])).reshape(X_train_raw.shape)
    X_train_aug = augment_data(X_train)

    print(f"\n[Phase 2] Starting RL Training (v13.6 - {len(classes)} Classes)...")
    fe_rl = StreamingSTGCN_Model(output_dim=32, l2_reg=L2_REG_VALUE)
    dummy_stream_input = tf.zeros((1, 1, X_train.shape[2]))
    fe_rl(dummy_stream_input, training=False)
    
    print(f"Loading weights from: {FE_WEIGHTS_PATH}")
    fe_rl.load_weights(FE_WEIGHTS_PATH)
    fe_rl.trainable = False 
    
    ch_rl = keras.models.load_model(CH_MODEL_PATH, compile=False)
    ch_rl.trainable = False
    
    # 这里的 global_max_len 也要确保和数据一致
    env = StreamingGestureEnv(fe_rl, ch_rl, X_train_aug, y_train_raw, global_max_len=200)
    agent = PPOAgent(env.obs_dim, 1)
    
    # [保持原设置]
    MAX_UPDATES = 350 
    
    state = env.reset()
    all_rewards = []; ep_reward = 0; episodes = 0
    
    for i in range(MAX_UPDATES):
        env.reset_epoch_stats()
        start_episodes = episodes
        s_lst, a_lst, r_lst, lp_lst, v_lst, d_lst = [], [], [], [], [], []
        
        for _ in range(2048):
            a, lp, v = agent.get_action_and_value(state)
            ns, r, d, _ = env.step(a)
            s_lst.append(state); a_lst.append(a); r_lst.append(r); lp_lst.append(lp); v_lst.append(v); d_lst.append(d)
            state = ns; ep_reward += r
            if d:
                all_rewards.append(ep_reward); ep_reward = 0; state = env.reset(); episodes += 1
                env.update_curriculum(episodes)
        
        _, _, last_val = agent.get_action_and_value(state)
        al, cl = agent.train(np.array(s_lst), np.array(a_lst), np.array(r_lst), np.array(lp_lst), np.append(v_lst, last_val), np.array(d_lst))
        
        avg_r = np.mean(all_rewards[-50:]) if all_rewards else 0
        episodes_delta = episodes - start_episodes
        
        print(f"\n=== Update {i+1}/{MAX_UPDATES} Report ===")
        print(f"Episodes: {episodes} (+{episodes_delta}) | Avg Reward: {avg_r:.2f} | A_Loss: {al:.4f} | C_Loss: {cl:.4f}")
        
        total_samples_in_batch = sum([v[1] for v in env.epoch_stats.values()])
        sorted_stats = sorted(env.epoch_stats.items())
        line_buffer = ""
        for idx, (cls_id, (succ, total)) in enumerate(sorted_stats):
            if total == 0: continue
            cls_name = idx_to_class[cls_id]
            # 修复之前的语法错误
            acc = succ / total * 100
            info = f"[{cls_id}:{cls_name[:10]}] Acc:{acc:.0f}% (N={total})"
            line_buffer += f"{info:<30}"
            if (idx + 1) % 2 == 0: print(line_buffer); line_buffer = ""
        if line_buffer: print(line_buffer)
        print("-------------------------")
        
        if (i+1) % 10 == 0: agent.actor.save_weights(PPO_ACTOR_PATH)

    agent.actor.save_weights(PPO_ACTOR_PATH)
    print("v13.6 RL Training Complete.")