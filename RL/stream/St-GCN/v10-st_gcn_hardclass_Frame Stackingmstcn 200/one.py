# -*- coding: utf-8 -*-
"""
归一化参数导出脚本
=====================================================
本脚本只运行一次，用于计算并保存训练数据的
StandardScaler（均值和方差），以便在Unity中进行
一模一样的数据预处理。
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import tensorflow as tf  # 导入tensorflow库并简写为tf

# --- 1. 从您的训练脚本中复制配置 ---
DATA_PATH = "D:/RuanJian/VsCode/Vscode_code/new_hmm/RL/stream/Data1"
TRAIN_PATH = os.path.join(DATA_PATH, "train")

if not os.path.isdir(TRAIN_PATH):
    raise FileNotFoundError(f"错误：找不到指定的训练数据路径: {TRAIN_PATH}")

GESTURE_CLASSES = sorted([f for f in os.listdir(TRAIN_PATH) if not f.startswith('.')])
NUM_CLASSES = len(GESTURE_CLASSES)
CLASS_TO_LABEL = {name: i for i, name in enumerate(GESTURE_CLASSES)}

# --- 2. 复制 load_data 函数 ---
def load_data(base_path, class_map, max_len=150):
    sequences, labels = [], []
    print(f"开始从 {base_path} 加载数据...")
    for class_name, label in class_map.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path): continue
        for file_name in os.listdir(class_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(class_path, file_name)
                try:
                    df = pd.read_csv(file_path)
                    if df.empty: continue
                    df_numeric = df.select_dtypes(include=np.number)
                    sequences.append(df_numeric.values)
                    labels.append(label)
                except (pd.errors.EmptyDataError, KeyError) as e:
                    print(f"警告: 处理文件 {file_path} 时出错: {e}，已跳过。")
                    continue
    print(f"发现 {len(sequences)} 个样本。")
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post'
    )
    return padded_sequences, np.array(labels)

# --- 3. 计算并保存Scaler参数 ---
if __name__ == "__main__":
    print("正在加载训练数据以计算Scaler...")
    # 只需要原始数据，不需要填充
    X_train_raw, _ = load_data(TRAIN_PATH, CLASS_TO_LABEL, max_len=9999) 
    
    nsamples, nsteps, nfeatures = X_train_raw.shape
    print(f"数据总帧数: {nsamples * nsteps}, 特征数: {nfeatures}")
    
    # 将数据重塑为2D
    X_train_reshaped = X_train_raw.reshape((nsamples * nsteps, nfeatures))
    
    # 拟合Scaler
    scaler = StandardScaler().fit(X_train_reshaped[X_train_reshaped.sum(axis=1) != 0])
    
    # 准备要导出的数据
    scaler_params = {
        "mean": scaler.mean_.tolist(),  # 均值
        "scale": scaler.scale_.tolist() # 标准差 (在StandardScaler中名为scale_)
    }
    
    # 核心修改：保存到 DATA_PATH 指定的文件夹
    save_path = os.path.join(DATA_PATH, "scaler_params4.json")
    with open(save_path, 'w') as f:
        json.dump(scaler_params, f)
        
    print(f"\n[成功] 归一化参数 (均值和标准差) 已保存到 {save_path}")
    print(f"请将此 scaler_params4.json 文件拖拽到您的Unity项目 Assets/Resources 文件夹中。")