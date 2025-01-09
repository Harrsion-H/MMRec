import torch
import numpy as np 
import pandas as pd
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载特征文件
npy_file = np.load(os.path.join(current_dir, 'image_feat.npy'))
npy2_file = np.load(os.path.join(current_dir, 'video_feat.npy'))
npy3_file = np.load(os.path.join(current_dir, 'text_feat.npy'))

# 加载映射文件
user = pd.read_csv(os.path.join(current_dir, 'u_id_mapping.csv'))
item = pd.read_csv(os.path.join(current_dir, 'i_id_mapping.csv'))

# 加载交互数据
inter = pd.read_csv(os.path.join(current_dir, 'microlens.inter'), sep='\t')

# 探索数据基本信息
print("图像特征维度:", npy_file.shape)
print("视频特征维度:", npy2_file.shape)
print("文本特征维度:", npy3_file.shape)
print("\n用户数量:", len(user))
print("物品数量:", len(item))
print("\n交互数据基本信息:")
print(inter.info())
print("\n交互数据前5行:")
print(inter.head())

# 统计标签分布并计算占比
print("\n数据集分布情况:")
label_counts = inter['x_label'].value_counts()
label_ratio = label_counts / len(inter) * 100

# 使用更清晰的格式输出
print("\n数据集分布统计:")
print("-" * 40)
print("标签  |  数量  |  占比")
print("-" * 40)
for label, count in label_counts.items():
    ratio = label_ratio[label]
    split_name = "训练集" if label == 0 else "验证集" if label == 1 else "测试集"
    print(f"{split_name:^6} | {count:^6d} | {ratio:^6.2f}%")
print("-" * 40)


# 检查时间戳范围
print("\n时间戳范围:")
print("最早时间:", pd.to_datetime(inter['timestamp'], unit='ms').min())
print("最晚时间:", pd.to_datetime(inter['timestamp'], unit='ms').max())
