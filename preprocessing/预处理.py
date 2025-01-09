# src/preprocess.py
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
"""
# Common Features
USER_ID_FIELD: userID
ITEM_ID_FIELD: itemID
#RATING_FIELD: rating
TIME_FIELD: timestamp

filter_out_cod_start_users: True

inter_file_name: 'microlens.inter'

# name of features
vision_feature_file: 'image_feat.npy'
text_feature_file: 'text_feat.npy'
user_graph_dict_file: 'user_graph_dict.npy'

field_separator: "\t"
"""

# 定义路径配置
BASE_DIR = Path("data")  # 基础数据目录
SOURCE_DIR = BASE_DIR / "microEnhancing"  # 源数据目录
PROCESSED_DIR = BASE_DIR / "processed"  # 处理后的数据目录

def process_data():
    """
    数据预处理函数
    
    功能:
    1. 加载原始数据并转换为所需格式 (userID, itemID, rating, timestamp, x_label)
    2. 保存为microlens.inter文件
    3. 处理并保存多模态特征
    """
    print("加载数据...")
    # 加载用户-物品交互数据
    df = pd.read_csv(SOURCE_DIR / "train_set.csv")
    
    # 对每个用户的交互按时间戳进行8:1:1分类
    df['x_label'] = 0  # 默认标签为0

    
    # 重新映射user_id和item_id
    unique_users = sorted(df['user'].unique())
    unique_items = sorted(df['item'].unique())
    
    user_map = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_users)}
    item_map = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_items)}
    
    # 创建处理后的数据目录
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 将映射表保存为CSV
    user_map_df = pd.DataFrame(list(user_map.items()), columns=['old_id', 'new_id'])
    user_map_df.to_csv(PROCESSED_DIR / 'user_map.csv', index=False)
    
    item_map_df = pd.DataFrame(list(item_map.items()), columns=['old_id', 'new_id'])
    item_map_df.to_csv(PROCESSED_DIR / 'item_map.csv', index=False)
    
    # 映射ID并添加所需列
    df['userID'] = df['user'].map(lambda x: int(user_map[int(x)]))
    df['itemID'] = df['item'].map(lambda x: int(item_map[int(x)]))
    df['rating'] = 5  # 统一评分为5

    # 生成label
    # 为每个用户生成标签
    # 每个用户的最后10%交互设为1(验证集),倒数第二个10%设为2(测试集)
    for user in df['userID'].unique():
        user_mask = df['userID'] == user
        user_interactions = df[user_mask].sort_values('timestamp')  # 按时间戳排序
        total_interactions = len(user_interactions)
        
        # 计算验证集和测试集的大小,确保每个集合至少有1个样本
        n_valid = max(1, int(total_interactions * 0.1))  # 验证集大小
        n_test = max(1, int(total_interactions * 0.1))   # 测试集大小
        
        # 获取最后n_valid个交互的索引作为验证集
        valid_indices = user_interactions.index[-n_valid:]
        # 获取倒数第二个n_test个交互的索引作为测试集
        test_indices = user_interactions.index[-(n_valid + n_test):-n_valid]
        
        # 设置标签: 1表示验证集,2表示测试集
        df.loc[valid_indices, 'x_label'] = 1  # 验证集标签
        df.loc[test_indices, 'x_label'] = 2   # 测试集标签

    # 选择并重排所需列
    inter_df = df[['userID', 'itemID', 'rating', 'timestamp', 'x_label']]
    
    # 保存交互数据为microlens.inter
    # 按照userID和timestamp从小到大排序
    inter_df = inter_df.sort_values(['userID', 'timestamp'], ascending=[True, True])
    # 确保列的顺序正确
    inter_df = inter_df[['userID', 'itemID', 'rating', 'timestamp', 'x_label']]
    inter_df.to_csv(PROCESSED_DIR / 'microlens.inter', sep='\t', index=False)

    print("处理多模态特征...")
    # 加载特征文件
    try:
        with open(SOURCE_DIR / 'MicroLens-100k_image_features_CLIPRN50.json', 'r') as f:
            image_features_dict = json.load(f)
        with open(SOURCE_DIR / 'MicroLens-100k_title_en_text_features_BgeM3.json', 'r') as f:
            text_features_dict = json.load(f)
    except Exception as e:
        print(f"加载特征文件失败: {e}")
        raise

    # 创建新的特征矩阵
    n_items = len(unique_items)
    new_image_features = np.zeros((n_items, len(next(iter(image_features_dict.values())))), dtype=np.float32)
    new_text_features = np.zeros((n_items, len(next(iter(text_features_dict.values())))), dtype=np.float32)
    
    # 重新映射特征
    for old_id, new_id in item_map.items():
        old_id_str = str(old_id)
        if old_id_str in image_features_dict:
            new_image_features[new_id] = image_features_dict[old_id_str]
        if old_id_str in text_features_dict:
            new_text_features[new_id] = text_features_dict[old_id_str]

    # 保存处理后的特征
    np.save(PROCESSED_DIR / 'image_feat.npy', new_image_features)
    np.save(PROCESSED_DIR / 'text_feat.npy', new_text_features)
    
    print("数据处理完成!")
    print(f"用户数量: {len(unique_users)}")
    print(f"物品数量: {len(unique_items)}")
    print(f"交互数量: {len(df)}")

if __name__ == "__main__":
    process_data()