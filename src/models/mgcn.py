# coding: utf-8
# @email: y463213402@gmail.com
r"""
MGCN (Multi-View Graph Convolutional Network)
################################################
Reference:
    https://github.com/demonph10/MGCN
    ACM MM'2023: [Multi-View Graph Convolutional Network for Multimedia Recommendation]

主要思想:
1. 利用多视图图卷积网络进行多媒体推荐
2. 包含用户-物品交互视图和物品-物品相似度视图
3. 使用行为引导的纯化器和行为感知的融合器来整合多模态信息
4. 采用对比学习来增强模型表示能力
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class MGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        """
        MGCN模型的初始化函数
        
        Args:
            config (Config): 全局配置对象,包含各种模型参数
            dataset (Dataset): 数据集对象,包含训练所需的所有数据
        """
        super(MGCN, self).__init__(config, dataset)
        self.sparse = True
        #cl_loss 指的是contrastive loss，对比损失
        self.cl_loss = config['cl_loss']
        #n_ui_layers指的是用户-物品交互矩阵的层数
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']

        # load dataset info
        # 构建用户-物品交互矩阵(COO格式)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        #将用户-物品交互矩阵转换为用户-用户矩阵
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        # 使用Xavier初始化嵌入层权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # 构建图像和文本相似度矩阵的文件路径
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        # 获取归一化的邻接矩阵
        self.norm_adj = self.get_adj_mat()
        # 将稀疏矩阵转换为PyTorch稀疏张量
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # 处理图像特征
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                # 构建图像相似度矩阵
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        # 处理文本特征
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                # 构建文本相似度矩阵
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        # 特征转换层
        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # softmax层用于注意力权重计算
        self.softmax = nn.Softmax(dim=-1)

        # 共同特征查询网络
        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        # 图像门控单元
        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        # 文本门控单元
        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        # 图像偏好门控单元
        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        # 文本偏好门控单元
        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        # 温度参数
        self.tau = 0.5

    def pre_epoch_processing(self):
        """
        每个epoch前的预处理
        """
        pass

    def get_adj_mat(self):
        """
        构建归一化的邻接矩阵
        
        Returns:
            scipy.sparse.csr_matrix: 归一化后的邻接矩阵
        """
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            """
            对邻接矩阵进行归一化
            
            Args:
                adj: 原始邻接矩阵
                
            Returns:
                归一化后的邻接矩阵
            """
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """
        将scipy稀疏矩阵转换为torch稀疏张量
        
        Args:
            sparse_mx: scipy稀疏矩阵
            
        Returns:
            torch.sparse.FloatTensor: 转换后的torch稀疏张量
        """
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        """
        模型的前向传播函数
        
        Args:
            adj: 归一化的邻接矩阵,表示用户-物品交互关系
            train: 是否为训练模式,默认为False
            
        Returns:
            train=True时:
                - all_embeddings_users: 最终的用户嵌入
                - all_embeddings_items: 最终的物品嵌入  
                - side_embeds: 侧面信息嵌入(图像和文本)
                - content_embeds: 内容信息嵌入(ID)
            train=False时:
                - all_embeddings_users: 最终的用户嵌入
                - all_embeddings_items: 最终的物品嵌入
        """
        # 1. 多模态特征转换
        if self.v_feat is not None:
            # 转换图像特征
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            # 转换文本特征
            text_feats = self.text_trs(self.text_embedding.weight)

        # 2. 行为引导的特征纯化
        # 使用门控机制过滤图像和文本特征
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats))  # 图像特征纯化
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))    # 文本特征纯化

        # 3. 用户-物品交互建模
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        # 拼接用户和物品嵌入
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        # 多层图卷积传播
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)  # 消息传递
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        # 聚合多层结果
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # 4. 物品-物品关系建模
        # 4.1 图像视图传播
        if self.sparse:
            # 稀疏矩阵乘法
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            # 密集矩阵乘法
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        # 获取用户在图像视图下的表示
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        # 4.2 文本视图传播
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        # 获取用户在文本视图下的表示
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # 5. 行为感知的多模态特征融合
        # 5.1 计算共同特征
        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)  # 注意力权重
        # 加权融合得到共同特征
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(dim=1) * text_embeds
        
        # 5.2 提取特定特征
        sep_image_embeds = image_embeds - common_embeds  # 图像特定特征
        sep_text_embeds = text_embeds - common_embeds    # 文本特定特征

        # 5.3 基于用户行为偏好的特征调整
        image_prefer = self.gate_image_prefer(content_embeds)  # 图像偏好门控
        text_prefer = self.gate_text_prefer(content_embeds)    # 文本偏好门控
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        # 融合所有特征
        side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3

        # 6. 最终特征融合
        all_embeds = content_embeds + side_embeds
        # 分离用户和物品嵌入
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        """
        计算BPR损失
        
        Args:
            users: 用户嵌入
            pos_items: 正样本物品嵌入
            neg_items: 负样本物品嵌入
            
        Returns:
            mf_loss: 矩阵分解损失
            emb_loss: 嵌入正则化损失
            reg_loss: 额外的正则化损失
        """
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        """
        计算InfoNCE对比损失
        
        Args:
            view1: 第一个视图的嵌入
            view2: 第二个视图的嵌入
            temperature: 温度参数
            
        Returns:
            对比损失值
        """
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        """
        计算总损失
        
        Args:
            interaction: 包含用户、正样本物品和负样本物品的交互信息
            
        Returns:
            总损失值
        """
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

    def full_sort_predict(self, interaction):
        """
        全排序预测
        
        Args:
            interaction: 包含用户ID的交互信息
            
        Returns:
            用户对所有物品的预测分数
        """
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores