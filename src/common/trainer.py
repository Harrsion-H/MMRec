# coding: utf-8
# @email: enoche.chow@gmail.com

"""
训练器模块,包含了训练和评估推荐系统模型的基本功能
主要包含:
- AbstractTrainer: 抽象基类,定义了训练器的基本接口
- Trainer: 具体实现类,包含了常用的训练和评估功能
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str  
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    """训练器的抽象基类
    
    定义了训练器需要实现的基本接口:
    - fit(): 训练模型
    - evaluate(): 评估模型
    
    Args:
        config (dict): 配置参数字典
        model (nn.Module): 待训练的模型
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        """训练模型的抽象方法
        
        Args:
            train_data: 训练数据
        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        """评估模型的抽象方法
        
        Args:
            eval_data: 评估数据
        """
        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    """推荐系统模型的基础训练器实现
    
    实现了基本的训练和评估流程,包括:
    - 模型训练(fit)
    - 模型评估(evaluate) 
    - 训练过程中的早停
    - 学习率调度
    - 梯度裁剪
    - 损失记录与可视化等功能
    
    Args:
        config (dict): 配置参数字典,包含学习率、训练轮数等超参数
        model (nn.Module): 待训练的模型
        mg (bool): 是否使用特殊的训练策略
    """

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        # 初始化日志记录器
        self.logger = getLogger()
        
        # 从配置中加载训练相关参数
        self.learner = config['learner']  # 优化器类型
        self.learning_rate = config['learning_rate']  # 学习率
        self.epochs = config['epochs']  # 训练轮数
        self.eval_step = min(config['eval_step'], self.epochs)  # 评估间隔
        self.stopping_step = config['stopping_step']  # 早停步数
        self.clip_grad_norm = config['clip_grad_norm']  # 梯度裁剪参数
        self.valid_metric = config['valid_metric'].lower()  # 验证指标
        self.valid_metric_bigger = config['valid_metric_bigger']  # 验证指标是否越大越好
        self.test_batch_size = config['eval_batch_size']  # 测试batch大小
        self.device = config['device']  # 训练设备
        
        # 权重衰减设置
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']  # 是否需要训练

        # 训练状态相关变量初始化
        self.start_epoch = 0
        self.cur_step = 0

        # 初始化最佳验证结果字典
        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        
        # 构建优化器
        self.optimizer = self._build_optimizer()

        # 设置学习率调度器
        lr_scheduler = config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        # 评估相关设置
        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        # 其他参数设置
        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg  # 是否使用特殊训练策略
        self.alpha1 = config['alpha1']  # 特殊训练策略参数
        self.alpha2 = config['alpha2']  # 特殊训练策略参数
        self.beta = config['beta']  # 特殊训练策略参数

    def _build_optimizer(self):
        """构建优化器
        
        根据配置选择不同的优化器(Adam/SGD/Adagrad/RMSprop)
        
        Returns:
            torch.optim: 优化器实例
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        """训练一个epoch
        
        Args:
            train_data (DataLoader): 训练数据
            epoch_idx (int): 当前epoch索引
            loss_func (function, optional): 损失函数,默认使用模型自带的损失计算
            
        Returns:
            float/tuple: 该epoch的总损失。如果损失包含多个部分,返回损失的元组
        """
        if not self.req_training:
            return 0.0, []
            
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        
        # 逐batch训练
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_inter = interaction.clone()
            losses = loss_func(interaction)
            
            # 处理损失(单个损失或多个损失的情况)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                
            # 检查损失是否为nan
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            
            # 特殊训练策略(如果启用)
            if self.mg and batch_idx % self.beta == 0:
                # 第一次前向传播和反向传播
                first_loss = self.alpha1 * loss
                first_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 第二次前向传播和反向传播
                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses
                    
                if self._check_nan(loss):
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()
                
            # 梯度裁剪(如果启用)
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                
            # 优化器步进
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, is_test=False):
        """验证一个epoch
        
        Args:
            valid_data (DataLoader): 验证数据
            
        Returns:
            float: 验证分数
            dict: 验证结果
        """
        valid_result = self.evaluate(valid_data, is_test=is_test)   
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        """检查损失值是否为nan
        
        Args:
            loss (torch.Tensor): 损失值
            
        Returns:
            bool: 是否为nan
        """
        if torch.isnan(loss):
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        """生成训练损失输出字符串
        
        Args:
            epoch_idx (int): epoch索引
            s_time (float): 开始时间
            e_time (float): 结束时间
            losses (float/tuple): 损失值
            
        Returns:
            str: 格式化的输出字符串
        """
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        """训练模型的主函数
        
        Args:
            train_data (DataLoader): 训练数据
            valid_data (DataLoader, optional): 验证数据
            test_data (DataLoader, optional): 测试数据
            saved (bool, optional): 是否保存模型
            verbose (bool, optional): 是否打印训练信息
            
        Returns:
            tuple: (最佳验证分数, 最佳验证结果, 最佳测试结果)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # 训练阶段
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            
            # 检查是否出现nan损失
            if torch.is_tensor(train_loss):
                break
                
            # 学习率调度
            self.lr_scheduler.step()

            # 记录训练损失
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            
            # 生成训练输出信息
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # 评估阶段
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                
                # 早停检查
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                    
                valid_end_time = time()
                
                # 生成验证输出信息
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                
                # 测试
                _, test_result = self._valid_epoch(test_data,is_test=True)
                
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                    
                # 更新最佳结果
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                # 早停检查
                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
                    
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        """评估模型
        
        Args:
            eval_data (DataLoader): 评估数据
            is_test (bool, optional): 是否为测试模式
            idx (int, optional): 评估索引
            
        Returns:
            dict: 评估结果,包含各项评估指标
        """
        self.model.eval()

        # 对所有用户进行批量评估
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # 预测得分
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # 屏蔽正样本项
            scores[masked_items[0], masked_items[1]] = -1e10
            # 排序获取topk
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        """绘制训练损失曲线
        
        Args:
            show (bool, optional): 是否显示图像
            save_path (str, optional): 保存图像的路径
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    def _save_checkpoint(self, epoch):
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        model_name = self.config['model']
        checkpoint_dir = self.config['checkpoint_dir']  # 使用配置的checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, model_name + '-' + str(epoch) + '.pth')
        torch.save(state, checkpoint_file)
        self.logger.info(f"Model saved to {checkpoint_file}")
