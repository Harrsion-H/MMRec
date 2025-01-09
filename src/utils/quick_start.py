# coding: utf-8
# @email: enoche.chow@gmail.com

"""
快速启动训练和评估流程的主函数
##########################
"""
# 导入所需的库
from logging import getLogger  # 日志记录
from itertools import product  # 用于生成超参数组合
from .dataset import RecDataset  # 推荐系统数据集类
from .dataloader import TrainDataLoader, EvalDataLoader  # 训练和评估数据加载器
from .logger import init_logger  # 日志初始化
from .configurator import Config  # 配置管理
from .utils import init_seed, get_model, get_trainer, dict2str  # 工具函数
import platform  # 获取系统信息
import os  # 操作系统接口


def quick_start(model, dataset, config_dict, save_model=False, mg=False):
    """
    快速启动训练和评估流程的主函数
    Args:
        model: 模型名称
        dataset: 数据集名称 
        config_dict: 配置字典
        save_model: 是否保存模型
        mg: 是否使用多图
    """
    # 合并配置字典,初始化配置对象
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)  # 初始化日志
    logger = getLogger()
    
    # 打印系统和目录信息
    logger.info('██Server: \t' + platform.node())  # 服务器名称
    logger.info('██Dir: \t' + os.getcwd() + '\n')  # 当前工作目录
    logger.info(config)  # 打印配置信息

    # 加载数据集
    dataset = RecDataset(config)
    logger.info(str(dataset))  # 打印数据集统计信息

    # 划分训练、验证、测试集
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))  # 打印训练集信息
    logger.info('\n====Validation====\n' + str(valid_dataset))  # 打印验证集信息
    logger.info('\n====Testing====\n' + str(test_dataset))  # 打印测试集信息

    # 包装为数据加载器
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ 数据集加载完成,开始运行模型
    hyper_ret = []  # 存储所有超参数组合的结果
    val_metric = config['valid_metric'].lower()  # 验证指标名称
    best_test_value = 0.0  # 最佳测试结果
    idx = best_test_idx = 0  # 当前和最佳结果的索引

    logger.info('\n\n=================================\n\n')

    # 处理超参数
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']  # 确保seed在超参数列表中
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])  # 获取每个超参数的取值列表
        
    # 生成所有超参数组合
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    
    # 遍历每种超参数组合
    for hyper_tuple in combinators:
        # 重置随机种子
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # 设置数据加载器的随机状态
        train_data.pretrain_setup()
        # 加载并初始化模型
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # 加载并初始化训练器
        trainer = get_trainer()(config, model, mg)
        # 训练模型
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=config['save_model'])
        
        # 保存结果
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # 更新最佳测试结果
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        # 打印当前结果
        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # 打印所有实验结果
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    # 打印最佳结果
    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))
