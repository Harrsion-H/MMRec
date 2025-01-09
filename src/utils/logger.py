# coding: utf-8
# @email: enoche.chow@gmail.com

"""
###############################
"""

import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    """初始化日志记录器
    
    Args:
        config (Config): 配置对象
    """
    # 使用配置文件中的路径
    log_dir = config['log']['save_dir'] if 'log' in config else './log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 生成日志文件名
    timestamp = get_local_time()
    model_name = config['model']
    dataset_name = config['dataset'] 
    logfilename = f'{timestamp}-{model_name}-{dataset_name}.log'
    logfilepath = os.path.join(log_dir, logfilename)

    # 使用配置文件中的格式
    file_formatter = logging.Formatter(
        fmt=config['log']['file_format'] if 'log' in config else "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt=config['log']['date_format'] if 'log' in config else "%Y-%m-%d %H:%M:%S"
    )
    
    console_formatter = logging.Formatter(
        fmt=config['log']['console_format'] if 'log' in config else "%(asctime)s [%(levelname)s] %(message)s",
        datefmt=config['log']['date_format'] if 'log' in config else "%Y-%m-%d %H:%M:%S"
    )

    # 文件处理器
    fh = logging.FileHandler(logfilepath, encoding='utf-8')
    fh.setFormatter(file_formatter)
    
    # 控制台处理器
    sh = logging.StreamHandler()
    sh.setFormatter(console_formatter)

    # 使用配置文件中的日志级别
    try:
        level = getattr(logging, config['log']['state'].upper()) if 'log' in config else logging.INFO
    except (AttributeError, KeyError):
        level = logging.INFO
    
    # 配置根日志记录器
    logging.basicConfig(
        level=level,
        handlers=[sh, fh]
    )


