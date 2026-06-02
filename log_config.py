import logging
import os
from datetime import datetime


def init_logger(
    logger_name: str = "max_agent",
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    log_file_prefix: str = "app",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    初始化日志配置
    
    Args:
        logger_name: 日志器名称
        log_level: 日志级别，默认INFO
        log_dir: 日志文件目录，默认logs
        log_file_prefix: 日志文件前缀，默认app
        console_output: 是否输出到控制台，默认True
        file_output: 是否输出到文件，默认True
    
    Returns:
        配置好的Logger实例
    """
    # 创建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if file_output:
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 生成日志文件名（按日期）
        current_date = datetime.now().strftime('%Y%m%d')
        log_filename = f"{log_file_prefix}_{current_date}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        
        # 文件handler
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
