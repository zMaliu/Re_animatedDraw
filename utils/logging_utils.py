# -*- coding: utf-8 -*-
"""
日志工具

提供日志记录和管理功能
包括日志配置、格式化、文件管理等
"""

import logging
import logging.handlers
import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import threading
import queue
import traceback
from enum import Enum
import colorama
from colorama import Fore, Back, Style

# 初始化colorama
colorama.init(autoreset=True)


class LogLevel(Enum):
    """
    日志级别枚举
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColoredFormatter(logging.Formatter):
    """
    彩色日志格式化器
    
    为不同级别的日志添加颜色
    """
    
    # 颜色映射
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        """
        初始化彩色格式化器
        
        Args:
            fmt (str): 日志格式
            datefmt (str): 日期格式
            use_colors (bool): 是否使用颜色
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        """
        格式化日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            str: 格式化后的日志字符串
        """
        # 获取基础格式化结果
        log_message = super().format(record)
        
        # 如果启用颜色且在终端中
        if self.use_colors and sys.stderr.isatty():
            level_name = record.levelname
            if level_name in self.COLORS:
                color = self.COLORS[level_name]
                log_message = f"{color}{log_message}{Style.RESET_ALL}"
        
        return log_message


class JsonFormatter(logging.Formatter):
    """
    JSON格式化器
    
    将日志记录格式化为JSON格式
    """
    
    def __init__(self, fields=None):
        """
        初始化JSON格式化器
        
        Args:
            fields (List[str]): 要包含的字段列表
        """
        super().__init__()
        self.fields = fields or [
            'timestamp', 'level', 'logger', 'message', 
            'module', 'function', 'line'
        ]
    
    def format(self, record):
        """
        格式化日志记录为JSON
        
        Args:
            record: 日志记录
            
        Returns:
            str: JSON格式的日志字符串
        """
        log_data = {}
        
        # 添加基础字段
        if 'timestamp' in self.fields:
            log_data['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
        
        if 'level' in self.fields:
            log_data['level'] = record.levelname
        
        if 'logger' in self.fields:
            log_data['logger'] = record.name
        
        if 'message' in self.fields:
            log_data['message'] = record.getMessage()
        
        if 'module' in self.fields:
            log_data['module'] = record.module
        
        if 'function' in self.fields:
            log_data['function'] = record.funcName
        
        if 'line' in self.fields:
            log_data['line'] = record.lineno
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process',
                          'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


class PerformanceFilter(logging.Filter):
    """
    性能过滤器
    
    添加性能相关的信息到日志记录中
    """
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def filter(self, record):
        """
        过滤日志记录，添加性能信息
        
        Args:
            record: 日志记录
            
        Returns:
            bool: 是否通过过滤
        """
        # 添加运行时间
        record.runtime = time.time() - self.start_time
        
        # 添加内存使用信息（如果可用）
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_percent = process.cpu_percent()
        except ImportError:
            record.memory_mb = 0
            record.cpu_percent = 0
        
        return True


class AsyncHandler(logging.Handler):
    """
    异步日志处理器
    
    在单独的线程中处理日志记录，避免阻塞主线程
    """
    
    def __init__(self, handler, queue_size=1000):
        """
        初始化异步处理器
        
        Args:
            handler: 实际的日志处理器
            queue_size (int): 队列大小
        """
        super().__init__()
        self.handler = handler
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self._stop_event = threading.Event()
    
    def emit(self, record):
        """
        发送日志记录到队列
        
        Args:
            record: 日志记录
        """
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # 队列满时丢弃最旧的记录
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(record)
            except queue.Empty:
                pass
    
    def _worker(self):
        """
        工作线程，处理队列中的日志记录
        """
        while not self._stop_event.is_set():
            try:
                record = self.queue.get(timeout=1)
                self.handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # 避免日志处理器的异常影响主程序
                pass
    
    def close(self):
        """
        关闭异步处理器
        """
        self._stop_event.set()
        self.thread.join(timeout=5)
        self.handler.close()
        super().close()


class LogManager:
    """
    日志管理器
    
    提供日志配置和管理功能
    """
    
    def __init__(self, config=None):
        """
        初始化日志管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.loggers = {}
        self.handlers = {}
        self.default_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.default_date_format = '%Y-%m-%d %H:%M:%S'
    
    def create_logger(self, name: str, level: Union[str, int] = 'INFO', 
                     handlers: Optional[List[str]] = None) -> logging.Logger:
        """
        创建日志记录器
        
        Args:
            name (str): 日志记录器名称
            level (Union[str, int]): 日志级别
            handlers (Optional[List[str]]): 处理器名称列表
            
        Returns:
            logging.Logger: 日志记录器
        """
        try:
            # 如果已存在，直接返回
            if name in self.loggers:
                return self.loggers[name]
            
            # 创建日志记录器
            logger = logging.getLogger(name)
            
            # 设置日志级别
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            logger.setLevel(level)
            
            # 清除现有处理器
            logger.handlers.clear()
            
            # 添加处理器
            if handlers:
                for handler_name in handlers:
                    if handler_name in self.handlers:
                        logger.addHandler(self.handlers[handler_name])
            
            # 防止日志传播到根日志记录器
            logger.propagate = False
            
            self.loggers[name] = logger
            return logger
            
        except Exception as e:
            print(f"Error creating logger {name}: {str(e)}")
            return logging.getLogger(name)
    
    def create_console_handler(self, name: str = 'console', 
                             level: Union[str, int] = 'INFO',
                             use_colors: bool = True,
                             format_string: Optional[str] = None) -> bool:
        """
        创建控制台处理器
        
        Args:
            name (str): 处理器名称
            level (Union[str, int]): 日志级别
            use_colors (bool): 是否使用颜色
            format_string (Optional[str]): 格式字符串
            
        Returns:
            bool: 是否成功
        """
        try:
            handler = logging.StreamHandler(sys.stdout)
            
            # 设置级别
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            handler.setLevel(level)
            
            # 设置格式化器
            fmt = format_string or self.default_format
            if use_colors:
                formatter = ColoredFormatter(fmt, self.default_date_format)
            else:
                formatter = logging.Formatter(fmt, self.default_date_format)
            
            handler.setFormatter(formatter)
            
            self.handlers[name] = handler
            return True
            
        except Exception as e:
            print(f"Error creating console handler: {str(e)}")
            return False
    
    def create_file_handler(self, name: str, file_path: str,
                          level: Union[str, int] = 'INFO',
                          max_bytes: int = 10 * 1024 * 1024,
                          backup_count: int = 5,
                          encoding: str = 'utf-8',
                          format_string: Optional[str] = None,
                          use_json: bool = False) -> bool:
        """
        创建文件处理器
        
        Args:
            name (str): 处理器名称
            file_path (str): 文件路径
            level (Union[str, int]): 日志级别
            max_bytes (int): 最大文件大小
            backup_count (int): 备份文件数量
            encoding (str): 文件编码
            format_string (Optional[str]): 格式字符串
            use_json (bool): 是否使用JSON格式
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            log_dir = os.path.dirname(file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # 创建轮转文件处理器
            handler = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=max_bytes, backupCount=backup_count,
                encoding=encoding
            )
            
            # 设置级别
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            handler.setLevel(level)
            
            # 设置格式化器
            if use_json:
                formatter = JsonFormatter()
            else:
                fmt = format_string or self.default_format
                formatter = logging.Formatter(fmt, self.default_date_format)
            
            handler.setFormatter(formatter)
            
            self.handlers[name] = handler
            return True
            
        except Exception as e:
            print(f"Error creating file handler: {str(e)}")
            return False
    
    def create_timed_file_handler(self, name: str, file_path: str,
                                when: str = 'midnight',
                                interval: int = 1,
                                backup_count: int = 30,
                                level: Union[str, int] = 'INFO',
                                encoding: str = 'utf-8',
                                format_string: Optional[str] = None,
                                use_json: bool = False) -> bool:
        """
        创建定时文件处理器
        
        Args:
            name (str): 处理器名称
            file_path (str): 文件路径
            when (str): 轮转时间 ('S', 'M', 'H', 'D', 'midnight')
            interval (int): 轮转间隔
            backup_count (int): 备份文件数量
            level (Union[str, int]): 日志级别
            encoding (str): 文件编码
            format_string (Optional[str]): 格式字符串
            use_json (bool): 是否使用JSON格式
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            log_dir = os.path.dirname(file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # 创建定时轮转文件处理器
            handler = logging.handlers.TimedRotatingFileHandler(
                file_path, when=when, interval=interval,
                backupCount=backup_count, encoding=encoding
            )
            
            # 设置级别
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            handler.setLevel(level)
            
            # 设置格式化器
            if use_json:
                formatter = JsonFormatter()
            else:
                fmt = format_string or self.default_format
                formatter = logging.Formatter(fmt, self.default_date_format)
            
            handler.setFormatter(formatter)
            
            self.handlers[name] = handler
            return True
            
        except Exception as e:
            print(f"Error creating timed file handler: {str(e)}")
            return False
    
    def create_async_handler(self, name: str, base_handler_name: str,
                           queue_size: int = 1000) -> bool:
        """
        创建异步处理器
        
        Args:
            name (str): 处理器名称
            base_handler_name (str): 基础处理器名称
            queue_size (int): 队列大小
            
        Returns:
            bool: 是否成功
        """
        try:
            if base_handler_name not in self.handlers:
                print(f"Base handler {base_handler_name} not found")
                return False
            
            base_handler = self.handlers[base_handler_name]
            async_handler = AsyncHandler(base_handler, queue_size)
            
            self.handlers[name] = async_handler
            return True
            
        except Exception as e:
            print(f"Error creating async handler: {str(e)}")
            return False
    
    def setup_default_logging(self, log_dir: str = 'logs',
                            app_name: str = 'app',
                            console_level: str = 'INFO',
                            file_level: str = 'DEBUG',
                            use_colors: bool = True,
                            use_json: bool = False) -> logging.Logger:
        """
        设置默认日志配置
        
        Args:
            log_dir (str): 日志目录
            app_name (str): 应用名称
            console_level (str): 控制台日志级别
            file_level (str): 文件日志级别
            use_colors (bool): 是否使用颜色
            use_json (bool): 是否使用JSON格式
            
        Returns:
            logging.Logger: 主日志记录器
        """
        try:
            # 创建控制台处理器
            self.create_console_handler(
                'console', console_level, use_colors
            )
            
            # 创建文件处理器
            log_file = os.path.join(log_dir, f'{app_name}.log')
            self.create_file_handler(
                'file', log_file, file_level, use_json=use_json
            )
            
            # 创建错误文件处理器
            error_log_file = os.path.join(log_dir, f'{app_name}_error.log')
            self.create_file_handler(
                'error_file', error_log_file, 'ERROR', use_json=use_json
            )
            
            # 创建主日志记录器
            main_logger = self.create_logger(
                app_name, 'DEBUG', ['console', 'file', 'error_file']
            )
            
            # 添加性能过滤器
            perf_filter = PerformanceFilter()
            for handler in main_logger.handlers:
                handler.addFilter(perf_filter)
            
            return main_logger
            
        except Exception as e:
            print(f"Error setting up default logging: {str(e)}")
            return logging.getLogger(app_name)
    
    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """
        获取日志记录器
        
        Args:
            name (str): 日志记录器名称
            
        Returns:
            Optional[logging.Logger]: 日志记录器
        """
        return self.loggers.get(name)
    
    def close_all_handlers(self):
        """
        关闭所有处理器
        """
        for handler in self.handlers.values():
            handler.close()
        self.handlers.clear()
    
    def cleanup_old_logs(self, log_dir: str, days: int = 30) -> int:
        """
        清理旧日志文件
        
        Args:
            log_dir (str): 日志目录
            days (int): 保留天数
            
        Returns:
            int: 删除的文件数量
        """
        try:
            if not os.path.exists(log_dir):
                return 0
            
            cutoff_time = datetime.now() - timedelta(days=days)
            deleted_count = 0
            
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith('.log') or '.log.' in file:
                        file_path = os.path.join(root, file)
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning up old logs: {str(e)}")
            return 0


class ContextLogger:
    """
    上下文日志记录器
    
    提供带上下文信息的日志记录功能
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any] = None):
        """
        初始化上下文日志记录器
        
        Args:
            logger (logging.Logger): 基础日志记录器
            context (Dict[str, Any]): 上下文信息
        """
        self.logger = logger
        self.context = context or {}
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """
        带上下文信息的日志记录
        
        Args:
            level (int): 日志级别
            msg (str): 日志消息
            *args: 位置参数
            **kwargs: 关键字参数
        """
        # 合并上下文信息
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """记录DEBUG级别日志"""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """记录INFO级别日志"""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """记录WARNING级别日志"""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """记录ERROR级别日志"""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """记录CRITICAL级别日志"""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """记录异常日志"""
        kwargs['exc_info'] = True
        self.error(msg, *args, **kwargs)
    
    def add_context(self, **kwargs):
        """
        添加上下文信息
        
        Args:
            **kwargs: 上下文键值对
        """
        self.context.update(kwargs)
    
    def remove_context(self, *keys):
        """
        移除上下文信息
        
        Args:
            *keys: 要移除的键
        """
        for key in keys:
            self.context.pop(key, None)
    
    def clear_context(self):
        """
        清除所有上下文信息
        """
        self.context.clear()


class LogAnalyzer:
    """
    日志分析器
    
    提供日志文件分析功能
    """
    
    def __init__(self, config=None):
        """
        初始化日志分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def analyze_log_file(self, file_path: str) -> Dict[str, Any]:
        """
        分析日志文件
        
        Args:
            file_path (str): 日志文件路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            if not os.path.exists(file_path):
                return {'error': 'File not found'}
            
            stats = {
                'total_lines': 0,
                'level_counts': {
                    'DEBUG': 0,
                    'INFO': 0,
                    'WARNING': 0,
                    'ERROR': 0,
                    'CRITICAL': 0
                },
                'time_range': {'start': None, 'end': None},
                'error_messages': [],
                'warning_messages': [],
                'file_size': os.path.getsize(file_path)
            }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stats['total_lines'] += 1
                    
                    # 分析日志级别
                    for level in stats['level_counts']:
                        if level in line:
                            stats['level_counts'][level] += 1
                            break
                    
                    # 收集错误和警告消息
                    if 'ERROR' in line:
                        stats['error_messages'].append(line.strip())
                    elif 'WARNING' in line:
                        stats['warning_messages'].append(line.strip())
                    
                    # 提取时间信息（简单实现）
                    # 这里可以根据实际日志格式进行调整
                    if stats['time_range']['start'] is None:
                        stats['time_range']['start'] = line[:19]  # 假设时间在开头
                    stats['time_range']['end'] = line[:19]
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def find_patterns(self, file_path: str, patterns: List[str]) -> Dict[str, List[str]]:
        """
        在日志文件中查找模式
        
        Args:
            file_path (str): 日志文件路径
            patterns (List[str]): 要查找的模式列表
            
        Returns:
            Dict[str, List[str]]: 匹配结果
        """
        try:
            import re
            
            results = {pattern: [] for pattern in patterns}
            
            if not os.path.exists(file_path):
                return results
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            results[pattern].append(f"Line {line_num}: {line.strip()}")
            
            return results
            
        except Exception as e:
            return {pattern: [f"Error: {str(e)}"] for pattern in patterns}
    
    def generate_report(self, log_dir: str, output_file: str = None) -> str:
        """
        生成日志分析报告
        
        Args:
            log_dir (str): 日志目录
            output_file (str): 输出文件路径
            
        Returns:
            str: 报告内容
        """
        try:
            report_lines = []
            report_lines.append("=" * 50)
            report_lines.append("日志分析报告")
            report_lines.append("=" * 50)
            report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"日志目录: {log_dir}")
            report_lines.append("")
            
            # 分析所有日志文件
            log_files = []
            if os.path.exists(log_dir):
                for file in os.listdir(log_dir):
                    if file.endswith('.log'):
                        log_files.append(os.path.join(log_dir, file))
            
            total_stats = {
                'total_files': len(log_files),
                'total_lines': 0,
                'total_size': 0,
                'level_counts': {
                    'DEBUG': 0,
                    'INFO': 0,
                    'WARNING': 0,
                    'ERROR': 0,
                    'CRITICAL': 0
                }
            }
            
            for log_file in log_files:
                stats = self.analyze_log_file(log_file)
                if 'error' not in stats:
                    total_stats['total_lines'] += stats['total_lines']
                    total_stats['total_size'] += stats['file_size']
                    
                    for level, count in stats['level_counts'].items():
                        total_stats['level_counts'][level] += count
                    
                    report_lines.append(f"文件: {os.path.basename(log_file)}")
                    report_lines.append(f"  行数: {stats['total_lines']}")
                    report_lines.append(f"  大小: {stats['file_size']} bytes")
                    report_lines.append(f"  错误数: {stats['level_counts']['ERROR']}")
                    report_lines.append(f"  警告数: {stats['level_counts']['WARNING']}")
                    report_lines.append("")
            
            # 总体统计
            report_lines.append("总体统计:")
            report_lines.append(f"  文件总数: {total_stats['total_files']}")
            report_lines.append(f"  行数总计: {total_stats['total_lines']}")
            report_lines.append(f"  大小总计: {total_stats['total_size']} bytes")
            report_lines.append("")
            
            report_lines.append("日志级别分布:")
            for level, count in total_stats['level_counts'].items():
                percentage = (count / total_stats['total_lines'] * 100) if total_stats['total_lines'] > 0 else 0
                report_lines.append(f"  {level}: {count} ({percentage:.1f}%)")
            
            report_content = "\n".join(report_lines)
            
            # 保存报告
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
            
            return report_content
            
        except Exception as e:
            return f"生成报告时出错: {str(e)}"


# 便捷函数
def setup_logging(log_dir: str = 'logs', app_name: str = 'app',
                 console_level: str = 'INFO', file_level: str = 'DEBUG',
                 use_colors: bool = True, use_json: bool = False) -> logging.Logger:
    """
    快速设置日志配置
    
    Args:
        log_dir (str): 日志目录
        app_name (str): 应用名称
        console_level (str): 控制台日志级别
        file_level (str): 文件日志级别
        use_colors (bool): 是否使用颜色
        use_json (bool): 是否使用JSON格式
        
    Returns:
        logging.Logger: 主日志记录器
    """
    log_manager = LogManager()
    return log_manager.setup_default_logging(
        log_dir, app_name, console_level, file_level, use_colors, use_json
    )


def get_context_logger(logger: logging.Logger, **context) -> ContextLogger:
    """
    获取上下文日志记录器
    
    Args:
        logger (logging.Logger): 基础日志记录器
        **context: 上下文信息
        
    Returns:
        ContextLogger: 上下文日志记录器
    """
    return ContextLogger(logger, context)