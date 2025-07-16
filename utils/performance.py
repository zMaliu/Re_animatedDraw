# -*- coding: utf-8 -*-
"""
性能监控工具

提供性能分析和监控功能
包括时间测量、内存监控、性能分析等
"""

import time
import functools
import threading
import psutil
import gc
import sys
import tracemalloc
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager
from collections import defaultdict, deque
import json
import os
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """
    性能指标数据类
    """
    name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    memory_before: float = 0.0
    memory_after: float = 0.0
    memory_peak: float = 0.0
    cpu_percent: float = 0.0
    call_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta(self) -> float:
        """内存变化量"""
        return self.memory_after - self.memory_before
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'duration': self.duration,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_delta': self.memory_delta,
            'memory_peak': self.memory_peak,
            'cpu_percent': self.cpu_percent,
            'call_count': self.call_count,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'metadata': self.metadata
        }


class Timer:
    """
    计时器类
    
    提供高精度时间测量功能
    """
    
    def __init__(self, name: str = "Timer"):
        """
        初始化计时器
        
        Args:
            name (str): 计时器名称
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0
        self.is_running = False
    
    def start(self):
        """
        开始计时
        """
        if self.is_running:
            raise RuntimeError(f"Timer '{self.name}' is already running")
        
        self.start_time = time.perf_counter()
        self.is_running = True
    
    def stop(self) -> float:
        """
        停止计时
        
        Returns:
            float: 经过的时间（秒）
        """
        if not self.is_running:
            raise RuntimeError(f"Timer '{self.name}' is not running")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        self.is_running = False
        
        return self.elapsed_time
    
    def reset(self):
        """
        重置计时器
        """
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0
        self.is_running = False
    
    def __enter__(self):
        """
        上下文管理器入口
        """
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口
        """
        self.stop()


class MemoryMonitor:
    """
    内存监控器
    
    监控内存使用情况
    """
    
    def __init__(self):
        """
        初始化内存监控器
        """
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()
        self.peak_memory = self.baseline_memory
        self.memory_history = deque(maxlen=1000)
        self._monitoring = False
        self._monitor_thread = None
    
    def get_memory_usage(self) -> float:
        """
        获取当前内存使用量（MB）
        
        Returns:
            float: 内存使用量（MB）
        """
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except Exception:
            return 0.0
    
    def get_memory_percent(self) -> float:
        """
        获取内存使用百分比
        
        Returns:
            float: 内存使用百分比
        """
        try:
            return self.process.memory_percent()
        except Exception:
            return 0.0
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始监控内存使用
        
        Args:
            interval (float): 监控间隔（秒）
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """
        停止监控内存使用
        """
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """
        监控循环
        
        Args:
            interval (float): 监控间隔
        """
        while self._monitoring:
            try:
                current_memory = self.get_memory_usage()
                self.memory_history.append({
                    'timestamp': time.time(),
                    'memory': current_memory,
                    'percent': self.get_memory_percent()
                })
                
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                time.sleep(interval)
            except Exception:
                break
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取内存统计信息
        
        Returns:
            Dict[str, float]: 内存统计信息
        """
        current_memory = self.get_memory_usage()
        
        stats = {
            'current': current_memory,
            'baseline': self.baseline_memory,
            'peak': self.peak_memory,
            'delta': current_memory - self.baseline_memory,
            'percent': self.get_memory_percent()
        }
        
        if self.memory_history:
            memories = [entry['memory'] for entry in self.memory_history]
            stats.update({
                'average': sum(memories) / len(memories),
                'min': min(memories),
                'max': max(memories)
            })
        
        return stats
    
    def reset_baseline(self):
        """
        重置基线内存
        """
        self.baseline_memory = self.get_memory_usage()
        self.peak_memory = self.baseline_memory
        self.memory_history.clear()


class CPUMonitor:
    """
    CPU监控器
    
    监控CPU使用情况
    """
    
    def __init__(self):
        """
        初始化CPU监控器
        """
        self.process = psutil.Process()
        self.cpu_history = deque(maxlen=1000)
        self._monitoring = False
        self._monitor_thread = None
    
    def get_cpu_percent(self, interval: float = None) -> float:
        """
        获取CPU使用百分比
        
        Args:
            interval (float): 测量间隔
            
        Returns:
            float: CPU使用百分比
        """
        try:
            return self.process.cpu_percent(interval=interval)
        except Exception:
            return 0.0
    
    def get_system_cpu_percent(self) -> float:
        """
        获取系统CPU使用百分比
        
        Returns:
            float: 系统CPU使用百分比
        """
        try:
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始监控CPU使用
        
        Args:
            interval (float): 监控间隔（秒）
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """
        停止监控CPU使用
        """
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """
        监控循环
        
        Args:
            interval (float): 监控间隔
        """
        while self._monitoring:
            try:
                cpu_percent = self.get_cpu_percent()
                system_cpu = self.get_system_cpu_percent()
                
                self.cpu_history.append({
                    'timestamp': time.time(),
                    'process_cpu': cpu_percent,
                    'system_cpu': system_cpu
                })
                
                time.sleep(interval)
            except Exception:
                break
    
    def get_cpu_stats(self) -> Dict[str, float]:
        """
        获取CPU统计信息
        
        Returns:
            Dict[str, float]: CPU统计信息
        """
        current_cpu = self.get_cpu_percent()
        system_cpu = self.get_system_cpu_percent()
        
        stats = {
            'current_process': current_cpu,
            'current_system': system_cpu
        }
        
        if self.cpu_history:
            process_cpus = [entry['process_cpu'] for entry in self.cpu_history]
            system_cpus = [entry['system_cpu'] for entry in self.cpu_history]
            
            stats.update({
                'average_process': sum(process_cpus) / len(process_cpus),
                'max_process': max(process_cpus),
                'average_system': sum(system_cpus) / len(system_cpus),
                'max_system': max(system_cpus)
            })
        
        return stats


class PerformanceProfiler:
    """
    性能分析器
    
    提供函数和代码块的性能分析功能
    """
    
    def __init__(self, name: str = "Profiler"):
        """
        初始化性能分析器
        
        Args:
            name (str): 分析器名称
        """
        self.name = name
        self.metrics = {}
        self.memory_monitor = MemoryMonitor()
        self.cpu_monitor = CPUMonitor()
        self.logger = logging.getLogger(__name__)
        self._call_stack = []
    
    def profile_function(self, func: Callable = None, *, name: str = None):
        """
        函数性能分析装饰器
        
        Args:
            func (Callable): 被装饰的函数
            name (str): 自定义名称
            
        Returns:
            装饰后的函数
        """
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                profile_name = name or f"{f.__module__}.{f.__name__}"
                
                with self.profile_context(profile_name):
                    return f(*args, **kwargs)
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    @contextmanager
    def profile_context(self, name: str, **metadata):
        """
        性能分析上下文管理器
        
        Args:
            name (str): 分析名称
            **metadata: 额外的元数据
        """
        # 开始分析
        start_time = time.perf_counter()
        memory_before = self.memory_monitor.get_memory_usage()
        
        # 记录调用栈
        self._call_stack.append(name)
        
        try:
            yield
        finally:
            # 结束分析
            end_time = time.perf_counter()
            memory_after = self.memory_monitor.get_memory_usage()
            cpu_percent = self.cpu_monitor.get_cpu_percent()
            
            # 创建性能指标
            metrics = PerformanceMetrics(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=self.memory_monitor.peak_memory,
                cpu_percent=cpu_percent,
                metadata=metadata
            )
            
            # 存储指标
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(metrics)
            
            # 移除调用栈
            if self._call_stack:
                self._call_stack.pop()
            
            # 记录日志
            self.logger.debug(
                f"Performance: {name} took {metrics.duration:.4f}s, "
                f"memory delta: {metrics.memory_delta:.2f}MB"
            )
    
    def get_metrics(self, name: str = None) -> Union[List[PerformanceMetrics], Dict[str, List[PerformanceMetrics]]]:
        """
        获取性能指标
        
        Args:
            name (str): 指标名称，如果为None则返回所有指标
            
        Returns:
            性能指标列表或字典
        """
        if name:
            return self.metrics.get(name, [])
        else:
            return self.metrics.copy()
    
    def get_summary(self, name: str = None) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Args:
            name (str): 指标名称，如果为None则返回所有指标的摘要
            
        Returns:
            Dict[str, Any]: 性能摘要
        """
        if name:
            metrics_list = self.metrics.get(name, [])
            if not metrics_list:
                return {}
            
            durations = [m.duration for m in metrics_list]
            memory_deltas = [m.memory_delta for m in metrics_list]
            
            return {
                'name': name,
                'call_count': len(metrics_list),
                'total_duration': sum(durations),
                'average_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_memory_delta': sum(memory_deltas),
                'average_memory_delta': sum(memory_deltas) / len(memory_deltas),
                'min_memory_delta': min(memory_deltas),
                'max_memory_delta': max(memory_deltas)
            }
        else:
            summary = {}
            for metric_name in self.metrics:
                summary[metric_name] = self.get_summary(metric_name)
            return summary
    
    def clear_metrics(self, name: str = None):
        """
        清除性能指标
        
        Args:
            name (str): 指标名称，如果为None则清除所有指标
        """
        if name:
            self.metrics.pop(name, None)
        else:
            self.metrics.clear()
    
    def export_metrics(self, file_path: str, format: str = 'json'):
        """
        导出性能指标
        
        Args:
            file_path (str): 文件路径
            format (str): 导出格式 ('json', 'csv')
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format.lower() == 'json':
                data = {}
                for name, metrics_list in self.metrics.items():
                    data[name] = [m.to_dict() for m in metrics_list]
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                import csv
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # 写入标题行
                    writer.writerow([
                        'name', 'duration', 'memory_before', 'memory_after',
                        'memory_delta', 'memory_peak', 'cpu_percent', 'call_count',
                        'start_time', 'end_time'
                    ])
                    
                    # 写入数据行
                    for name, metrics_list in self.metrics.items():
                        for m in metrics_list:
                            writer.writerow([
                                m.name, m.duration, m.memory_before, m.memory_after,
                                m.memory_delta, m.memory_peak, m.cpu_percent, m.call_count,
                                m.start_time, m.end_time
                            ])
            
            self.logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {str(e)}")
    
    def print_summary(self, name: str = None, top_n: int = 10):
        """
        打印性能摘要
        
        Args:
            name (str): 指标名称
            top_n (int): 显示前N个最耗时的函数
        """
        if name:
            summary = self.get_summary(name)
            if summary:
                print(f"\n=== Performance Summary: {name} ===")
                print(f"Call Count: {summary['call_count']}")
                print(f"Total Duration: {summary['total_duration']:.4f}s")
                print(f"Average Duration: {summary['average_duration']:.4f}s")
                print(f"Min Duration: {summary['min_duration']:.4f}s")
                print(f"Max Duration: {summary['max_duration']:.4f}s")
                print(f"Average Memory Delta: {summary['average_memory_delta']:.2f}MB")
        else:
            all_summaries = self.get_summary()
            if not all_summaries:
                print("No performance data available")
                return
            
            # 按总耗时排序
            sorted_summaries = sorted(
                all_summaries.items(),
                key=lambda x: x[1].get('total_duration', 0),
                reverse=True
            )
            
            print(f"\n=== Top {top_n} Performance Summary ===")
            print(f"{'Function':<40} {'Calls':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Memory(MB)':<12}")
            print("-" * 90)
            
            for i, (func_name, summary) in enumerate(sorted_summaries[:top_n]):
                print(f"{func_name:<40} {summary['call_count']:<8} "
                      f"{summary['total_duration']:<10.4f} {summary['average_duration']:<10.4f} "
                      f"{summary['average_memory_delta']:<12.2f}")


class SystemMonitor:
    """
    系统监控器
    
    监控整个系统的性能指标
    """
    
    def __init__(self, interval: float = 1.0):
        """
        初始化系统监控器
        
        Args:
            interval (float): 监控间隔（秒）
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=3600)  # 保存1小时的数据
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """
        开始系统监控
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """
        停止系统监控
        """
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """
        监控循环
        """
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                break
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """
        收集系统指标
        
        Returns:
            Dict[str, Any]: 系统指标
        """
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # 内存指标
            memory = psutil.virtual_memory()
            
            # 磁盘指标
            disk = psutil.disk_usage('/')
            
            # 网络指标
            network = psutil.net_io_counters()
            
            # 进程指标
            process_count = len(psutil.pids())
            
            return {
                'timestamp': time.time(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'processes': process_count
            }
        
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            return {'timestamp': time.time(), 'error': str(e)}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        获取当前系统指标
        
        Returns:
            Dict[str, Any]: 当前系统指标
        """
        return self._collect_system_metrics()
    
    def get_metrics_history(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        获取历史指标
        
        Args:
            duration_minutes (int): 历史时长（分钟）
            
        Returns:
            List[Dict[str, Any]]: 历史指标列表
        """
        cutoff_time = time.time() - (duration_minutes * 60)
        return [
            metrics for metrics in self.metrics_history
            if metrics.get('timestamp', 0) >= cutoff_time
        ]
    
    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        获取指标摘要
        
        Args:
            duration_minutes (int): 统计时长（分钟）
            
        Returns:
            Dict[str, Any]: 指标摘要
        """
        history = self.get_metrics_history(duration_minutes)
        if not history:
            return {}
        
        # 提取各项指标
        cpu_percents = [m['cpu']['percent'] for m in history if 'cpu' in m]
        memory_percents = [m['memory']['percent'] for m in history if 'memory' in m]
        disk_percents = [m['disk']['percent'] for m in history if 'disk' in m]
        
        summary = {
            'duration_minutes': duration_minutes,
            'sample_count': len(history),
            'cpu': {
                'average': sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0,
                'max': max(cpu_percents) if cpu_percents else 0,
                'min': min(cpu_percents) if cpu_percents else 0
            },
            'memory': {
                'average': sum(memory_percents) / len(memory_percents) if memory_percents else 0,
                'max': max(memory_percents) if memory_percents else 0,
                'min': min(memory_percents) if memory_percents else 0
            },
            'disk': {
                'average': sum(disk_percents) / len(disk_percents) if disk_percents else 0,
                'max': max(disk_percents) if disk_percents else 0,
                'min': min(disk_percents) if disk_percents else 0
            }
        }
        
        return summary


# 便捷函数和装饰器
def timeit(func: Callable = None, *, name: str = None, logger: logging.Logger = None):
    """
    计时装饰器
    
    Args:
        func (Callable): 被装饰的函数
        name (str): 自定义名称
        logger (logging.Logger): 日志记录器
        
    Returns:
        装饰后的函数
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{f.__module__}.{f.__name__}"
            
            with Timer(timer_name) as timer:
                result = f(*args, **kwargs)
            
            log_msg = f"{timer_name} took {timer.elapsed_time:.4f} seconds"
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def measure_time(name: str = "Operation", logger: logging.Logger = None):
    """
    时间测量上下文管理器
    
    Args:
        name (str): 操作名称
        logger (logging.Logger): 日志记录器
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_time = time.perf_counter() - start_time
        log_msg = f"{name} took {elapsed_time:.4f} seconds"
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)


@contextmanager
def measure_memory(name: str = "Operation", logger: logging.Logger = None):
    """
    内存测量上下文管理器
    
    Args:
        name (str): 操作名称
        logger (logging.Logger): 日志记录器
    """
    monitor = MemoryMonitor()
    memory_before = monitor.get_memory_usage()
    
    try:
        yield monitor
    finally:
        memory_after = monitor.get_memory_usage()
        memory_delta = memory_after - memory_before
        
        log_msg = f"{name} memory delta: {memory_delta:.2f} MB"
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)


# 全局性能分析器实例
_global_profiler = PerformanceProfiler("Global")


def profile(func: Callable = None, *, name: str = None):
    """
    全局性能分析装饰器
    
    Args:
        func (Callable): 被装饰的函数
        name (str): 自定义名称
        
    Returns:
        装饰后的函数
    """
    return _global_profiler.profile_function(func, name=name)


def get_global_profiler() -> PerformanceProfiler:
    """
    获取全局性能分析器
    
    Returns:
        PerformanceProfiler: 全局性能分析器
    """
    return _global_profiler


def print_performance_summary(top_n: int = 10):
    """
    打印全局性能摘要
    
    Args:
        top_n (int): 显示前N个最耗时的函数
    """
    _global_profiler.print_summary(top_n=top_n)