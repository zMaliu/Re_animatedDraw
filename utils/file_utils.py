# -*- coding: utf-8 -*-
"""
文件操作工具

提供文件和目录的读写、管理功能
包括文件操作、路径处理、数据序列化等
"""

import os
import json
import pickle
import yaml
import csv
import shutil
import glob
import zipfile
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import hashlib
import tempfile
from datetime import datetime
import mimetypes
import configparser
from xml.etree import ElementTree as ET
import h5py
import numpy as np


class FileManager:
    """
    文件管理器
    
    提供文件和目录的基本操作功能
    """
    
    def __init__(self, config=None):
        """
        初始化文件管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.temp_files = []  # 临时文件列表
    
    def create_directory(self, dir_path: str, exist_ok: bool = True) -> bool:
        """
        创建目录
        
        Args:
            dir_path (str): 目录路径
            exist_ok (bool): 如果目录已存在是否报错
            
        Returns:
            bool: 是否成功
        """
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=exist_ok)
            self.logger.info(f"Directory created: {dir_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating directory {dir_path}: {str(e)}")
            return False
    
    def remove_directory(self, dir_path: str, force: bool = False) -> bool:
        """
        删除目录
        
        Args:
            dir_path (str): 目录路径
            force (bool): 是否强制删除非空目录
            
        Returns:
            bool: 是否成功
        """
        try:
            if not os.path.exists(dir_path):
                self.logger.warning(f"Directory does not exist: {dir_path}")
                return True
            
            if force:
                shutil.rmtree(dir_path)
            else:
                os.rmdir(dir_path)
            
            self.logger.info(f"Directory removed: {dir_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing directory {dir_path}: {str(e)}")
            return False
    
    def copy_file(self, src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        """
        复制文件
        
        Args:
            src_path (str): 源文件路径
            dst_path (str): 目标文件路径
            overwrite (bool): 是否覆盖已存在的文件
            
        Returns:
            bool: 是否成功
        """
        try:
            if not os.path.exists(src_path):
                self.logger.error(f"Source file does not exist: {src_path}")
                return False
            
            if os.path.exists(dst_path) and not overwrite:
                self.logger.warning(f"Destination file already exists: {dst_path}")
                return False
            
            # 确保目标目录存在
            dst_dir = os.path.dirname(dst_path)
            if dst_dir:
                self.create_directory(dst_dir)
            
            shutil.copy2(src_path, dst_path)
            self.logger.info(f"File copied: {src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error copying file: {str(e)}")
            return False
    
    def move_file(self, src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        """
        移动文件
        
        Args:
            src_path (str): 源文件路径
            dst_path (str): 目标文件路径
            overwrite (bool): 是否覆盖已存在的文件
            
        Returns:
            bool: 是否成功
        """
        try:
            if not os.path.exists(src_path):
                self.logger.error(f"Source file does not exist: {src_path}")
                return False
            
            if os.path.exists(dst_path) and not overwrite:
                self.logger.warning(f"Destination file already exists: {dst_path}")
                return False
            
            # 确保目标目录存在
            dst_dir = os.path.dirname(dst_path)
            if dst_dir:
                self.create_directory(dst_dir)
            
            shutil.move(src_path, dst_path)
            self.logger.info(f"File moved: {src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving file: {str(e)}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        删除文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"File deleted: {file_path}")
            else:
                self.logger.warning(f"File does not exist: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def list_files(self, dir_path: str, pattern: str = "*", 
                  recursive: bool = False) -> List[str]:
        """
        列出目录中的文件
        
        Args:
            dir_path (str): 目录路径
            pattern (str): 文件模式
            recursive (bool): 是否递归搜索
            
        Returns:
            List[str]: 文件路径列表
        """
        try:
            if recursive:
                search_pattern = os.path.join(dir_path, "**", pattern)
                files = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(dir_path, pattern)
                files = glob.glob(search_pattern)
            
            # 只返回文件，不包括目录
            files = [f for f in files if os.path.isfile(f)]
            
            self.logger.info(f"Found {len(files)} files in {dir_path}")
            return files
            
        except Exception as e:
            self.logger.error(f"Error listing files in {dir_path}: {str(e)}")
            return []
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            Optional[Dict[str, Any]]: 文件信息字典
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            
            info = {
                'path': file_path,
                'name': os.path.basename(file_path),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'accessed': datetime.fromtimestamp(stat.st_atime),
                'is_file': os.path.isfile(file_path),
                'is_dir': os.path.isdir(file_path),
                'extension': os.path.splitext(file_path)[1],
                'mime_type': mimetypes.guess_type(file_path)[0]
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return None
    
    def calculate_file_hash(self, file_path: str, algorithm: str = 'md5') -> Optional[str]:
        """
        计算文件哈希值
        
        Args:
            file_path (str): 文件路径
            algorithm (str): 哈希算法 ('md5', 'sha1', 'sha256')
            
        Returns:
            Optional[str]: 哈希值
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return None
    
    def create_temp_file(self, suffix: str = "", prefix: str = "tmp", 
                        dir: Optional[str] = None) -> Optional[str]:
        """
        创建临时文件
        
        Args:
            suffix (str): 文件后缀
            prefix (str): 文件前缀
            dir (Optional[str]): 临时目录
            
        Returns:
            Optional[str]: 临时文件路径
        """
        try:
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
            os.close(fd)  # 关闭文件描述符
            
            self.temp_files.append(temp_path)
            self.logger.info(f"Temporary file created: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Error creating temporary file: {str(e)}")
            return None
    
    def cleanup_temp_files(self) -> bool:
        """
        清理临时文件
        
        Returns:
            bool: 是否成功
        """
        try:
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            self.temp_files.clear()
            self.logger.info("Temporary files cleaned up")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary files: {str(e)}")
            return False


class DataSerializer:
    """
    数据序列化器
    
    提供多种格式的数据序列化和反序列化功能
    """
    
    def __init__(self, config=None):
        """
        初始化数据序列化器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def save_json(self, data: Any, file_path: str, indent: int = 2, 
                 ensure_ascii: bool = False) -> bool:
        """
        保存JSON文件
        
        Args:
            data (Any): 要保存的数据
            file_path (str): 文件路径
            indent (int): 缩进空格数
            ensure_ascii (bool): 是否确保ASCII编码
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, 
                         default=self._json_serializer)
            
            self.logger.info(f"JSON data saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving JSON to {file_path}: {str(e)}")
            return False
    
    def load_json(self, file_path: str) -> Optional[Any]:
        """
        加载JSON文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            Optional[Any]: 加载的数据
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"JSON file does not exist: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"JSON data loaded from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading JSON from {file_path}: {str(e)}")
            return None
    
    def save_pickle(self, data: Any, file_path: str, protocol: int = pickle.HIGHEST_PROTOCOL) -> bool:
        """
        保存Pickle文件
        
        Args:
            data (Any): 要保存的数据
            file_path (str): 文件路径
            protocol (int): Pickle协议版本
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=protocol)
            
            self.logger.info(f"Pickle data saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving pickle to {file_path}: {str(e)}")
            return False
    
    def load_pickle(self, file_path: str) -> Optional[Any]:
        """
        加载Pickle文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            Optional[Any]: 加载的数据
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Pickle file does not exist: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.info(f"Pickle data loaded from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading pickle from {file_path}: {str(e)}")
            return None
    
    def save_yaml(self, data: Any, file_path: str, default_flow_style: bool = False) -> bool:
        """
        保存YAML文件
        
        Args:
            data (Any): 要保存的数据
            file_path (str): 文件路径
            default_flow_style (bool): 是否使用流式风格
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=default_flow_style, 
                         allow_unicode=True, indent=2)
            
            self.logger.info(f"YAML data saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving YAML to {file_path}: {str(e)}")
            return False
    
    def load_yaml(self, file_path: str) -> Optional[Any]:
        """
        加载YAML文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            Optional[Any]: 加载的数据
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"YAML file does not exist: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            self.logger.info(f"YAML data loaded from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading YAML from {file_path}: {str(e)}")
            return None
    
    def save_csv(self, data: List[Dict[str, Any]], file_path: str, 
                fieldnames: Optional[List[str]] = None) -> bool:
        """
        保存CSV文件
        
        Args:
            data (List[Dict[str, Any]]): 要保存的数据
            file_path (str): 文件路径
            fieldnames (Optional[List[str]]): 字段名列表
            
        Returns:
            bool: 是否成功
        """
        try:
            if not data:
                self.logger.warning("No data to save to CSV")
                return True
            
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 如果没有指定字段名，使用第一行的键
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            self.logger.info(f"CSV data saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving CSV to {file_path}: {str(e)}")
            return False
    
    def load_csv(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        加载CSV文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            Optional[List[Dict[str, Any]]]: 加载的数据
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"CSV file does not exist: {file_path}")
                return None
            
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(dict(row))
            
            self.logger.info(f"CSV data loaded from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading CSV from {file_path}: {str(e)}")
            return None
    
    def save_hdf5(self, data: Dict[str, np.ndarray], file_path: str) -> bool:
        """
        保存HDF5文件
        
        Args:
            data (Dict[str, np.ndarray]): 要保存的数据字典
            file_path (str): 文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with h5py.File(file_path, 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)
            
            self.logger.info(f"HDF5 data saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving HDF5 to {file_path}: {str(e)}")
            return False
    
    def load_hdf5(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        加载HDF5文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            Optional[Dict[str, np.ndarray]]: 加载的数据字典
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"HDF5 file does not exist: {file_path}")
                return None
            
            data = {}
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:]
            
            self.logger.info(f"HDF5 data loaded from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading HDF5 from {file_path}: {str(e)}")
            return None
    
    def _json_serializer(self, obj):
        """
        JSON序列化辅助函数
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            序列化后的对象
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ArchiveManager:
    """
    压缩文件管理器
    
    提供压缩和解压缩功能
    """
    
    def __init__(self, config=None):
        """
        初始化压缩文件管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_zip(self, source_path: str, zip_path: str, 
                  include_patterns: Optional[List[str]] = None,
                  exclude_patterns: Optional[List[str]] = None) -> bool:
        """
        创建ZIP压缩文件
        
        Args:
            source_path (str): 源路径
            zip_path (str): ZIP文件路径
            include_patterns (Optional[List[str]]): 包含模式列表
            exclude_patterns (Optional[List[str]]): 排除模式列表
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目标目录存在
            zip_dir = os.path.dirname(zip_path)
            if zip_dir:
                os.makedirs(zip_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.isfile(source_path):
                    # 单个文件
                    zipf.write(source_path, os.path.basename(source_path))
                else:
                    # 目录
                    for root, dirs, files in os.walk(source_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            
                            # 检查包含和排除模式
                            if self._should_include_file(file_path, include_patterns, exclude_patterns):
                                arcname = os.path.relpath(file_path, source_path)
                                zipf.write(file_path, arcname)
            
            self.logger.info(f"ZIP archive created: {zip_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating ZIP archive: {str(e)}")
            return False
    
    def extract_zip(self, zip_path: str, extract_path: str) -> bool:
        """
        解压ZIP文件
        
        Args:
            zip_path (str): ZIP文件路径
            extract_path (str): 解压路径
            
        Returns:
            bool: 是否成功
        """
        try:
            if not os.path.exists(zip_path):
                self.logger.error(f"ZIP file does not exist: {zip_path}")
                return False
            
            # 确保解压目录存在
            os.makedirs(extract_path, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_path)
            
            self.logger.info(f"ZIP archive extracted to: {extract_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting ZIP archive: {str(e)}")
            return False
    
    def create_tar(self, source_path: str, tar_path: str, 
                  compression: str = 'gz') -> bool:
        """
        创建TAR压缩文件
        
        Args:
            source_path (str): 源路径
            tar_path (str): TAR文件路径
            compression (str): 压缩格式 ('gz', 'bz2', 'xz', '')
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目标目录存在
            tar_dir = os.path.dirname(tar_path)
            if tar_dir:
                os.makedirs(tar_dir, exist_ok=True)
            
            mode = 'w'
            if compression:
                mode += ':' + compression
            
            with tarfile.open(tar_path, mode) as tarf:
                if os.path.isfile(source_path):
                    # 单个文件
                    tarf.add(source_path, arcname=os.path.basename(source_path))
                else:
                    # 目录
                    tarf.add(source_path, arcname=os.path.basename(source_path))
            
            self.logger.info(f"TAR archive created: {tar_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating TAR archive: {str(e)}")
            return False
    
    def extract_tar(self, tar_path: str, extract_path: str) -> bool:
        """
        解压TAR文件
        
        Args:
            tar_path (str): TAR文件路径
            extract_path (str): 解压路径
            
        Returns:
            bool: 是否成功
        """
        try:
            if not os.path.exists(tar_path):
                self.logger.error(f"TAR file does not exist: {tar_path}")
                return False
            
            # 确保解压目录存在
            os.makedirs(extract_path, exist_ok=True)
            
            with tarfile.open(tar_path, 'r:*') as tarf:
                tarf.extractall(extract_path)
            
            self.logger.info(f"TAR archive extracted to: {extract_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting TAR archive: {str(e)}")
            return False
    
    def _should_include_file(self, file_path: str, 
                           include_patterns: Optional[List[str]], 
                           exclude_patterns: Optional[List[str]]) -> bool:
        """
        检查文件是否应该包含在压缩包中
        
        Args:
            file_path (str): 文件路径
            include_patterns (Optional[List[str]]): 包含模式列表
            exclude_patterns (Optional[List[str]]): 排除模式列表
            
        Returns:
            bool: 是否应该包含
        """
        import fnmatch
        
        # 检查排除模式
        if exclude_patterns:
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file_path, pattern):
                    return False
        
        # 检查包含模式
        if include_patterns:
            for pattern in include_patterns:
                if fnmatch.fnmatch(file_path, pattern):
                    return True
            return False
        
        return True


class ConfigManager:
    """
    配置文件管理器
    
    提供配置文件的读写和管理功能
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path (Optional[str]): 配置文件路径
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.config_data = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> bool:
        """
        加载配置文件
        
        Args:
            config_path (Optional[str]): 配置文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            if config_path:
                self.config_path = config_path
            
            if not self.config_path or not os.path.exists(self.config_path):
                self.logger.warning(f"Config file does not exist: {self.config_path}")
                return False
            
            # 根据文件扩展名选择加载方式
            ext = os.path.splitext(self.config_path)[1].lower()
            
            if ext == '.json':
                serializer = DataSerializer()
                self.config_data = serializer.load_json(self.config_path) or {}
            
            elif ext in ['.yml', '.yaml']:
                serializer = DataSerializer()
                self.config_data = serializer.load_yaml(self.config_path) or {}
            
            elif ext in ['.ini', '.cfg']:
                config = configparser.ConfigParser()
                config.read(self.config_path, encoding='utf-8')
                self.config_data = {section: dict(config[section]) 
                                  for section in config.sections()}
            
            else:
                self.logger.error(f"Unsupported config file format: {ext}")
                return False
            
            self.logger.info(f"Config loaded from {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        保存配置文件
        
        Args:
            config_path (Optional[str]): 配置文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            if config_path:
                self.config_path = config_path
            
            if not self.config_path:
                self.logger.error("No config path specified")
                return False
            
            # 根据文件扩展名选择保存方式
            ext = os.path.splitext(self.config_path)[1].lower()
            
            if ext == '.json':
                serializer = DataSerializer()
                return serializer.save_json(self.config_data, self.config_path)
            
            elif ext in ['.yml', '.yaml']:
                serializer = DataSerializer()
                return serializer.save_yaml(self.config_data, self.config_path)
            
            elif ext in ['.ini', '.cfg']:
                config = configparser.ConfigParser()
                for section, options in self.config_data.items():
                    config.add_section(section)
                    for key, value in options.items():
                        config.set(section, key, str(value))
                
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    config.write(f)
                
                self.logger.info(f"Config saved to {self.config_path}")
                return True
            
            else:
                self.logger.error(f"Unsupported config file format: {ext}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key (str): 配置键，支持点号分隔的嵌套键
            default (Any): 默认值
            
        Returns:
            Any: 配置值
        """
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值
        
        Args:
            key (str): 配置键，支持点号分隔的嵌套键
            value (Any): 配置值
            
        Returns:
            bool: 是否成功
        """
        try:
            keys = key.split('.')
            config = self.config_data
            
            # 创建嵌套字典结构
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config value: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除配置值
        
        Args:
            key (str): 配置键
            
        Returns:
            bool: 是否成功
        """
        try:
            keys = key.split('.')
            config = self.config_data
            
            # 导航到父级字典
            for k in keys[:-1]:
                if isinstance(config, dict) and k in config:
                    config = config[k]
                else:
                    return False
            
            # 删除最后一个键
            if isinstance(config, dict) and keys[-1] in config:
                del config[keys[-1]]
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting config value: {str(e)}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return self.config_data.copy()
    
    def update(self, config_dict: Dict[str, Any]) -> bool:
        """
        更新配置
        
        Args:
            config_dict (Dict[str, Any]): 配置字典
            
        Returns:
            bool: 是否成功
        """
        try:
            self.config_data.update(config_dict)
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating config: {str(e)}")
            return False