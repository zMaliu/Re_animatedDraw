# -*- coding: utf-8 -*-
"""
笔触数据库模块

提供笔触的存储、检索、匹配和分类功能
"""

from .stroke_database import StrokeDatabase, StrokeTemplate
from .stroke_matcher import StrokeMatcher, MatchResult
from .stroke_classifier import StrokeClassifier, StrokeCategory, ClassificationResult

__all__ = [
    'StrokeDatabase',
    'StrokeTemplate',
    'StrokeMatcher',
    'MatchResult', 
    'StrokeClassifier',
    'StrokeCategory',
    'ClassificationResult'
]