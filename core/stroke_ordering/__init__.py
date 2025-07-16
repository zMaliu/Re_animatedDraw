# -*- coding: utf-8 -*-
"""
笔触排序模块

提供笔触排序和优化功能
"""

from .energy_function import EnergyFunction, EnergyComponents
from .nes_optimizer import NESOptimizer, OptimizationResult
from .constraint_handler import ConstraintHandler, StrokeConstraint
from .order_evaluator import OrderEvaluator, EvaluationMetric, EvaluationResult, OverallEvaluation
from .spearman_correlation import SpearmanCorrelationCalculator, CorrelationResult
from .stroke_organizer import StrokeOrganizer, OrganizationResult
from .ordering_optimizer import OrderingOptimizer, OptimizationResult
from .rule_based_ordering import RuleBasedOrdering, RuleBasedResult, OrderingRule, OrderingContext

__all__ = [
    # Energy function
    'EnergyFunction',
    'EnergyComponents', 
    
    # NES optimizer
    'NESOptimizer',
    'OptimizationResult',
    
    # Constraint handler
    'ConstraintHandler',
    'StrokeConstraint',
    
    # Order evaluator
    'OrderEvaluator',
    'EvaluationMetric',
    'EvaluationResult',
    'OverallEvaluation',
    
    # Spearman correlation
    'SpearmanCorrelationCalculator',
    'CorrelationResult',
    
    # Stroke organizer
    'StrokeOrganizer',
    'OrganizationResult',
    
    # Ordering optimizer
    'OrderingOptimizer',
    
    # Rule-based ordering
    'RuleBasedOrdering',
    'RuleBasedResult',
    'OrderingRule',
    'OrderingContext'
]