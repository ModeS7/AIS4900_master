"""
Configuration module for AIS4005_IP project.

This module provides centralized configuration management for paths,
hyperparameters, and environment-specific settings.
"""
from .paths import PathConfig, get_path_config, ComputeEnvironment

__all__ = ['PathConfig', 'get_path_config', 'ComputeEnvironment']
