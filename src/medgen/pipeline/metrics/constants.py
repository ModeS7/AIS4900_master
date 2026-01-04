"""Shared constants for regional metrics tracking.

This module provides a single source of truth for tumor classification thresholds
based on RANO-BM (Response Assessment in Neuro-Oncology Brain Metastases) guidelines.
"""
from typing import Dict, Tuple

# RANO-BM clinical thresholds for tumor size classification by Feret diameter (mm)
# Feret diameter = longest distance between any two points on the tumor boundary
TUMOR_SIZE_THRESHOLDS_MM: Dict[str, Tuple[float, float]] = {
    'tiny': (0, 10),        # <10mm Feret diameter
    'small': (10, 20),      # 10-20mm
    'medium': (20, 30),     # 20-30mm
    'large': (30, float('inf')),  # >=30mm (no upper bound)
}

# Ordered tuple for consistent iteration
TUMOR_SIZE_CATEGORIES = tuple(TUMOR_SIZE_THRESHOLDS_MM.keys())
