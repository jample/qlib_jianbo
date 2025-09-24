#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Alpha158 Handler for Shanghai Stock Data

This module provides an enhanced Alpha158 handler that enables rolling features
needed for correlation operations (CORR, CORD, RSQR, WVMA, VSTD, RESI).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL


class Alpha158Enhanced(Alpha158):
    """Enhanced Alpha158 handler that enables rolling features for correlation operations"""
    
    def get_feature_config(self):
        """Get Alpha158 feature config with rolling features enabled"""
        # Enable rolling features needed for CORR, CORD, RSQR, WVMA, VSTD, RESI
        config = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],
                "include": ["CORR", "CORD", "RSQR", "WVMA", "VSTD", "RESI", "STD", "ROC"],
                "exclude": []
            }
        }
        
        return Alpha158DL.get_feature_config(config)


class Alpha158Robust(Alpha158):
    """Robust Alpha158 handler that only uses features that work with basic OHLCV data"""

    def get_feature_config(self):
        """Get Alpha158 feature config with enhanced predictive features"""
        # Enhanced feature set with better predictive power
        config = {
            "kbar": {},  # 9 K-bar features (momentum, volatility indicators)
            "price": {
                "windows": [0, 1, 2, 3, 4],  # Multi-day price features
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE"],  # Remove VWAP to avoid issues
            },
            "volume": {
                "windows": [0, 1, 2, 3, 4]  # Volume features for liquidity analysis
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],  # Multiple time horizons
                "include": ["STD", "ROC", "RESI", "MA", "MAX", "LOW", "RANK", "RSV", "BETA", "RSQR"],  # Enhanced feature set
                "exclude": ["CORR", "CORD", "WVMA", "VSTD", "VMA"]  # Exclude volume-correlation ops that may fail
            }
        }

        return Alpha158DL.get_feature_config(config)
