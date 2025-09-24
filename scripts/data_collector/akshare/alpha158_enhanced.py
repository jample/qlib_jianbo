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


class Alpha158WithROIAndBOLL(Alpha158):
    """Enhanced Alpha158 with ROI and BOLL features for TFT prediction"""

    def get_feature_config(self):
        """Get Alpha158 feature config with ROI and BOLL features"""
        # Get base Alpha158 features
        base_config = {
            "kbar": {},  # 9 K-bar features
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],
                "include": ["STD", "ROC", "MA", "MAX", "MIN", "RANK", "RSV", "BETA", "RSQR"],
                "exclude": ["CORR", "CORD", "WVMA", "VSTD"]  # Exclude problematic features
            }
        }

        # Get base features
        fields, names = Alpha158DL.get_feature_config(base_config)

        # Add custom ROI features using qlib expressions
        roi_fields, roi_names = self._get_roi_features()
        fields.extend(roi_fields)
        names.extend(roi_names)

        # Add custom BOLL features using qlib expressions
        boll_fields, boll_names = self._get_boll_features()
        fields.extend(boll_fields)
        names.extend(boll_names)

        return fields, names

    def _get_roi_features(self):
        """Get ROI (Return on Investment) features using qlib expressions"""
        fields = []
        names = []

        # ROI features for different horizons
        windows = [1, 5]

        for d in windows:
            # Forward ROI: (close_t+d / close_t) - 1
            # Using Ref with negative values for future prices
            fields.append("Ref($close, -%d)/$close - 1" % d)
            names.append("ROI_%dD" % d)

            # ROI volatility: std of returns over window
            fields.append("Std($close/Ref($close, 1) - 1, %d)" % d)
            names.append("ROI_VOL_%dD" % d)

        # ROI ratio: 1-day / 5-day
        fields.append("(Ref($close, -1)/$close - 1) / (Ref($close, -5)/$close - 1 + 1e-12)")
        names.append("ROI_RATIO_1D_5D")

        # Cumulative ROI approximation using log returns
        for d in [1, 5]:
            fields.append("Sum(Log($close/Ref($close, 1)), %d)" % d)
            names.append("ROI_CUM_%dD" % d)

        # ROI Sharpe ratio approximation
        for d in [1, 5]:
            fields.append("Mean($close/Ref($close, 1) - 1, %d) / (Std($close/Ref($close, 1) - 1, %d) + 1e-12)" % (d, d))
            names.append("ROI_SHARPE_%dD" % d)

        return fields, names

    def _get_boll_features(self):
        """Get BOLL (Bollinger Bands) features using qlib expressions"""
        fields = []
        names = []

        # BOLL features for different windows
        windows = [1, 5, 20]

        for d in windows:
            # BOLL volatility: std / mean (normalized volatility)
            fields.append("Std($close, %d) / (Mean($close, %d) + 1e-12)" % (d, d))
            names.append("BOLL_VOL_%dD" % d)

            # BOLL momentum: (close - mean) / std (Z-score)
            fields.append("($close - Mean($close, %d)) / (Std($close, %d) + 1e-12)" % (d, d))
            names.append("BOLL_MOMENTUM_%dD" % d)

            # BOLL trend: slope of moving average
            if d > 1:
                fields.append("(Mean($close, %d) - Ref(Mean($close, %d), 1)) / (Ref(Mean($close, %d), 1) + 1e-12)" % (d, d, d))
                names.append("BOLL_TREND_%dD" % d)

        # BOLL volatility ratio: 1-day / 5-day
        fields.append("(Std($close, 1) / (Mean($close, 1) + 1e-12)) / (Std($close, 5) / (Mean($close, 5) + 1e-12) + 1e-12)")
        names.append("BOLL_VOL_RATIO_1D_5D")

        # Traditional Bollinger Bands (20-day, 2 std)
        fields.append("Mean($close, 20) + 2 * Std($close, 20)")
        names.append("BB_UPPER")

        fields.append("Mean($close, 20) - 2 * Std($close, 20)")
        names.append("BB_LOWER")

        fields.append("(2 * Std($close, 20)) / (Mean($close, 20) + 1e-12)")
        names.append("BB_WIDTH")

        fields.append("($close - (Mean($close, 20) - 2 * Std($close, 20))) / ((Mean($close, 20) + 2 * Std($close, 20)) - (Mean($close, 20) - 2 * Std($close, 20)) + 1e-12)")
        names.append("BB_POSITION")

        return fields, names

    def get_label_config(self):
        """Get label configuration for ROI and BOLL prediction targets"""
        # Multi-target labels using qlib expressions
        label_fields = [
            # ROI ratio label: N+1 day / 5 day profit ratio
            "(Ref($close, -1)/$close - 1) / (Ref($close, -5)/$close - 1 + 1e-12)",
            # BOLL volatility ratio label: N+1 day / 5 day BOLL indicator
            "(Std($close, 1) / (Mean($close, 1) + 1e-12)) / (Std($close, 5) / (Mean($close, 5) + 1e-12) + 1e-12)"
        ]

        label_names = ["LABEL_ROI_RATIO", "LABEL_BOLL_VOL_RATIO"]

        return label_fields, label_names
