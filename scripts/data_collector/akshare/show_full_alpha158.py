#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show complete Alpha158 feature set (158 features)
"""

import sys
sys.path.insert(0, '/root/mycode/qlibjianbo')

from qlib.contrib.data.loader import Alpha158DL
from loguru import logger

def show_complete_alpha158():
    """Show the complete Alpha158 feature set with all 158 features"""
    
    logger.info("🔍 Complete Alpha158 Feature Set (158 Features)")
    logger.info("=" * 60)
    
    # Full Alpha158 configuration to get all 158 features
    full_config = {
        "kbar": {},  # 9 features
        "price": {
            "windows": [0, 1, 2, 3, 4],  # 5 windows
            "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]  # 5 fields = 25 features
        },
        "volume": {
            "windows": [0, 1, 2, 3, 4]  # 5 features
        },
        "rolling": {
            "windows": [5, 10, 20, 30, 60],  # 5 windows
            "include": None,  # Use all default operators
            "exclude": []
        }
    }
    
    # Get all features
    fields, names = Alpha158DL.get_feature_config(full_config)
    
    logger.info(f"Total features: {len(fields)}")
    logger.info(f"Feature names: {len(names)}")
    
    # Categorize features
    categories = {
        'K-bar': [],
        'Price': [],
        'Volume': [],
        'Rolling': []
    }
    
    for name in names:
        if name.startswith('K'):
            categories['K-bar'].append(name)
        elif any(name.startswith(p) for p in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP']):
            categories['Price'].append(name)
        elif name.startswith('VOLUME'):
            categories['Volume'].append(name)
        else:
            categories['Rolling'].append(name)
    
    # Show breakdown
    logger.info("\n📊 Feature Breakdown:")
    total = 0
    for category, features in categories.items():
        logger.info(f"  {category:8}: {len(features):3d} features")
        total += len(features)
    logger.info(f"  {'Total':8}: {total:3d} features")
    
    # Show all feature names by category
    logger.info("\n📝 Complete Feature List:")
    
    for category, features in categories.items():
        logger.info(f"\n{category} Features ({len(features)}):")
        for i, feature in enumerate(features):
            if i % 10 == 0:
                logger.info(f"  {', '.join(features[i:i+10])}")
    
    # Show sample expressions for rolling features
    logger.info("\n🧮 Rolling Feature Operators:")
    rolling_ops = set()
    for name in categories['Rolling']:
        for op in ['ROC', 'MA', 'STD', 'RANK', 'MAX', 'MIN', 'QTLU', 'QTLD', 'RSQR', 'RESI', 'BETA', 'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN', 'SUMD', 'VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD']:
            if name.startswith(op):
                rolling_ops.add(op)
                break
    
    logger.info(f"  Rolling operators used: {sorted(rolling_ops)}")
    
    return len(fields)

if __name__ == "__main__":
    feature_count = show_complete_alpha158()
    if feature_count >= 158:
        logger.info(f"\n✅ Successfully showing {feature_count} Alpha158+ features!")
    else:
        logger.info(f"\n⚠️  Showing {feature_count} features (may need config adjustment for full 158)")
