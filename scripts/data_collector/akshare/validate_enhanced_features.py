#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate Enhanced Features

This script validates the enhanced ROI and BOLL features implementation
and demonstrates the new prediction targets.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.insert(0, '/root/mycode/qlibjianbo')
sys.path.insert(0, '/root/mycode/qlibjianbo/scripts/data_collector/akshare')

from alpha158_enhanced import Alpha158WithROIAndBOLL


def create_sample_data() -> pd.DataFrame:
    """Create sample stock data for validation"""
    logger.info("Creating sample stock data...")
    
    # Create realistic stock price data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    dates = dates[dates.dayofweek < 5]  # Business days only
    
    data = []
    base_price = 100.0
    
    for i, date in enumerate(dates):
        # Simulate price movement with some trend and volatility
        price_change = np.random.normal(0, 0.02)  # 2% daily volatility
        base_price *= (1 + price_change)
        
        # Generate OHLCV data
        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = base_price * (1 + np.random.normal(0, 0.005))
        close = base_price
        volume = int(1000000 * (1 + np.random.normal(0, 0.3)))
        
        data.append({
            'symbol': '600000',
            'date': date,
            'open': max(low, min(high, open_price)),
            'high': high,
            'low': low,
            'close': max(low, min(high, close)),
            'volume': max(1000, volume),
            'amount': close * volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created {len(df)} records of sample data")
    return df


def validate_roi_features(df: pd.DataFrame) -> bool:
    """Validate ROI feature expressions"""
    logger.info("Validating ROI feature expressions...")

    try:
        # Test the enhanced Alpha158 handler
        handler = Alpha158WithROIAndBOLL()
        fields, names = handler.get_feature_config()

        # Find ROI features
        roi_features = [(i, name) for i, name in enumerate(names) if 'ROI' in name]

        if len(roi_features) < 5:
            logger.error(f"Expected at least 5 ROI features, found {len(roi_features)}")
            return False

        # Validate ROI expressions
        for idx, name in roi_features:
            expr = fields[idx]

            # Check that expressions contain expected qlib operators
            if 'ROI_1D' in name or 'ROI_5D' in name:
                if not ('Ref($close' in expr and '/$close' in expr):
                    logger.error(f"Invalid ROI expression for {name}: {expr}")
                    return False

            elif 'ROI_VOL' in name:
                if not ('Std(' in expr):
                    logger.error(f"Invalid ROI volatility expression for {name}: {expr}")
                    return False

            elif 'ROI_SHARPE' in name:
                if not ('Mean(' in expr and 'Std(' in expr):
                    logger.error(f"Invalid ROI Sharpe expression for {name}: {expr}")
                    return False

        logger.info(f"✅ ROI features validation passed - {len(roi_features)} features validated")
        return True

    except Exception as e:
        logger.error(f"❌ ROI features validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_boll_features(df: pd.DataFrame) -> bool:
    """Validate BOLL feature expressions"""
    logger.info("Validating BOLL feature expressions...")

    try:
        # Test the enhanced Alpha158 handler
        handler = Alpha158WithROIAndBOLL()
        fields, names = handler.get_feature_config()

        # Find BOLL features
        boll_features = [(i, name) for i, name in enumerate(names) if 'BOLL' in name or 'BB_' in name]

        if len(boll_features) < 8:
            logger.error(f"Expected at least 8 BOLL features, found {len(boll_features)}")
            return False

        # Validate BOLL expressions
        for idx, name in boll_features:
            expr = fields[idx]

            # Check that expressions contain expected qlib operators
            if 'BOLL_VOL' in name:
                if not ('Std(' in expr and 'Mean(' in expr):
                    logger.error(f"Invalid BOLL volatility expression for {name}: {expr}")
                    return False

            elif 'BOLL_MOMENTUM' in name:
                if not ('$close' in expr and 'Mean(' in expr and 'Std(' in expr):
                    logger.error(f"Invalid BOLL momentum expression for {name}: {expr}")
                    return False

            elif 'BOLL_TREND' in name:
                if not ('Mean(' in expr and 'Ref(' in expr):
                    logger.error(f"Invalid BOLL trend expression for {name}: {expr}")
                    return False

            elif 'BB_' in name:
                if not ('Mean(' in expr or 'Std(' in expr or '$close' in expr):
                    logger.error(f"Invalid Bollinger Band expression for {name}: {expr}")
                    return False

        logger.info(f"✅ BOLL features validation passed - {len(boll_features)} features validated")
        return True

    except Exception as e:
        logger.error(f"❌ BOLL features validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_prediction_targets(df: pd.DataFrame):
    """Demonstrate the new prediction targets"""
    logger.info("Demonstrating prediction targets...")

    try:
        # Test the enhanced Alpha158 handler
        handler = Alpha158WithROIAndBOLL()
        fields, names = handler.get_feature_config()
        label_fields, label_names = handler.get_label_config()

        # Show ROI-based prediction targets
        logger.info("\n--- ROI-based Features ---")
        roi_features = [name for name in names if 'ROI' in name]
        logger.info(f"Available ROI features ({len(roi_features)}): {roi_features}")

        # Show corresponding expressions
        for name in roi_features[:3]:  # Show first 3
            idx = names.index(name)
            logger.info(f"  {name}: {fields[idx]}")

        # Show BOLL-based prediction targets
        logger.info("\n--- BOLL-based Features ---")
        boll_features = [name for name in names if 'BOLL' in name or 'BB_' in name]
        logger.info(f"Available BOLL features ({len(boll_features)}): {boll_features}")

        # Show corresponding expressions
        for name in boll_features[:3]:  # Show first 3
            idx = names.index(name)
            logger.info(f"  {name}: {fields[idx]}")

        # Show label configuration
        logger.info("\n--- Prediction Labels ---")
        logger.info(f"Label names: {label_names}")
        for i, (name, expr) in enumerate(zip(label_names, label_fields)):
            logger.info(f"  {name}: {expr}")

        # Show feature statistics
        logger.info("\n--- Feature Statistics ---")
        logger.info(f"Total features: {len(names)}")
        logger.info(f"ROI features: {len(roi_features)}")
        logger.info(f"BOLL features: {len(boll_features)}")
        logger.info(f"Prediction labels: {len(label_names)}")

        # Save configuration
        config_data = {
            'features': dict(zip(names, fields)),
            'labels': dict(zip(label_names, label_fields)),
            'feature_count': len(names),
            'roi_features': roi_features,
            'boll_features': boll_features
        }

        import json
        output_file = Path("enhanced_features_config.json")
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Feature configuration saved to: {output_file}")

        logger.info("✅ Prediction targets demonstration completed")

    except Exception as e:
        logger.error(f"❌ Prediction targets demonstration failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main validation function"""
    logger.info("=" * 60)
    logger.info("ENHANCED FEATURES VALIDATION")
    logger.info("=" * 60)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Run validations
    tests = [
        ("ROI Features", lambda: validate_roi_features(sample_data)),
        ("BOLL Features", lambda: validate_boll_features(sample_data))
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Validation ---")
        result = test_func()
        results.append((test_name, result))
    
    # Demonstrate prediction targets
    logger.info(f"\n--- Prediction Targets Demonstration ---")
    demonstrate_prediction_targets(sample_data)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        logger.info("🎉 All validations passed! Enhanced features are working correctly.")
        return True
    else:
        logger.error("❌ Some validations failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
