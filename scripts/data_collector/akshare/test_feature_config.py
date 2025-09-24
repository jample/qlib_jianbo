#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Feature Configuration

This script tests the enhanced Alpha158 feature configuration without requiring
qlib initialization or data loading.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add paths for imports
sys.path.insert(0, '/root/mycode/qlibjianbo')
sys.path.insert(0, '/root/mycode/qlibjianbo/scripts/data_collector/akshare')

# Import the enhanced Alpha158 class directly
from alpha158_enhanced import Alpha158WithROIAndBOLL


def test_feature_configuration():
    """Test the feature configuration without qlib initialization"""
    logger.info("Testing enhanced Alpha158 feature configuration...")
    
    try:
        # Create handler instance (this will fail if we try to load data)
        # But we can still test the configuration methods
        handler = Alpha158WithROIAndBOLL.__new__(Alpha158WithROIAndBOLL)
        
        # Test feature configuration
        fields, names = handler.get_feature_config()
        
        logger.info(f"✅ Feature configuration loaded successfully")
        logger.info(f"Total features: {len(fields)}")
        logger.info(f"Feature names: {len(names)}")
        
        # Test ROI features
        roi_features = [name for name in names if 'ROI' in name]
        logger.info(f"ROI features ({len(roi_features)}): {roi_features}")
        
        # Test BOLL features
        boll_features = [name for name in names if 'BOLL' in name or 'BB_' in name]
        logger.info(f"BOLL features ({len(boll_features)}): {boll_features}")
        
        # Test label configuration
        label_fields, label_names = handler.get_label_config()
        logger.info(f"Labels ({len(label_names)}): {label_names}")
        
        # Validate ROI expressions
        logger.info("\n--- ROI Feature Expressions ---")
        for name in roi_features[:5]:  # Show first 5
            idx = names.index(name)
            expr = fields[idx]
            logger.info(f"{name}: {expr}")
            
            # Basic validation
            if 'ROI_1D' in name or 'ROI_5D' in name:
                assert 'Ref($close' in expr and '/$close' in expr, f"Invalid ROI expression: {expr}"
            elif 'ROI_VOL' in name:
                assert 'Std(' in expr, f"Invalid ROI volatility expression: {expr}"
        
        # Validate BOLL expressions
        logger.info("\n--- BOLL Feature Expressions ---")
        for name in boll_features[:5]:  # Show first 5
            idx = names.index(name)
            expr = fields[idx]
            logger.info(f"{name}: {expr}")
            
            # Basic validation
            if 'BOLL_VOL' in name:
                assert 'Std(' in expr and 'Mean(' in expr, f"Invalid BOLL volatility expression: {expr}"
            elif 'BB_' in name:
                assert 'Mean(' in expr or 'Std(' in expr, f"Invalid Bollinger Band expression: {expr}"
        
        # Validate label expressions
        logger.info("\n--- Label Expressions ---")
        for name, expr in zip(label_names, label_fields):
            logger.info(f"{name}: {expr}")
            
            if 'ROI' in name:
                assert 'Ref($close' in expr and '/$close' in expr, f"Invalid ROI label expression: {expr}"
            elif 'BOLL' in name:
                assert 'Std(' in expr and 'Mean(' in expr, f"Invalid BOLL label expression: {expr}"
        
        # Summary
        logger.info("\n--- Configuration Summary ---")
        logger.info(f"✅ Total features: {len(names)}")
        logger.info(f"✅ ROI features: {len(roi_features)}")
        logger.info(f"✅ BOLL features: {len(boll_features)}")
        logger.info(f"✅ Prediction labels: {len(label_names)}")
        
        # Check for required features
        required_features = [
            'ROI_1D', 'ROI_5D', 'ROI_RATIO_1D_5D',
            'BOLL_VOL_1D', 'BOLL_VOL_5D', 'BOLL_VOL_RATIO_1D_5D'
        ]
        
        missing_features = [f for f in required_features if f not in names]
        if missing_features:
            logger.error(f"❌ Missing required features: {missing_features}")
            return False
        
        required_labels = ['LABEL_ROI_RATIO', 'LABEL_BOLL_VOL_RATIO']
        missing_labels = [l for l in required_labels if l not in label_names]
        if missing_labels:
            logger.error(f"❌ Missing required labels: {missing_labels}")
            return False
        
        logger.info("🎉 All feature configuration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_splits():
    """Test dataset split configuration"""
    logger.info("Testing dataset split configuration...")

    try:
        # Import and check the TFT runner (should work with Python 3.12 now)
        from shanghai_tft_runner import ShanghaiTFTRunner
        import inspect

        # Check default parameters for both TFT and alternative methods
        sig_tft = inspect.signature(ShanghaiTFTRunner.step3_train_tft_model)
        sig_alt = inspect.signature(ShanghaiTFTRunner.step3_alternative_model_training)

        expected_defaults = {
            'train_start': '2023-01-01',
            'train_end': '2024-06-07',
            'valid_start': '2024-06-10',
            'valid_end': '2024-09-30',
            'test_start': '2024-10-16',
            'test_end': '2024-12-31'
        }

        # Test TFT method defaults
        logger.info("Testing TFT method defaults:")
        for param, expected in expected_defaults.items():
            actual = sig_tft.parameters[param].default
            assert actual == expected, f"Wrong TFT default for {param}: expected {expected}, got {actual}"
            logger.info(f"✅ TFT {param}: {actual}")

        # Test alternative method defaults
        logger.info("Testing alternative method defaults:")
        for param, expected in expected_defaults.items():
            actual = sig_alt.parameters[param].default
            assert actual == expected, f"Wrong alternative default for {param}: expected {expected}, got {actual}"
            logger.info(f"✅ Alternative {param}: {actual}")

        # Test that the runner can be instantiated without errors
        runner = ShanghaiTFTRunner(model_folder="test_models")
        logger.info("✅ Runner instantiation successful")

        logger.info("✅ Dataset split configuration test passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Dataset split test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("ENHANCED TFT FEATURE CONFIGURATION TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Feature Configuration", test_feature_configuration),
        ("Dataset Splits", test_dataset_splits)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Test ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced TFT configuration is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
