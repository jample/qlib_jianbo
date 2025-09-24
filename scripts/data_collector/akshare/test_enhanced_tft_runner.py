#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Enhanced Shanghai TFT Runner

This module tests the enhanced Shanghai TFT runner with new ROI and BOLL features,
updated dataset splits, and multi-target prediction capabilities.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.insert(0, '/root/mycode/qlibjianbo')
sys.path.insert(0, '/root/mycode/qlibjianbo/scripts/data_collector/akshare')

from shanghai_tft_runner import ShanghaiTFTRunner
from alpha158_enhanced import Alpha158WithROIAndBOLL


class EnhancedTFTTester:
    """Test class for enhanced Shanghai TFT runner"""
    
    def __init__(self):
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
    def create_test_data(self) -> pd.DataFrame:
        """Create synthetic test data for validation"""
        logger.info("Creating synthetic test data...")
        
        # Create date range covering the new dataset splits
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        # Filter to business days only
        dates = dates[dates.dayofweek < 5]
        
        symbols = ['600000', '600036', '600519']
        
        data = []
        for symbol in symbols:
            for date in dates:
                # Generate synthetic OHLCV data
                base_price = 10 + np.random.normal(0, 0.1)
                high = base_price * (1 + abs(np.random.normal(0, 0.02)))
                low = base_price * (1 - abs(np.random.normal(0, 0.02)))
                open_price = base_price * (1 + np.random.normal(0, 0.01))
                close = base_price * (1 + np.random.normal(0, 0.01))
                volume = int(1000000 * (1 + np.random.normal(0, 0.5)))
                
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': max(low, min(high, open_price)),
                    'high': high,
                    'low': low,
                    'close': max(low, min(high, close)),
                    'volume': max(1000, volume),
                    'amount': max(low, min(high, close)) * max(1000, volume)
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Created test data: {len(df)} records for {len(symbols)} symbols")
        return df
    
    def test_dataset_splits(self) -> bool:
        """Test that dataset splits match the new requirements"""
        logger.info("Testing dataset splits...")
        
        try:
            runner = ShanghaiTFTRunner(model_folder="test_models")
            
            # Check default date ranges
            expected_train_start = "2023-01-01"
            expected_train_end = "2024-06-07"
            expected_valid_start = "2024-06-10"
            expected_valid_end = "2024-09-30"
            expected_test_start = "2024-10-16"
            expected_test_end = "2024-12-31"
            
            # Test method signatures have correct defaults
            import inspect
            sig = inspect.signature(runner.step3_train_tft_model)
            
            assert sig.parameters['train_start'].default == expected_train_start
            assert sig.parameters['train_end'].default == expected_train_end
            assert sig.parameters['valid_start'].default == expected_valid_start
            assert sig.parameters['valid_end'].default == expected_valid_end
            assert sig.parameters['test_start'].default == expected_test_start
            assert sig.parameters['test_end'].default == expected_test_end
            
            logger.info("✅ Dataset splits test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dataset splits test failed: {e}")
            return False
    
    def test_roi_calculations(self) -> bool:
        """Test ROI feature configuration"""
        logger.info("Testing ROI feature configuration...")

        try:
            # Test the enhanced Alpha158 handler
            handler = Alpha158WithROIAndBOLL()
            fields, names = handler.get_feature_config()

            # Check that ROI features are included
            expected_roi_features = [
                'ROI_1D', 'ROI_5D', 'ROI_RATIO_1D_5D',
                'ROI_CUM_1D', 'ROI_CUM_5D',
                'ROI_VOL_1D', 'ROI_VOL_5D',
                'ROI_SHARPE_1D', 'ROI_SHARPE_5D'
            ]

            for feature in expected_roi_features:
                assert feature in names, f"Missing ROI feature: {feature}"

            # Check that corresponding expressions exist
            roi_indices = [i for i, name in enumerate(names) if 'ROI' in name]
            assert len(roi_indices) >= len(expected_roi_features), "Not enough ROI expressions"

            # Verify expressions are valid qlib expressions
            for idx in roi_indices[:3]:  # Check first 3 ROI expressions
                expr = fields[idx]
                assert '$close' in expr or 'Ref(' in expr, f"Invalid ROI expression: {expr}"

            logger.info("✅ ROI feature configuration test passed")
            return True

        except Exception as e:
            logger.error(f"❌ ROI feature configuration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_boll_indicators(self) -> bool:
        """Test BOLL feature configuration"""
        logger.info("Testing BOLL feature configuration...")

        try:
            # Test the enhanced Alpha158 handler
            handler = Alpha158WithROIAndBOLL()
            fields, names = handler.get_feature_config()

            # Check that BOLL features are included
            expected_boll_features = [
                'BOLL_VOL_1D', 'BOLL_VOL_5D', 'BOLL_VOL_RATIO_1D_5D',
                'BOLL_MOMENTUM_1D', 'BOLL_MOMENTUM_5D',
                'BOLL_TREND_5D', 'BOLL_TREND_20D',
                'BB_UPPER', 'BB_LOWER', 'BB_WIDTH', 'BB_POSITION'
            ]

            found_features = [f for f in expected_boll_features if f in names]
            assert len(found_features) >= 8, f"Missing BOLL features. Found: {found_features}"

            # Check that corresponding expressions exist
            boll_indices = [i for i, name in enumerate(names) if 'BOLL' in name or 'BB_' in name]
            assert len(boll_indices) >= 8, "Not enough BOLL expressions"

            # Verify expressions are valid qlib expressions
            for idx in boll_indices[:3]:  # Check first 3 BOLL expressions
                expr = fields[idx]
                assert 'Std(' in expr or 'Mean(' in expr or '$close' in expr, f"Invalid BOLL expression: {expr}"

            logger.info("✅ BOLL feature configuration test passed")
            return True

        except Exception as e:
            logger.error(f"❌ BOLL feature configuration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_label_calculations(self) -> bool:
        """Test multi-target label configuration"""
        logger.info("Testing label configuration...")

        try:
            # Test the enhanced Alpha158 handler label configuration
            handler = Alpha158WithROIAndBOLL()
            label_fields, label_names = handler.get_label_config()

            # Check that new labels are configured
            expected_labels = ['LABEL_ROI_RATIO', 'LABEL_BOLL_VOL_RATIO']

            for label in expected_labels:
                assert label in label_names, f"Missing label: {label}"

            # Check that label expressions are valid
            assert len(label_fields) == len(label_names), "Mismatch between label fields and names"

            for i, (field, name) in enumerate(zip(label_fields, label_names)):
                if 'ROI' in name:
                    assert 'Ref($close' in field and '/$close' in field, f"Invalid ROI label expression: {field}"
                elif 'BOLL' in name:
                    assert 'Std(' in field and 'Mean(' in field, f"Invalid BOLL label expression: {field}"

            logger.info("✅ Label configuration test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Label configuration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        logger.info("=" * 60)
        logger.info("RUNNING ENHANCED TFT RUNNER TESTS")
        logger.info("=" * 60)
        
        tests = [
            ("Dataset Splits", self.test_dataset_splits),
            ("ROI Feature Configuration", self.test_roi_calculations),
            ("BOLL Feature Configuration", self.test_boll_indicators),
            ("Label Configuration", self.test_label_calculations)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            result = test_func()
            results.append((test_name, result))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            logger.info("🎉 All tests passed! Enhanced TFT runner is ready.")
            return True
        else:
            logger.error("❌ Some tests failed. Please check the implementation.")
            return False


def main():
    """Main function to run tests"""
    tester = EnhancedTFTTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
