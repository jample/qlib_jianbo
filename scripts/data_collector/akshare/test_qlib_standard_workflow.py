#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for qlib standard workflow
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from qlib_workflow_runner import QlibWorkflowRunner


def test_data_preparation():
    """Test data preparation step"""
    
    logger.info("🧪 Testing data preparation...")
    
    try:
        runner = QlibWorkflowRunner()
        
        # Test with small dataset
        data_config = {
            "data_type": "stock",
            "exchange_filter": "shanghai",
            "symbol_filter": ["600000", "600036"],  # Just 2 stocks
            "start_date": "2024-01-01",
            "end_date": "2024-03-31"  # Just 3 months
        }
        
        success = runner.step1_prepare_qlib_data(**data_config)
        
        if success:
            logger.info("✅ Data preparation test passed!")
            
            # Check if qlib data structure is created
            qlib_data_dir = Path("scripts/data_collector/akshare/qlib_data")
            
            checks = [
                (qlib_data_dir / "features", "Features directory"),
                (qlib_data_dir / "instruments" / "all.txt", "Instruments file"),
                (qlib_data_dir / "calendars" / "day.txt", "Calendar file")
            ]
            
            for path, description in checks:
                if path.exists():
                    logger.info(f"  ✓ {description}: {path}")
                else:
                    logger.warning(f"  ✗ {description}: {path} not found")
            
            return True
        else:
            logger.error("❌ Data preparation test failed!")
            return False
            
    except Exception as e:
        logger.error(f"Data preparation test failed with error: {e}")
        return False


def test_qlib_workflow_config():
    """Test qlib workflow configuration"""
    
    logger.info("🧪 Testing qlib workflow configuration...")
    
    try:
        config_path = Path("scripts/data_collector/akshare/workflow_config_shanghai_alpha158.yaml")
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        # Read and validate YAML structure
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['qlib_init', 'task']
        for section in required_sections:
            if section in config:
                logger.info(f"  ✓ {section} section found")
            else:
                logger.error(f"  ✗ {section} section missing")
                return False
        
        # Check task structure
        task = config.get('task', {})
        if 'model' in task and 'dataset' in task:
            logger.info("  ✓ Task structure is valid")
            
            # Check Alpha158 handler
            handler = task.get('dataset', {}).get('kwargs', {}).get('handler', {})
            if handler.get('class') == 'Alpha158':
                logger.info("  ✓ Alpha158 handler configured")
            else:
                logger.warning("  ⚠ Alpha158 handler not found")
        else:
            logger.error("  ✗ Task structure is invalid")
            return False
        
        logger.info("✅ Qlib workflow configuration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed with error: {e}")
        return False


def test_qrun_compatibility():
    """Test qrun compatibility"""
    
    logger.info("🧪 Testing qrun compatibility...")
    
    try:
        runner = QlibWorkflowRunner()
        
        # Test qrun compatibility setup
        success = runner.run_with_qrun_compatibility()
        
        if success:
            logger.info("✅ qrun compatibility test passed!")
            logger.info("You can now run:")
            logger.info("  qrun scripts/data_collector/akshare/workflow_config_shanghai_alpha158.yaml")
            return True
        else:
            logger.error("❌ qrun compatibility test failed!")
            return False
            
    except Exception as e:
        logger.error(f"qrun compatibility test failed with error: {e}")
        return False


def main():
    """Main test function"""
    
    logger.info("=" * 60)
    logger.info("QLIB STANDARD WORKFLOW TESTING")
    logger.info("=" * 60)
    
    tests = [
        ("Data Preparation", test_data_preparation),
        ("Workflow Configuration", test_qlib_workflow_config),
        ("qrun Compatibility", test_qrun_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Test: {test_name}")
        logger.info("-" * 40)
        
        result = test_func()
        results.append((test_name, result))
        
        if result:
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name:25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Qlib standard workflow is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python scripts/data_collector/akshare/qlib_workflow_runner.py --prepare-only")
        logger.info("2. Run: qrun scripts/data_collector/akshare/workflow_config_shanghai_alpha158.yaml")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
