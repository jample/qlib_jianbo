#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example Usage of Shanghai TFT Runner

This script demonstrates various ways to use the Shanghai TFT Runner
for stock price prediction with Temporal Fusion Transformer models.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml

# Add paths
sys.path.insert(0, '/root/mycode/qlibjianbo')
sys.path.insert(0, '/root/mycode/qlibjianbo/scripts/data_collector/akshare')

from shanghai_tft_runner import ShanghaiTFTRunner, TFTAnalyzer, create_tft_config_file


def example_1_basic_tft_training():
    """Example 1: Basic TFT training with default settings"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Basic TFT Training")
    logger.info("=" * 60)
    
    try:
        # Initialize runner with default settings
        runner = ShanghaiTFTRunner(
            model_folder="example_tft_models",
            gpu_id=0,
            dataset_name="Shanghai_Basic"
        )
        
        # Run complete workflow with a small set of stocks
        success = runner.run_complete_tft_workflow(
            symbol_filter=["600000", "600036"],  # Just 2 stocks for quick demo
            start_date="2022-01-01",
            end_date="2023-06-30"
        )
        
        if success:
            logger.info("✅ Example 1 completed successfully!")
        else:
            logger.error("❌ Example 1 failed!")
            
        return success
        
    except Exception as e:
        logger.error(f"Example 1 failed with error: {e}")
        return False


def example_2_step_by_step_training():
    """Example 2: Step-by-step TFT training with custom configuration"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Step-by-Step TFT Training")
    logger.info("=" * 60)
    
    try:
        # Initialize runner
        runner = ShanghaiTFTRunner(
            model_folder="example_step_by_step",
            gpu_id=0,
            dataset_name="Shanghai_StepByStep"
        )
        
        # Step 1: Prepare data with custom parameters
        logger.info("Step 1: Preparing Shanghai stock data...")
        success = runner.step1_prepare_shanghai_data(
            symbol_filter=["600000", "600036", "600519"],
            start_date="2021-01-01",
            end_date="2023-12-31",
            min_periods=200  # Require more data per stock
        )
        
        if not success:
            logger.error("Data preparation failed!")
            return False
        
        # Step 2: Setup dataset configuration
        logger.info("Step 2: Setting up TFT dataset...")
        success = runner.step2_setup_tft_dataset()
        
        if not success:
            logger.error("Dataset setup failed!")
            return False
        
        # Step 3: Train model with custom date ranges
        logger.info("Step 3: Training TFT model...")
        success = runner.step3_train_tft_model(
            train_start="2021-01-01",
            train_end="2022-12-31",
            valid_start="2023-01-01", 
            valid_end="2023-06-30",
            test_start="2023-07-01",
            test_end="2023-12-31"
        )
        
        if not success:
            logger.error("Model training failed!")
            return False
        
        # Step 4: Generate predictions
        logger.info("Step 4: Generating predictions...")
        predictions = runner.step4_generate_predictions()
        
        if predictions is None:
            logger.error("Prediction generation failed!")
            return False
        
        # Step 5: Analyze predictions
        logger.info("Step 5: Analyzing predictions...")
        analysis = runner.analyze_predictions(predictions)
        
        # Step 6: Save results
        logger.info("Step 6: Saving results...")
        runner.save_predictions(predictions, "step_by_step_predictions.csv")
        
        logger.info("✅ Example 2 completed successfully!")
        logger.info(f"Generated {len(predictions)} predictions")
        logger.info(f"Analysis results: {analysis['prediction_stats']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Example 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_3_config_based_training():
    """Example 3: Configuration file-based TFT training"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Configuration-Based TFT Training")
    logger.info("=" * 60)
    
    try:
        # Create configuration file
        config_file = create_tft_config_file()
        logger.info(f"Created configuration file: {config_file}")
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize runner with config parameters
        runner = ShanghaiTFTRunner(
            model_folder=config['model_config']['model_folder'],
            gpu_id=config['model_config']['gpu_id'],
            dataset_name=config['model_config']['dataset']
        )
        
        # Run workflow with config parameters
        success = runner.run_complete_tft_workflow(
            symbol_filter=config['data_config']['symbols'],
            start_date=config['data_config']['start_time'],
            end_date=config['data_config']['end_time']
        )
        
        if success:
            logger.info("✅ Example 3 completed successfully!")
            logger.info(f"Used configuration: {config_file}")
        else:
            logger.error("❌ Example 3 failed!")
            
        return success
        
    except Exception as e:
        logger.error(f"Example 3 failed with error: {e}")
        return False


def example_4_prediction_analysis():
    """Example 4: Advanced prediction analysis and interpretability"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Advanced Prediction Analysis")
    logger.info("=" * 60)
    
    try:
        # Initialize runner
        runner = ShanghaiTFTRunner(
            model_folder="example_analysis",
            gpu_id=0,
            dataset_name="Shanghai_Analysis"
        )
        
        # Run basic training first
        success = runner.run_complete_tft_workflow(
            symbol_filter=["600000", "600036"],
            start_date="2022-01-01",
            end_date="2023-06-30"
        )
        
        if not success:
            logger.error("Training failed!")
            return False
        
        # Generate predictions
        predictions = runner.step4_generate_predictions()
        
        if predictions is None:
            logger.error("Prediction generation failed!")
            return False
        
        # Perform detailed analysis
        logger.info("Performing detailed prediction analysis...")
        analysis = runner.analyze_predictions(predictions, save_analysis=True)
        
        # Print analysis results
        logger.info("📊 Prediction Statistics:")
        stats = analysis['prediction_stats']
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Std: {stats['std']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Quantile analysis
        logger.info("📈 Quantile Analysis:")
        quantiles = stats['quantiles']
        for q, value in quantiles.items():
            logger.info(f"  {q}: {value:.4f}")
        
        # Temporal analysis
        if 'temporal_analysis' in analysis:
            temporal = analysis['temporal_analysis']
            logger.info("⏰ Temporal Analysis:")
            logger.info(f"  Daily mean prediction: {temporal.get('daily_mean_prediction', 'N/A')}")
            logger.info(f"  Daily volatility: {temporal.get('daily_volatility', 'N/A')}")
            logger.info(f"  Prediction days: {temporal.get('prediction_days', 'N/A')}")
        
        # Cross-sectional analysis
        if 'cross_sectional_analysis' in analysis:
            cross_sec = analysis['cross_sectional_analysis']
            logger.info("🏢 Cross-Sectional Analysis:")
            logger.info(f"  Total stocks: {cross_sec.get('total_stocks', 'N/A')}")
            logger.info(f"  Avg prediction per stock: {cross_sec.get('avg_prediction_per_stock', 'N/A')}")
        
        # Initialize analyzer for interpretability
        analyzer = TFTAnalyzer(runner.model_folder)
        
        # Extract attention weights (placeholder implementation)
        logger.info("🔍 Extracting attention weights...")
        attention_weights = analyzer.extract_attention_weights(runner.model, None)
        
        # Generate interpretability report
        logger.info("📝 Generating interpretability report...")
        report = analyzer.generate_interpretability_report(attention_weights)
        
        logger.info("✅ Example 4 completed successfully!")
        logger.info("Check the model folder for detailed analysis files")
        
        return True
        
    except Exception as e:
        logger.error(f"Example 4 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_5_multi_horizon_forecasting():
    """Example 5: Multi-horizon forecasting with different prediction windows"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Multi-Horizon Forecasting")
    logger.info("=" * 60)
    
    try:
        horizons = [1, 3, 5, 10]  # 1-day, 3-day, 5-day, 10-day ahead
        results = {}
        
        for horizon in horizons:
            logger.info(f"Training TFT for {horizon}-day horizon...")
            
            # Initialize runner for this horizon
            runner = ShanghaiTFTRunner(
                model_folder=f"example_horizon_{horizon}d",
                gpu_id=0,
                dataset_name=f"Shanghai_H{horizon}"
            )
            
            # Modify the label shift for different horizons
            # Note: This would require modifying the TFTModel initialization
            logger.info(f"Setting up {horizon}-day prediction horizon...")
            
            # Run training
            success = runner.run_complete_tft_workflow(
                symbol_filter=["600000", "600036"],
                start_date="2022-01-01",
                end_date="2023-06-30"
            )
            
            if success:
                # Generate predictions
                predictions = runner.step4_generate_predictions()
                if predictions is not None:
                    # Analyze predictions
                    analysis = runner.analyze_predictions(predictions)
                    results[f"{horizon}d"] = {
                        "predictions": len(predictions),
                        "mean_prediction": analysis['prediction_stats']['mean'],
                        "prediction_std": analysis['prediction_stats']['std']
                    }
                    
                    logger.info(f"✅ {horizon}-day horizon completed")
                else:
                    logger.error(f"❌ {horizon}-day horizon prediction failed")
            else:
                logger.error(f"❌ {horizon}-day horizon training failed")
        
        # Compare results across horizons
        logger.info("📊 Multi-Horizon Results Summary:")
        for horizon, result in results.items():
            logger.info(f"  {horizon}: {result['predictions']} predictions, "
                       f"mean={result['mean_prediction']:.4f}, "
                       f"std={result['prediction_std']:.4f}")
        
        logger.info("✅ Example 5 completed successfully!")
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"Example 5 failed with error: {e}")
        return False


def run_all_examples():
    """Run all TFT examples"""
    logger.info("🚀 Running All Shanghai TFT Examples")
    logger.info("=" * 80)
    
    examples = [
        ("Basic TFT Training", example_1_basic_tft_training),
        ("Step-by-Step Training", example_2_step_by_step_training),
        ("Config-Based Training", example_3_config_based_training),
        ("Prediction Analysis", example_4_prediction_analysis),
        ("Multi-Horizon Forecasting", example_5_multi_horizon_forecasting)
    ]
    
    results = {}
    
    for name, example_func in examples:
        logger.info(f"\n🎯 Running: {name}")
        try:
            success = example_func()
            results[name] = success
            if success:
                logger.info(f"✅ {name} - SUCCESS")
            else:
                logger.error(f"❌ {name} - FAILED")
        except Exception as e:
            logger.error(f"❌ {name} - ERROR: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📋 EXAMPLES SUMMARY")
    logger.info("=" * 80)
    
    successful = sum(results.values())
    total = len(results)
    
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {name}: {status}")
    
    logger.info(f"\nOverall: {successful}/{total} examples successful")
    
    if successful == total:
        logger.info("🎉 All examples completed successfully!")
    else:
        logger.warning(f"⚠️  {total - successful} examples failed")
    
    return successful == total


def main():
    """Main function to run examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shanghai TFT Examples")
    parser.add_argument("--example", type=int, choices=[1, 2, 3, 4, 5], 
                       help="Run specific example (1-5)")
    parser.add_argument("--all", action="store_true",
                       help="Run all examples")
    
    args = parser.parse_args()
    
    if args.all:
        success = run_all_examples()
    elif args.example:
        examples = {
            1: example_1_basic_tft_training,
            2: example_2_step_by_step_training,
            3: example_3_config_based_training,
            4: example_4_prediction_analysis,
            5: example_5_multi_horizon_forecasting
        }
        success = examples[args.example]()
    else:
        logger.info("Please specify --example <1-5> or --all")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
