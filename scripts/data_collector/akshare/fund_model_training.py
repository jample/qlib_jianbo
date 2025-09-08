#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fund Model Training Script

This script trains machine learning models on fund data collected through the AkShare workflow.
It includes data preprocessing, model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class FundModelTrainer:
    """Fund model training and evaluation"""
    
    def __init__(self, data_path="./fund_workflow_data/features/fund_features.csv"):
        self.data_path = Path(data_path)
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and prepare the feature data"""
        print("Loading fund feature data...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Feature data not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"✓ Loaded {len(self.df)} samples with {len(self.df.columns)} columns")
        print(f"✓ Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"✓ Funds: {self.df['symbol'].nunique()}")
        
        # Identify feature columns (exclude date, symbol, close, target)
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['date', 'symbol', 'close', 'target']]
        
        print(f"✓ Features: {self.feature_columns}")
        
    def prepare_data(self, test_size=0.2, validation_size=0.1):
        """Prepare data for training"""
        print("\nPreparing data for training...")
        
        # Remove any rows with NaN values
        self.df_clean = self.df.dropna()
        print(f"✓ Clean data: {len(self.df_clean)} samples")
        
        # Prepare features and target
        X = self.df_clean[self.feature_columns].values
        y = self.df_clean['target'].values
        
        # Time-based split (more realistic for time series)
        # Sort by date first
        self.df_clean = self.df_clean.sort_values('date')
        X = self.df_clean[self.feature_columns].values
        y = self.df_clean['target'].values
        
        # Split data
        n_samples = len(X)
        n_train = int(n_samples * (1 - test_size - validation_size))
        n_val = int(n_samples * validation_size)
        
        self.X_train = X[:n_train]
        self.y_train = y[:n_train]
        self.X_val = X[n_train:n_train+n_val]
        self.y_val = y[n_train:n_train+n_val]
        self.X_test = X[n_train+n_val:]
        self.y_test = y[n_train+n_val:]
        
        print(f"✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Validation set: {len(self.X_val)} samples")
        print(f"✓ Test set: {len(self.X_test)} samples")
        
        # Scale features
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✓ Features scaled using RobustScaler")
        
    def train_models(self):
        """Train multiple models"""
        print("\nTraining models...")
        
        # Define models to train
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(self.X_train_scaled, self.y_train)
                
                # Make predictions
                train_pred = model.predict(self.X_train_scaled)
                val_pred = model.predict(self.X_val_scaled)
                test_pred = model.predict(self.X_test_scaled)
                
                # Calculate metrics
                train_mse = mean_squared_error(self.y_train, train_pred)
                val_mse = mean_squared_error(self.y_val, val_pred)
                test_mse = mean_squared_error(self.y_test, test_pred)
                
                train_r2 = r2_score(self.y_train, train_pred)
                val_r2 = r2_score(self.y_val, val_pred)
                test_r2 = r2_score(self.y_test, test_pred)
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'test_r2': test_r2,
                    'train_pred': train_pred,
                    'val_pred': val_pred,
                    'test_pred': test_pred
                }
                
                print(f"  ✓ {name}: Val R² = {val_r2:.4f}, Test R² = {test_r2:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")
                continue
        
        print(f"✓ Trained {len(self.models)} models successfully")
        
    def evaluate_models(self):
        """Evaluate and compare models"""
        print("\nModel Evaluation Results:")
        print("=" * 80)
        
        # Create results DataFrame
        results_data = []
        for name, metrics in self.results.items():
            results_data.append({
                'Model': name,
                'Train R²': metrics['train_r2'],
                'Val R²': metrics['val_r2'],
                'Test R²': metrics['test_r2'],
                'Train MSE': metrics['train_mse'],
                'Val MSE': metrics['val_mse'],
                'Test MSE': metrics['test_mse']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Val R²', ascending=False)
        
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Validation R²: {results_df.iloc[0]['Val R²']:.4f}")
        print(f"   Test R²: {results_df.iloc[0]['Test R²']:.4f}")
        
        return best_model_name, results_df
        
    def feature_importance_analysis(self, model_name=None):
        """Analyze feature importance"""
        if model_name is None:
            # Use best model
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['val_r2'])
            model_name = best_model_name
        
        model = self.models[model_name]
        
        print(f"\nFeature Importance Analysis - {model_name}:")
        print("-" * 50)
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.to_string(index=False, float_format='%.4f'))
            
            return feature_importance
            
        elif hasattr(model, 'coef_'):
            # Linear models
            coefficients = np.abs(model.coef_)
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'abs_coefficient': coefficients
            }).sort_values('abs_coefficient', ascending=False)
            
            print(feature_importance.to_string(index=False, float_format='%.4f'))
            
            return feature_importance
        
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_models(self, output_dir="./fund_models"):
        """Save trained models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving models to {output_dir}...")
        
        # Save scaler
        joblib.dump(self.scaler, output_dir / "scaler.pkl")
        
        # Save models
        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(model, output_dir / filename)
            print(f"✓ Saved {name}")
        
        # Save feature columns
        with open(output_dir / "feature_columns.txt", 'w') as f:
            for col in self.feature_columns:
                f.write(f"{col}\n")
        
        # Save results
        results_df = pd.DataFrame([
            {
                'model': name,
                'val_r2': metrics['val_r2'],
                'test_r2': metrics['test_r2'],
                'val_mse': metrics['val_mse'],
                'test_mse': metrics['test_mse']
            }
            for name, metrics in self.results.items()
        ])
        results_df.to_csv(output_dir / "model_results.csv", index=False)
        
        print(f"✓ Models and results saved to {output_dir}")
    
    def predict_future_returns(self, model_name=None, days_ahead=5):
        """Make predictions for future returns"""
        if model_name is None:
            # Use best model
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['val_r2'])
            model_name = best_model_name
        
        model = self.models[model_name]
        
        print(f"\nMaking predictions with {model_name}...")
        
        # Use the most recent data for prediction
        recent_data = self.df_clean.tail(days_ahead)
        X_recent = recent_data[self.feature_columns].values
        X_recent_scaled = self.scaler.transform(X_recent)
        
        predictions = model.predict(X_recent_scaled)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'date': recent_data['date'],
            'symbol': recent_data['symbol'],
            'current_close': recent_data['close'],
            'predicted_return': predictions,
            'predicted_price': recent_data['close'] * (1 + predictions)
        })
        
        print("Recent Predictions:")
        print(pred_df.to_string(index=False, float_format='%.4f'))
        
        return pred_df


def main():
    """Main training workflow"""
    print("Fund Model Training Workflow")
    print("=" * 50)
    
    # Initialize trainer
    trainer = FundModelTrainer()
    
    # Prepare data
    trainer.prepare_data()
    
    # Train models
    trainer.train_models()
    
    # Evaluate models
    best_model, results_df = trainer.evaluate_models()
    
    # Feature importance analysis
    trainer.feature_importance_analysis()
    
    # Save models
    trainer.save_models()
    
    # Make predictions
    predictions = trainer.predict_future_returns()
    
    print("\n" + "=" * 50)
    print("🎉 Model training completed successfully!")
    print(f"📊 Best model: {best_model}")
    print("💾 Models saved to ./fund_models/")
    print("🔮 Predictions generated for recent data")


if __name__ == "__main__":
    main()
