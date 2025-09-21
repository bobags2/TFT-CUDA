#!/usr/bin/env python3
"""
Comprehensive TFT Model Evaluation and Analysis
===============================================
Production-grade evaluation metrics for the trained TFT model.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "python")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import model
from tft_model import TemporalFusionTransformer

class TFTEvaluator:
    """Comprehensive evaluation suite for TFT model."""
    
    def __init__(self, checkpoint_path: str = "checkpoints/tft_best.pth"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.metrics = {}
        self.predictions = {}
        
    def load_model(self) -> bool:
        """Load the trained TFT model from checkpoint."""
        try:
            print(f"📦 Loading model from {self.checkpoint_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Extract config (from train.py configuration)
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                # Reconstruct config from train.py
                self.config = {
                    'input_size': 134,  # Full feature set from train.py
                    'output_size': 3,
                    'hidden_size': 256,
                    'num_heads': 4,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'dropout_rate': 0.3,
                    'sequence_length': 128,
                    'quantile_levels': [0.5],
                    'prediction_horizon': [1, 5, 10],
                    'num_historical_features': 134,
                    'num_future_features': 134,
                    'static_input_size': 10
                }
            
            # Initialize model
            self.model = TemporalFusionTransformer(self.config)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Count parameters
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {param_count:,}")
            print(f"   Device: {self.device}")
            print(f"   Config: Hidden={self.config.get('hidden_size')}, Heads={self.config.get('num_heads')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Load validation and test data."""
        print("\n📊 Loading evaluation data...")
        
        # Try to load full feature dataset first (what the model was trained on)
        try:
            X_val = np.load('data/X_val.npy')
            y_val = np.load('data/y_val.npy')
            print(f"   Loaded full validation data: X_val{X_val.shape}, y_val{y_val.shape}")
            
            # Try to load test data
            X_test, y_test = None, None
            if Path('data/X_test.npy').exists():
                X_test = np.load('data/X_test.npy')
                y_test = np.load('data/y_test.npy')
                print(f"   Loaded full test data: X_test{X_test.shape}, y_test{y_test.shape}")
            
        except Exception as e:
            print(f"⚠️  Full dataset not found, trying temporal data...")
            X_val = np.load('data/X_val_temporal.npy')
            y_val = np.load('data/y_val_temporal.npy')
            X_test = np.load('data/X_test_temporal.npy') if Path('data/X_test_temporal.npy').exists() else None
            y_test = np.load('data/y_test_temporal.npy') if Path('data/y_test_temporal.npy').exists() else None
            print(f"   Loaded temporal data: X_val{X_val.shape}, y_val{y_val.shape}")
        
        return X_val, y_val, X_test, y_test
    
    def evaluate_batch(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Evaluate a single batch of data."""
        # Ensure model is loaded
        if self.model is None:
            if not self.load_model() or self.model is None:
                raise RuntimeError("TFT model is not loaded. Call load_model() before evaluation.")
        self.model.eval()

        batch_size, seq_len, num_features = X_batch.shape
        
        # Convert to TFT input format
        tft_inputs = {
            'historical_features': X_batch,
            'future_features': X_batch[:, -1:, :],  # Last timestep
            'static_features': torch.zeros(batch_size, 10, device=self.device)
        }
        
        # Clamp inputs for stability
        tft_inputs['historical_features'] = torch.clamp(tft_inputs['historical_features'], min=-50, max=50)
        tft_inputs['future_features'] = torch.clamp(tft_inputs['future_features'], min=-50, max=50)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(tft_inputs)
        
        # Extract predictions for all horizons
        all_predictions = []
        for horizon in [1, 5, 10]:
            if f'horizon_{horizon}' in outputs['predictions']:
                all_predictions.append(outputs['predictions'][f'horizon_{horizon}'])
        
        if all_predictions:
            pred = torch.cat(all_predictions, dim=-1)  # Combine all horizons
        else:
            pred = list(outputs['predictions'].values())[0]
            if pred.dim() == 2 and pred.shape[1] == 1:
                pred = pred.repeat(1, 9)  # Expand to match 9 targets
        
        return pred, outputs.get('interpretability', {})
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                       prefix: str = "") -> Dict:
        """Compute comprehensive evaluation metrics."""
        # Move to CPU for numpy operations
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        metrics = {}
        
        # Handle different target dimensions
        if len(target_np.shape) == 1:
            target_np = target_np.reshape(-1, 1)
        if len(pred_np.shape) == 1:
            pred_np = pred_np.reshape(-1, 1)
        
        # Match dimensions
        min_dim = min(pred_np.shape[1], target_np.shape[1])
        pred_np = pred_np[:, :min_dim]
        target_np = target_np[:, :min_dim]
        
        # Basic regression metrics
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        rmse = np.sqrt(mse)
        
        # Avoid division by zero
        target_std = np.std(target_np)
        if target_std > 1e-8:
            nrmse = rmse / target_std
        else:
            nrmse = float('inf')
        
        # R-squared
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Direction accuracy (for returns)
        if min_dim >= 1:  # First column is returns
            pred_direction = np.sign(pred_np[:, 0])
            target_direction = np.sign(target_np[:, 0])
            direction_accuracy = np.mean(pred_direction == target_direction)
        else:
            direction_accuracy = 0.0
        
        # Trading metrics (if returns available)
        if min_dim >= 1:
            returns_pred = pred_np[:, 0]
            returns_target = target_np[:, 0]
            
            # Sharpe ratio (annualized for 10-min bars: ~39,000 bars/year)
            if np.std(returns_pred) > 1e-8:
                sharpe_pred = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(39000)
            else:
                sharpe_pred = 0.0
            
            if np.std(returns_target) > 1e-8:
                sharpe_target = np.mean(returns_target) / np.std(returns_target) * np.sqrt(39000)
            else:
                sharpe_target = 0.0
            
            # Max drawdown
            cum_returns_pred = np.cumprod(1 + returns_pred)
            cum_returns_target = np.cumprod(1 + returns_target)
            
            running_max_pred = np.maximum.accumulate(cum_returns_pred)
            drawdown_pred = (cum_returns_pred - running_max_pred) / running_max_pred
            max_drawdown_pred = np.min(drawdown_pred)
            
            running_max_target = np.maximum.accumulate(cum_returns_target)
            drawdown_target = (cum_returns_target - running_max_target) / running_max_target
            max_drawdown_target = np.min(drawdown_target)
        else:
            sharpe_pred = sharpe_target = 0.0
            max_drawdown_pred = max_drawdown_target = 0.0
        
        # Store metrics
        metrics[f'{prefix}mse'] = float(mse)
        metrics[f'{prefix}mae'] = float(mae)
        metrics[f'{prefix}rmse'] = float(rmse)
        metrics[f'{prefix}nrmse'] = float(nrmse)
        metrics[f'{prefix}r2'] = float(r2)
        metrics[f'{prefix}direction_accuracy'] = float(direction_accuracy)
        metrics[f'{prefix}sharpe_pred'] = float(sharpe_pred)
        metrics[f'{prefix}sharpe_target'] = float(sharpe_target)
        metrics[f'{prefix}max_drawdown_pred'] = float(max_drawdown_pred)
        metrics[f'{prefix}max_drawdown_target'] = float(max_drawdown_target)
        
        return metrics
    
    def evaluate_dataset(self, X: np.ndarray, y: np.ndarray, 
                        dataset_name: str = "validation") -> Dict:
        """Evaluate entire dataset with batching."""
        print(f"\n🔍 Evaluating {dataset_name} set...")
        
        # Ensure model is loaded
        if self.model is None:
            if not self.load_model() or self.model is None:
                print("❌ Model not loaded; aborting dataset evaluation.")
                return {}

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Handle target dimensions
        if len(y_tensor.shape) > 1 and y_tensor.shape[1] > 9:
            y_tensor = y_tensor[:, :9]  # Use first 9 targets
        elif len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        
        # Evaluate in batches
        batch_size = 128
        all_predictions = []
        all_targets = []
        
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"   Processing {len(dataloader)} batches...")
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            try:
                pred_batch, _ = self.evaluate_batch(X_batch, y_batch)
                all_predictions.append(pred_batch)
                all_targets.append(y_batch)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Processed {batch_idx + 1}/{len(dataloader)} batches")
                    
            except Exception as e:
                print(f"   ⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        if not all_predictions:
            print(f"   ❌ No successful predictions for {dataset_name}")
            return {}
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_targets, prefix=f"{dataset_name}_")
        
        # Store predictions for analysis
        self.predictions[dataset_name] = {
            'predictions': all_predictions.cpu().numpy(),
            'targets': all_targets.cpu().numpy()
        }
        
        return metrics
    
    def analyze_predictions(self, dataset_name: str = "validation"):
        """Detailed analysis of predictions."""
        if dataset_name not in self.predictions:
            print(f"⚠️ No predictions available for {dataset_name}")
            return
        
        pred = self.predictions[dataset_name]['predictions']
        target = self.predictions[dataset_name]['targets']
        
        print(f"\n📈 Prediction Analysis for {dataset_name}:")
        print("=" * 60)
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(f"  Predictions shape: {pred.shape}")
        print(f"  Targets shape: {target.shape}")
        print(f"  Prediction range: [{np.min(pred):.4f}, {np.max(pred):.4f}]")
        print(f"  Target range: [{np.min(target):.4f}, {np.max(target):.4f}]")
        print(f"  Prediction mean: {np.mean(pred):.4f} ± {np.std(pred):.4f}")
        print(f"  Target mean: {np.mean(target):.4f} ± {np.std(target):.4f}")
        
        # Per-horizon analysis (if we have 9 targets: 3 per horizon)
        if pred.shape[1] >= 9:
            print("\nPer-Horizon Analysis:")
            horizons = [1, 5, 10]
            for i, horizon in enumerate(horizons):
                start_idx = i * 3
                end_idx = start_idx + 3
                
                horizon_pred = pred[:, start_idx:end_idx]
                horizon_target = target[:, start_idx:end_idx]
                
                # Return predictions (first column of each horizon)
                if horizon_pred.shape[1] > 0:
                    return_pred = horizon_pred[:, 0]
                    return_target = horizon_target[:, 0]
                    
                    mse = np.mean((return_pred - return_target) ** 2)
                    corr = np.corrcoef(return_pred, return_target)[0, 1]
                    direction_acc = np.mean(np.sign(return_pred) == np.sign(return_target))
                    
                    print(f"\n  Horizon {horizon}:")
                    print(f"    Return MSE: {mse:.6f}")
                    print(f"    Return Correlation: {corr:.4f}")
                    print(f"    Direction Accuracy: {direction_acc:.2%}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n📊 Creating visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Use validation predictions
        if 'validation' not in self.predictions:
            print("⚠️ No validation predictions available for visualization")
            return
        
        pred = self.predictions['validation']['predictions']
        target = self.predictions['validation']['targets']
        
        # 1. Predictions vs Targets scatter plot
        ax1 = plt.subplot(2, 3, 1)
        if pred.shape[1] > 0:
            ax1.scatter(target[:, 0], pred[:, 0], alpha=0.5, s=1)
            ax1.plot([target[:, 0].min(), target[:, 0].max()], 
                    [target[:, 0].min(), target[:, 0].max()], 
                    'r--', lw=2, label='Perfect Prediction')
            ax1.set_xlabel('Actual Returns')
            ax1.set_ylabel('Predicted Returns')
            ax1.set_title('Horizon 1: Predictions vs Actuals')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Residuals histogram
        ax2 = plt.subplot(2, 3, 2)
        if pred.shape[1] > 0:
            residuals = pred[:, 0] - target[:, 0]
            ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='r', linestyle='--', label=f'Mean: {np.mean(residuals):.4f}')
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Residual Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Time series plot (first 500 points)
        ax3 = plt.subplot(2, 3, 3)
        if pred.shape[1] > 0:
            n_points = min(500, len(pred))
            ax3.plot(target[:n_points, 0], label='Actual', alpha=0.7, linewidth=1)
            ax3.plot(pred[:n_points, 0], label='Predicted', alpha=0.7, linewidth=1)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Returns')
            ax3.set_title('Time Series Comparison (First 500 points)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative returns
        ax4 = plt.subplot(2, 3, 4)
        if pred.shape[1] > 0:
            cum_actual = np.cumprod(1 + target[:, 0])
            cum_pred = np.cumprod(1 + pred[:, 0])
            ax4.plot(cum_actual, label='Actual Strategy', linewidth=2)
            ax4.plot(cum_pred, label='Predicted Strategy', linewidth=2)
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Cumulative Return')
            ax4.set_title('Cumulative Performance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Q-Q plot
        ax5 = plt.subplot(2, 3, 5)
        if pred.shape[1] > 0:
            from scipy import stats
            residuals = pred[:, 0] - target[:, 0]
            stats.probplot(residuals, dist="norm", plot=ax5)
            ax5.set_title('Q-Q Plot of Residuals')
            ax5.grid(True, alpha=0.3)

        # 6. Rolling correlation
        ax6 = plt.subplot(2, 3, 6)
        if pred.shape[1] > 0 and len(pred) > 100:
            window = 100
            rolling_corr = pd.Series(pred[:, 0]).rolling(window).corr(pd.Series(target[:, 0]))
            ax6.plot(rolling_corr)
            ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Correlation')
            ax6.set_title(f'Rolling Correlation (Window={window})')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('TFT Model Evaluation Visualizations', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = Path('evaluation_results')
        save_path.mkdir(exist_ok=True)
        
        fig_path = save_path / f'tft_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        print(f"   Saved visualization to {fig_path}")
        
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\n" + "="*80)
        print("TFT MODEL EVALUATION REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {self.checkpoint_path}")
        
        if self.config:
            print("\nModel Configuration:")
            print(f"  Input Size: {self.config.get('input_size', 'N/A')}")
            print(f"  Hidden Size: {self.config.get('hidden_size', 'N/A')}")
            print(f"  Attention Heads: {self.config.get('num_heads', 'N/A')}")
            print(f"  Prediction Horizons: {self.config.get('prediction_horizon', 'N/A')}")
            print(f"  Dropout Rate: {self.config.get('dropout_rate', 'N/A')}")
        
        print("\n" + "-"*80)
        print("PERFORMANCE METRICS")
        print("-"*80)
        
        # Validation metrics
        if any(k.startswith('validation_') for k in self.metrics):
            print("\n📊 Validation Set Performance:")
            for key, value in sorted(self.metrics.items()):
                if key.startswith('validation_'):
                    metric_name = key.replace('validation_', '').upper()
                    if 'accuracy' in key:
                        print(f"  {metric_name:25s}: {value:.2%}")
                    else:
                        print(f"  {metric_name:25s}: {value:.6f}")
        
        # Test metrics
        if any(k.startswith('test_') for k in self.metrics):
            print("\n📊 Test Set Performance:")
            for key, value in sorted(self.metrics.items()):
                if key.startswith('test_'):
                    metric_name = key.replace('test_', '').upper()
                    if 'accuracy' in key:
                        print(f"  {metric_name:25s}: {value:.2%}")
                    else:
                        print(f"  {metric_name:25s}: {value:.6f}")
        
        # Trading performance summary
        print("\n" + "-"*80)
        print("TRADING PERFORMANCE SUMMARY")
        print("-"*80)
        
        val_dir_acc = self.metrics.get('validation_direction_accuracy', 0)
        val_sharpe = self.metrics.get('validation_sharpe_pred', 0)
        val_dd = self.metrics.get('validation_max_drawdown_pred', 0)
        
        print(f"\n🎯 Key Trading Metrics:")
        print(f"  Direction Accuracy: {val_dir_acc:.2%}")
        print(f"  Sharpe Ratio (Predicted): {val_sharpe:.3f}")
        print(f"  Max Drawdown (Predicted): {val_dd:.2%}")
        
        # Model quality assessment
        print("\n" + "-"*80)
        print("MODEL QUALITY ASSESSMENT")
        print("-"*80)
        
        r2 = self.metrics.get('validation_r2', 0)
        rmse = self.metrics.get('validation_rmse', float('inf'))
        
        if r2 > 0.5:
            quality = "EXCELLENT ⭐⭐⭐⭐⭐"
        elif r2 > 0.3:
            quality = "GOOD ⭐⭐⭐⭐"
        elif r2 > 0.1:
            quality = "MODERATE ⭐⭐⭐"
        elif r2 > 0:
            quality = "POOR ⭐⭐"
        else:
            quality = "INADEQUATE ⭐"
        
        print(f"\n  Overall Quality: {quality}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.6f}")
        
        # Trading viability
        print("\n  Trading Viability:")
        if val_dir_acc > 0.55 and val_sharpe > 1.0:
            print("    ✅ VIABLE for live trading (with proper risk management)")
        elif val_dir_acc > 0.52 and val_sharpe > 0.5:
            print("    ⚠️  MARGINAL - needs further optimization")
        else:
            print("    ❌ NOT RECOMMENDED for live trading yet")
        
        # Save report to file
        report_path = Path('evaluation_results')
        report_path.mkdir(exist_ok=True)
        
        report_file = report_path / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.checkpoint_path),
            'config': self.config,
            'metrics': self.metrics,
            'quality_assessment': {
                'overall_quality': quality,
                'r2_score': r2,
                'rmse': rmse,
                'direction_accuracy': val_dir_acc,
                'sharpe_ratio': val_sharpe,
                'max_drawdown': val_dd
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📁 Report saved to: {report_file}")
        print("="*80)
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n🚀 Starting TFT Model Evaluation")
        print("="*80)
        
        # Load model
        if not self.load_model():
            print("❌ Evaluation aborted - could not load model")
            return
        
        # Load data
        X_val, y_val, X_test, y_test = self.load_data()
        
        # Evaluate validation set
        val_metrics = self.evaluate_dataset(X_val, y_val, "validation")
        self.metrics.update(val_metrics)
        
        # Evaluate test set if available
        if X_test is not None and y_test is not None:
            test_metrics = self.evaluate_dataset(X_test, y_test, "test")
            self.metrics.update(test_metrics)
        
        # Analyze predictions
        self.analyze_predictions("validation")
        if X_test is not None:
            self.analyze_predictions("test")
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        print("\n✅ Evaluation complete!")
        
        return self.metrics


def main():
    """Main evaluation function."""
    evaluator = TFTEvaluator(checkpoint_path="checkpoints/tft_best.pth")
    metrics = evaluator.run_full_evaluation()
    
    # Return metrics for programmatic use
    return metrics


if __name__ == "__main__":
    metrics = main()
