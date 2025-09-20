"""
Interpretability utilities for TFT financial forecasting model.
Provides attention analysis, feature importance, and counterfactual analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive plots disabled.")

try:
    # Try to import CUDA backend for interpretability
    import tft_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

from .tft_model import TemporalFusionTransformer


class TFTInterpretability:
    """
    Comprehensive interpretability analysis for TFT models.
    Provides attention visualization, feature importance, and counterfactual analysis.
    """
    
    def __init__(self, model: TemporalFusionTransformer, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or [f'feature_{i}' for i in range(model.config['input_size'])]
        self.device = next(model.parameters()).device
        
        # Cache for storing analysis results
        self.analysis_cache = {}
    
    def extract_attention_weights(self, inputs: Dict[str, torch.Tensor], 
                                use_cuda: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from the model.
        
        Args:
            inputs: Model inputs
            use_cuda: Whether to use CUDA kernels
            
        Returns:
            Dictionary containing attention weights and related information
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs, use_cuda=use_cuda)
            
            attention_weights = outputs['interpretability']['attention_weights']
            historical_selection = outputs['interpretability']['historical_variable_selection']
            static_weights = outputs['interpretability']['static_weights']
            
            # Additional processing for CUDA kernels
            if use_cuda and CUDA_AVAILABLE:
                try:
                    # Use CUDA attention aggregation kernel
                    batch_size, seq_len, num_heads, _ = attention_weights.shape
                    temporal_importance = torch.zeros(batch_size, seq_len, device=self.device)
                    
                    tft_cuda.attention_aggregate(
                        attention_weights.contiguous(),
                        temporal_importance,
                        batch_size, seq_len, num_heads
                    )
                    
                except Exception as e:
                    print(f"CUDA attention aggregation failed: {e}")
                    # Fallback to PyTorch
                    temporal_importance = self._compute_temporal_importance_pytorch(attention_weights)
            else:
                temporal_importance = self._compute_temporal_importance_pytorch(attention_weights)
        
        return {
            'attention_weights': attention_weights,
            'temporal_importance': temporal_importance,
            'historical_selection': historical_selection,
            'static_weights': static_weights,
            'hidden_states': outputs.get('hidden_states')
        }
    
    def _compute_temporal_importance_pytorch(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback for temporal importance computation."""
        # attention_weights: (batch_size, seq_len, num_heads, seq_len)
        # Sum across heads and normalize by causal mask
        batch_size, seq_len, num_heads, _ = attention_weights.shape
        
        temporal_importance = torch.zeros(batch_size, seq_len, device=attention_weights.device)
        
        for t in range(seq_len):
            # Sum attention weights for position t across heads and valid time steps
            valid_attention = attention_weights[:, t, :, :t+1]  # Causal mask
            importance = valid_attention.sum(dim=(1, 2)) / (num_heads * (t + 1))
            temporal_importance[:, t] = importance
        
        return temporal_importance
    
    def analyze_feature_importance(self, inputs: Dict[str, torch.Tensor], 
                                 method: str = 'gradient') -> Dict[str, torch.Tensor]:
        """
        Analyze feature importance using various methods.
        
        Args:
            inputs: Model inputs
            method: 'gradient', 'integrated_gradients', or 'permutation'
            
        Returns:
            Feature importance scores
        """
        if method == 'gradient':
            return self._gradient_based_importance(inputs)
        elif method == 'integrated_gradients':
            return self._integrated_gradients_importance(inputs)
        elif method == 'permutation':
            return self._permutation_importance(inputs)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _gradient_based_importance(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute gradient-based feature importance."""
        self.model.eval()
        
        # Enable gradients for inputs
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.requires_grad_(True)
        
        outputs = self.model(inputs)
        predictions = outputs['predictions']
        
        # Use median quantile prediction for gradient computation
        if isinstance(predictions, dict):
            pred_tensor = next(iter(predictions.values()))
        else:
            pred_tensor = predictions
        
        # Compute gradients with respect to median quantile
        median_idx = pred_tensor.size(-1) // 2
        median_pred = pred_tensor[:, median_idx].sum()
        
        gradients = torch.autograd.grad(
            median_pred, 
            inputs['historical_features'],
            retain_graph=True,
            create_graph=False
        )[0]
        
        # Compute importance as gradient magnitude
        importance = torch.abs(gradients).mean(dim=(0, 1))  # Average over batch and time
        
        return {
            'feature_importance': importance,
            'gradients': gradients
        }
    
    def _integrated_gradients_importance(self, inputs: Dict[str, torch.Tensor], 
                                       steps: int = 50) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients feature importance."""
        baseline = {key: torch.zeros_like(value) for key, value in inputs.items()}
        
        integrated_grads = torch.zeros_like(inputs['historical_features'])
        
        for i in range(steps):
            alpha = i / steps
            interpolated_inputs = {}
            
            for key in inputs:
                interpolated_inputs[key] = baseline[key] + alpha * (inputs[key] - baseline[key])
                if torch.is_tensor(interpolated_inputs[key]):
                    interpolated_inputs[key] = interpolated_inputs[key].requires_grad_(True)
            
            outputs = self.model(interpolated_inputs)
            predictions = outputs['predictions']
            
            if isinstance(predictions, dict):
                pred_tensor = next(iter(predictions.values()))
            else:
                pred_tensor = predictions
            
            median_idx = pred_tensor.size(-1) // 2
            median_pred = pred_tensor[:, median_idx].sum()
            
            grads = torch.autograd.grad(
                median_pred,
                interpolated_inputs['historical_features'],
                retain_graph=True,
                create_graph=False
            )[0]
            
            integrated_grads += grads / steps
        
        # Scale by input difference
        scaled_grads = integrated_grads * (inputs['historical_features'] - baseline['historical_features'])
        importance = torch.abs(scaled_grads).mean(dim=(0, 1))
        
        return {
            'feature_importance': importance,
            'integrated_gradients': integrated_grads
        }
    
    def _permutation_importance(self, inputs: Dict[str, torch.Tensor], 
                              n_permutations: int = 10) -> Dict[str, torch.Tensor]:
        """Compute permutation-based feature importance."""
        self.model.eval()
        
        with torch.no_grad():
            # Get baseline prediction
            baseline_outputs = self.model(inputs)
            baseline_predictions = baseline_outputs['predictions']
            
            if isinstance(baseline_predictions, dict):
                baseline_pred = next(iter(baseline_predictions.values()))
            else:
                baseline_pred = baseline_predictions
            
            baseline_score = baseline_pred[:, baseline_pred.size(-1) // 2].mean()
            
            # Initialize importance scores
            num_features = inputs['historical_features'].size(-1)
            importance_scores = torch.zeros(num_features)
            
            # Permute each feature
            for feature_idx in range(num_features):
                feature_importance = 0
                
                for _ in range(n_permutations):
                    # Create permuted inputs
                    permuted_inputs = {key: value.clone() for key, value in inputs.items()}
                    
                    # Permute the feature across batch dimension
                    perm_idx = torch.randperm(inputs['historical_features'].size(0))
                    permuted_inputs['historical_features'][:, :, feature_idx] = \
                        inputs['historical_features'][perm_idx, :, feature_idx]
                    
                    # Get prediction with permuted feature
                    outputs = self.model(permuted_inputs)
                    predictions = outputs['predictions']
                    
                    if isinstance(predictions, dict):
                        pred_tensor = next(iter(predictions.values()))
                    else:
                        pred_tensor = predictions
                    
                    permuted_score = pred_tensor[:, pred_tensor.size(-1) // 2].mean()
                    
                    # Importance as difference in performance
                    feature_importance += torch.abs(baseline_score - permuted_score)
                
                importance_scores[feature_idx] = feature_importance / n_permutations
        
        return {'feature_importance': importance_scores}
    
    def generate_attention_heatmap(self, inputs: Dict[str, torch.Tensor], 
                                 sample_idx: int = 0, save_path: Optional[str] = None) -> plt.Figure:
        """Generate attention heatmap visualization."""
        attention_data = self.extract_attention_weights(inputs)
        attention_weights = attention_data['attention_weights'][sample_idx]  # (seq_len, num_heads, seq_len)
        
        # Average across heads
        avg_attention = attention_weights.mean(dim=1).cpu().numpy()  # (seq_len, seq_len)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            avg_attention,
            annot=False,
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(f'Temporal Attention Heatmap (Sample {sample_idx})')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Add temporal labels if available
        if hasattr(self, 'time_labels'):
            ax.set_xticklabels(self.time_labels[-avg_attention.shape[1]:], rotation=45)
            ax.set_yticklabels(self.time_labels[-avg_attention.shape[0]:])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, inputs: Dict[str, torch.Tensor], 
                              method: str = 'gradient', top_k: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance scores."""
        importance_data = self.analyze_feature_importance(inputs, method)
        importance_scores = importance_data['feature_importance'].cpu().numpy()
        
        # Get top-k features
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        top_scores = importance_scores[top_indices]
        top_names = [self.feature_names[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(top_scores)), top_scores)
        ax.set_yticks(range(len(top_scores)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_k} Feature Importance ({method.replace("_", " ").title()})')
        
        # Color bars by score
        colors = plt.cm.viridis(top_scores / top_scores.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_temporal_importance(self, inputs: Dict[str, torch.Tensor], 
                               sample_idx: int = 0, save_path: Optional[str] = None) -> plt.Figure:
        """Plot temporal importance over sequence."""
        attention_data = self.extract_attention_weights(inputs)
        temporal_importance = attention_data['temporal_importance'][sample_idx].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        time_steps = range(len(temporal_importance))
        ax.plot(time_steps, temporal_importance, linewidth=2, marker='o', markersize=4)
        ax.fill_between(time_steps, temporal_importance, alpha=0.3)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temporal Importance')
        ax.set_title(f'Temporal Importance Over Sequence (Sample {sample_idx})')
        ax.grid(True, alpha=0.3)
        
        # Highlight most important time steps
        top_k_steps = 5
        top_indices = np.argsort(temporal_importance)[-top_k_steps:]
        ax.scatter(top_indices, temporal_importance[top_indices], 
                  color='red', s=100, zorder=5, label=f'Top {top_k_steps} Steps')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def counterfactual_analysis(self, inputs: Dict[str, torch.Tensor], 
                              feature_perturbations: Dict[str, float],
                              sample_idx: int = 0) -> Dict[str, Any]:
        """
        Perform counterfactual analysis by perturbing features.
        
        Args:
            inputs: Original model inputs
            feature_perturbations: Dict mapping feature names to perturbation multipliers
            sample_idx: Sample to analyze
            
        Returns:
            Dictionary containing original and counterfactual predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get original prediction
            original_outputs = self.model(inputs)
            original_predictions = original_outputs['predictions']
            
            results = {
                'original_predictions': original_predictions,
                'counterfactual_predictions': {},
                'prediction_changes': {}
            }
            
            # Apply each perturbation
            for feature_name, multiplier in feature_perturbations.items():
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    
                    # Create perturbed inputs
                    perturbed_inputs = {key: value.clone() for key, value in inputs.items()}
                    perturbed_inputs['historical_features'][sample_idx, :, feature_idx] *= multiplier
                    
                    # Get counterfactual prediction
                    cf_outputs = self.model(perturbed_inputs)
                    cf_predictions = cf_outputs['predictions']
                    
                    results['counterfactual_predictions'][feature_name] = cf_predictions
                    
                    # Compute prediction changes
                    if isinstance(original_predictions, dict):
                        changes = {}
                        for horizon, orig_pred in original_predictions.items():
                            cf_pred = cf_predictions[horizon]
                            change = cf_pred[sample_idx] - orig_pred[sample_idx]
                            changes[horizon] = change.cpu().numpy()
                        results['prediction_changes'][feature_name] = changes
                    else:
                        change = cf_predictions[sample_idx] - original_predictions[sample_idx]
                        results['prediction_changes'][feature_name] = change.cpu().numpy()
        
        return results
    
    def export_interpretability_report(self, inputs: Dict[str, torch.Tensor], 
                                     output_dir: str = 'interpretability_report') -> str:
        """
        Generate comprehensive interpretability report.
        
        Args:
            inputs: Model inputs for analysis
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating interpretability report...")
        
        # 1. Attention analysis
        print("  - Attention heatmaps...")
        for i in range(min(3, inputs['historical_features'].size(0))):
            fig = self.generate_attention_heatmap(inputs, sample_idx=i)
            fig.savefig(output_path / f'attention_heatmap_sample_{i}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # 2. Feature importance
        print("  - Feature importance...")
        for method in ['gradient', 'integrated_gradients']:
            try:
                fig = self.plot_feature_importance(inputs, method=method)
                fig.savefig(output_path / f'feature_importance_{method}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"    Warning: {method} failed: {e}")
        
        # 3. Temporal importance
        print("  - Temporal importance...")
        for i in range(min(3, inputs['historical_features'].size(0))):
            fig = self.plot_temporal_importance(inputs, sample_idx=i)
            fig.savefig(output_path / f'temporal_importance_sample_{i}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # 4. Counterfactual analysis
        print("  - Counterfactual analysis...")
        perturbations = {
            self.feature_names[0]: 1.1,  # 10% increase
            self.feature_names[1]: 0.9,  # 10% decrease
        }
        
        cf_results = self.counterfactual_analysis(inputs, perturbations)
        
        # Save counterfactual results
        with open(output_path / 'counterfactual_results.json', 'w') as f:
            # Convert tensors to lists for JSON serialization
            serializable_results = {}
            for key, value in cf_results.items():
                if isinstance(value, torch.Tensor):
                    serializable_results[key] = value.cpu().numpy().tolist()
                elif isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        # 5. Generate summary report
        print("  - Summary report...")
        self._generate_summary_report(inputs, output_path)
        
        print(f"Interpretability report saved to: {output_path}")
        return str(output_path)
    
    def _generate_summary_report(self, inputs: Dict[str, torch.Tensor], output_path: Path):
        """Generate a summary HTML report."""
        # Extract key statistics
        attention_data = self.extract_attention_weights(inputs)
        importance_data = self.analyze_feature_importance(inputs)
        
        # Create summary statistics
        temporal_importance = attention_data['temporal_importance'].mean(dim=0).cpu().numpy()
        feature_importance = importance_data['feature_importance'].cpu().numpy()
        
        # Top features and time steps
        top_features = np.argsort(feature_importance)[-10:][::-1]
        top_timesteps = np.argsort(temporal_importance)[-10:][::-1]
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TFT Interpretability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>TFT Interpretability Report</h1>
            
            <div class="section">
                <h2>Model Summary</h2>
                <div class="metric">Input Features: {len(self.feature_names)}</div>
                <div class="metric">Sequence Length: {inputs['historical_features'].size(1)}</div>
                <div class="metric">Batch Size: {inputs['historical_features'].size(0)}</div>
                <div class="metric">Quantile Levels: {self.model.config['quantile_levels']}</div>
            </div>
            
            <div class="section">
                <h2>Top 10 Most Important Features</h2>
                <table>
                    <tr><th>Rank</th><th>Feature</th><th>Importance Score</th></tr>
                    {''.join([f'<tr><td>{i+1}</td><td>{self.feature_names[idx]}</td><td>{feature_importance[idx]:.4f}</td></tr>' 
                             for i, idx in enumerate(top_features)])}
                </table>
            </div>
            
            <div class="section">
                <h2>Top 10 Most Important Time Steps</h2>
                <table>
                    <tr><th>Rank</th><th>Time Step</th><th>Temporal Importance</th></tr>
                    {''.join([f'<tr><td>{i+1}</td><td>{idx}</td><td>{temporal_importance[idx]:.4f}</td></tr>' 
                             for i, idx in enumerate(top_timesteps)])}
                </table>
            </div>
            
            <div class="section">
                <h2>Generated Visualizations</h2>
                <ul>
                    <li>Attention heatmaps (attention_heatmap_sample_*.png)</li>
                    <li>Feature importance plots (feature_importance_*.png)</li>
                    <li>Temporal importance plots (temporal_importance_sample_*.png)</li>
                    <li>Counterfactual analysis results (counterfactual_results.json)</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path / 'summary_report.html', 'w') as f:
            f.write(html_content)


def main():
    """Example usage of TFT interpretability."""
    from .tft_model import TemporalFusionTransformer, create_tft_config
    
    print("TFT Interpretability Example")
    print("=" * 30)
    
    # Create sample model
    config = create_tft_config(input_size=32)
    model = TemporalFusionTransformer(config)
    model.eval()
    
    # Create sample inputs
    batch_size, seq_len, input_size = 4, 100, 32
    sample_inputs = {
        'historical_features': torch.randn(batch_size, seq_len, input_size),
        'static_features': torch.randn(batch_size, 10)
    }
    
    # Create interpretability analyzer
    feature_names = [f'feature_{i}' for i in range(input_size)]
    interpreter = TFTInterpretability(model, feature_names)
    
    # Extract attention weights
    print("Extracting attention weights...")
    attention_data = interpreter.extract_attention_weights(sample_inputs)
    print(f"Attention weights shape: {attention_data['attention_weights'].shape}")
    print(f"Temporal importance shape: {attention_data['temporal_importance'].shape}")
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    importance_data = interpreter.analyze_feature_importance(sample_inputs)
    print(f"Feature importance shape: {importance_data['feature_importance'].shape}")
    
    # Generate visualizations
    print("Generating visualizations...")
    fig1 = interpreter.generate_attention_heatmap(sample_inputs)
    fig2 = interpreter.plot_feature_importance(sample_inputs)
    fig3 = interpreter.plot_temporal_importance(sample_inputs)
    
    plt.show()
    
    # Counterfactual analysis
    print("Performing counterfactual analysis...")
    perturbations = {'feature_0': 1.2, 'feature_1': 0.8}
    cf_results = interpreter.counterfactual_analysis(sample_inputs, perturbations)
    
    print("Interpretability analysis complete!")


if __name__ == "__main__":
    main()