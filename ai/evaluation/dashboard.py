"""
Evaluation metrics dashboard for Phase 3 models.
Provides interactive visualization of model performance and trading metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


class ModelEvaluationDashboard:
    """
    Interactive dashboard for visualizing model evaluation metrics.
    Generates reports and visualizations for model performance analysis.
    """
    
    def __init__(self, training_results: Optional[Dict[str, Dict[str, Any]]] = None):
        self.training_results = training_results or {}
        self.report_data: Dict[str, Any] = {}
        self.charts: List[Dict[str, Any]] = []
    
    def add_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Add training results to the dashboard."""
        self.training_results.update(results)
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all model performances."""
        rows = []
        
        for horizon, models in self.training_results.items():
            for model_type, metrics in models.items():
                if isinstance(metrics, dict):
                    rows.append({
                        'Horizon': horizon,
                        'Model': model_type,
                        'Accuracy': metrics.get('accuracy', 0),
                        'ROC-AUC': metrics.get('roc_auc', 0),
                        'Win Rate': metrics.get('win_rate', 0),
                        'Max DD': metrics.get('max_drawdown', 0),
                        'Sharpe': metrics.get('sharpe_ratio', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1 Score': metrics.get('f1_score', 0)
                    })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Accuracy', ascending=False)
        
        self.report_data['summary_table'] = df
        return df
    
    def generate_performance_chart(self) -> Dict[str, Any]:
        """Generate performance comparison chart data."""
        df = self.generate_summary_table()
        
        if df.empty:
            return {"error": "No data available"}
        
        chart_data = {
            'type': 'bar',
            'data': {
                'labels': df['Horizon'] + ' - ' + df['Model'],
                'datasets': [
                    {
                        'label': 'Accuracy',
                        'data': df['Accuracy'].tolist(),
                        'backgroundColor': 'rgba(54, 162, 235, 0.5)'
                    },
                    {
                        'label': 'Win Rate',
                        'data': df['Win Rate'].tolist(),
                        'backgroundColor': 'rgba(75, 192, 192, 0.5)'
                    },
                    {
                        'label': 'ROC-AUC',
                        'data': df['ROC-AUC'].tolist(),
                        'backgroundColor': 'rgba(255, 205, 86, 0.5)'
                    }
                ]
            },
            'options': {
                'title': 'Model Performance Comparison',
                'scales': {
                    'y': {
                        'beginAtZero': False,
                        'min': 0.4,
                        'max': 1.0
                    }
                }
            }
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def generate_confidence_calibration_chart(self) -> Dict[str, Any]:
        """Generate confidence calibration chart."""
        # This would typically come from prediction history
        # For now, generate sample data
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        actual_win_rates = [0.52, 0.55, 0.62, 0.71, 0.78, 0.85, 0.92]
        
        chart_data = {
            'type': 'line',
            'data': {
                'labels': [f'{b:.2f}' for b in confidence_bins],
                'datasets': [
                    {
                        'label': 'Actual Win Rate',
                        'data': actual_win_rates,
                        'borderColor': 'rgba(255, 99, 132, 1)',
                        'tension': 0.1
                    },
                    {
                        'label': 'Perfect Calibration',
                        'data': confidence_bins,
                        'borderColor': 'rgba(75, 192, 192, 1)',
                        'borderDash': [5, 5],
                        'tension': 0.1
                    }
                ]
            },
            'options': {
                'title': 'Probability Calibration Analysis',
                'scales': {
                    'y': {
                        'min': 0,
                        'max': 1
                    }
                }
            }
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def generate_trading_metrics_chart(self) -> Dict[str, Any]:
        """Generate trading performance chart."""
        df = self.generate_summary_table()
        
        if df.empty:
            return {"error": "No data available"}
        
        chart_data = {
            'type': 'scatter',
            'data': {
                'datasets': [
                    {
                        'label': 'Models',
                        'data': df[['Win Rate', 'Max DD']].values.tolist(),
                        'backgroundColor': 'rgba(54, 162, 235, 0.7)',
                        'pointRadius': 8
                    }
                ]
            },
            'options': {
                'title': 'Win Rate vs Maximum Drawdown',
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': 'Win Rate'
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': 'Max Drawdown'
                        }
                    }
                }
            }
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate complete evaluation report."""
        report = {
            'summary_table': self.generate_summary_table(),
            'performance_chart': self.generate_performance_chart(),
            'calibration_chart': self.generate_confidence_calibration_chart(),
            'trading_metrics_chart': self.generate_trading_metrics_chart(),
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        self.report_data = report
        return report
    
    def save_report(self, path: str = "evaluation_report.json") -> None:
        """Save evaluation report to JSON file."""
        report = self.generate_report()
        
        # Convert DataFrames to dict for JSON serialization
        if 'summary_table' in report:
            report['summary_table'] = report['summary_table'].to_dict('records')
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Evaluation report saved to: {path}")
    
    def print_summary(self) -> None:
        """Print summary to console."""
        df = self.generate_summary_table()
        
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        if not df.empty:
            best_model = df.iloc[0]
            print(f"\nBest Model: {best_model['Horizon']} - {best_model['Model']}")
            print(f"  Accuracy: {best_model['Accuracy']:.4f}")
            print(f"  ROC-AUC: {best_model['ROC-AUC']:.4f}")
            print(f"  Win Rate: {best_model['Win Rate']:.4f}")


class SignalQualityMonitor:
    """
    Monitors signal quality in real-time to ensure trading decisions
    are based on reliable predictions.
    """
    
    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence
        self.signal_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def validate_signal(self, signal: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate a trading signal before execution.
        
        Returns:
            (is_valid, reason, enriched_signal)
        """
        confidence = signal.get('confidence', 0.0)
        direction = signal.get('direction', 'NEUTRAL')
        horizon = signal.get('horizon', '60s')
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence:.2f} below threshold {self.min_confidence}", signal
        
        # Check direction
        if direction not in ['UP', 'DOWN']:
            return False, f"Invalid direction: {direction}", signal
        
        # Check horizon
        if horizon not in ['30s', '60s', '120s', '300s']:
            return False, f"Invalid horizon: {horizon}", signal
        
        # Enrich signal with metadata
        enriched_signal = signal.copy()
        enriched_signal['validated_at'] = pd.Timestamp.now().isoformat()
        enriched_signal['validation_status'] = 'APPROVED'
        
        # Record in history
        self.record_signal(enriched_signal)
        
        return True, "Signal approved", enriched_signal
    
    def record_signal(self, signal: Dict[str, Any]) -> None:
        """Record signal in history for quality monitoring."""
        self.signal_history.append(signal)
        
        # Maintain history size
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get signal quality metrics from history."""
        if not self.signal_history:
            return {"message": "No signal history available"}
        
        total = len(self.signal_history)
        up_signals = sum(1 for s in self.signal_history if s.get('direction') == 'UP')
        down_signals = sum(1 for s in self.signal_history if s.get('direction') == 'DOWN')
        avg_confidence = np.mean([s.get('confidence', 0) for s in self.signal_history])
        
        # Calculate approval rate
        approved = sum(1 for s in self.signal_history if s.get('validation_status') == 'APPROVED')
        approval_rate = approved / total * 100 if total > 0 else 0
        
        return {
            'total_signals': total,
            'up_signals': up_signals,
            'down_signals': down_signals,
            'up_percentage': up_signals / total * 100 if total > 0 else 0,
            'avg_confidence': avg_confidence,
            'approval_rate': approval_rate
        }
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate quality monitoring report."""
        metrics = self.get_quality_metrics()
        
        report = {
            'metrics': metrics,
            'recent_signals': self.signal_history[-10:] if self.signal_history else [],
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        return report


def create_evaluation_dashboard(
    training_results: Optional[Dict[str, Dict[str, Any]]] = None
) -> ModelEvaluationDashboard:
    """Create and configure evaluation dashboard."""
    dashboard = ModelEvaluationDashboard(training_results)
    return dashboard


def create_signal_monitor(min_confidence: float = 0.65) -> SignalQualityMonitor:
    """Create and configure signal quality monitor."""
    monitor = SignalQualityMonitor(min_confidence=min_confidence)
    return monitor