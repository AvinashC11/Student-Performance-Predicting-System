"""
Visualization Module
Handles creation of charts and graphs for model analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os

class ResultVisualizer:
    """Class to handle visualization of results and model analysis"""
    
    def __init__(self):
        """Initialize the visualizer with default settings"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
    def plot_feature_importance(self, model_path="models/trained_model.pkl", 
                                save_path=None):
        """
        Plot feature importance for tree-based models
        
        Args:
            model_path (str): Path to trained model
            save_path (str): Optional path to save the figure
        """
        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data['model']
            feature_names = model_data.get('feature_names', None)
            
            if not hasattr(model, 'feature_importances_'):
                print("This model doesn't support feature importance visualization")
                return
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names if feature_names else range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'], importance_df['importance'], 
                    color=self.colors[0])
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.title('Feature Importance Analysis')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Figure saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting feature importance: {str(e)}")
    
    def plot_model_comparison(self, results_data=None):
        """
        Compare performance of different models
        
        Args:
            results_data (dict): Dictionary with model names as keys and metrics as values
        """
        # Sample data if none provided
        if results_data is None:
            results_data = {
                'Random Forest': {'R2': 0.87, 'RMSE': 5.42, 'MAE': 4.18},
                'Decision Tree': {'R2': 0.82, 'RMSE': 6.31, 'MAE': 4.92},
                'Linear Regression': {'R2': 0.75, 'RMSE': 7.45, 'MAE': 5.67}
            }
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(results_data.keys())
        metrics = ['R2', 'RMSE', 'MAE']
        titles = ['R² Score (Higher is Better)', 'RMSE (Lower is Better)', 
                 'MAE (Lower is Better)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            values = [results_data[model][metric] for model in models]
            axes[idx].bar(models, values, color=self.colors[:len(models)])
            axes[idx].set_title(title)
            axes[idx].set_ylabel(metric)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, data_path="data/processed/cleaned_data.csv"):
        """
        Plot correlation heatmap of features
        
        Args:
            data_path (str): Path to data CSV file
        """
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting correlation heatmap: {str(e)}")
    
    def plot_predictions_vs_actual(self, model_path="models/trained_model.pkl",
                                   data_path="data/processed/cleaned_data.csv"):
        """
        Plot predicted vs actual values
        
        Args:
            model_path (str): Path to trained model
            data_path (str): Path to data CSV
        """
        try:
            # Load model and data
            model_data = joblib.load(model_path)
            model = model_data['model']
            
            df = pd.read_csv(data_path)
            
            # Prepare data
            X = df.drop(columns=['final_grade'])
            y_actual = df['final_grade']
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_actual, y_pred, alpha=0.5, color=self.colors[0])
            
            # Add perfect prediction line
            min_val = min(y_actual.min(), y_pred.min())
            max_val = max(y_actual.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 
                    'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('Actual Grade')
            plt.ylabel('Predicted Grade')
            plt.title('Predicted vs Actual Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting predictions vs actual: {str(e)}")
    
    def plot_residuals(self, model_path="models/trained_model.pkl",
                      data_path="data/processed/cleaned_data.csv"):
        """
        Plot residual analysis
        
        Args:
            model_path (str): Path to trained model
            data_path (str): Path to data CSV
        """
        try:
            # Load model and data
            model_data = joblib.load(model_path)
            model = model_data['model']
            
            df = pd.read_csv(data_path)
            X = df.drop(columns=['final_grade'])
            y_actual = df['final_grade']
            
            # Calculate residuals
            y_pred = model.predict(X)
            residuals = y_actual - y_pred
            
            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Residual plot
            axes[0].scatter(y_pred, residuals, alpha=0.5, color=self.colors[1])
            axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residual Plot')
            axes[0].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[1].hist(residuals, bins=30, color=self.colors[2], edgecolor='black')
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Residuals')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting residuals: {str(e)}")

if __name__ == "__main__":
    # Example usage
    print("Visualization Module")
    print("-" * 50)
    
    visualizer = ResultVisualizer()
    
    print("\n1. Feature Importance")
    print("2. Model Comparison")
    print("3. Correlation Heatmap")
    print("4. Predictions vs Actual")
    
    choice = input("\nSelect visualization (1-4): ")
    
    try:
        if choice == '1':
            visualizer.plot_feature_importance()
        elif choice == '2':
            visualizer.plot_model_comparison()
        elif choice == '3':
            visualizer.plot_correlation_heatmap()
        elif choice == '4':
            visualizer.plot_predictions_vs_actual()
        else:
            print("Invalid choice")
    except Exception as e:
        print(f"Error: {str(e)}")
