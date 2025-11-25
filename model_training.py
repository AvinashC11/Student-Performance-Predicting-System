"""
Model Training Module
Handles ML model training, evaluation, and persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import time

class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self, data_path):
        """
        Initialize the model trainer
        
        Args:
            data_path (str): Path to preprocessed data CSV
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.metrics = {}
        
    def load_data(self):
        """Load preprocessed data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def split_data(self, target_column='final_grade', test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            target_column (str): Name of the target variable column
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Separate features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Data split - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        
    def train(self, algorithm='random_forest', **kwargs):
        """
        Train the ML model
        
        Args:
            algorithm (str): Algorithm to use ('random_forest', 'decision_tree', 'linear_regression')
            **kwargs: Additional parameters for the model
        """
        if self.X_train is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        start_time = time.time()
        
        # Select algorithm
        if algorithm == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == 'decision_tree':
            self.model = DecisionTreeRegressor(
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif algorithm == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        self.metrics['training_time'] = training_time
        
        print(f"✓ Model trained successfully ({training_time:.2f} seconds)")
        
        return self.model
    
    def evaluate(self):
        """
        Evaluate model performance on test set
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        self.metrics['r2_score'] = r2_score(self.y_test, y_pred)
        self.metrics['rmse'] = np.sqrt(mean_squared_error(self.y_test, y_pred))
        self.metrics['mae'] = mean_absolute_error(self.y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                     cv=5, scoring='r2')
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()
        
        return self.metrics
    
    def get_feature_importance(self):
        """
        Get feature importance (only for tree-based models)
        
        Returns:
            pandas.DataFrame: Feature names and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("This model doesn't support feature importance.")
        
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_path):
        """
        Save trained model to disk
        
        Args:
            model_path (str): Path where model should be saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metrics
        model_data = {
            'model': self.model,
            'metrics': self.metrics,
            'feature_names': list(self.X_train.columns)
        }
        
        joblib.dump(model_data, model_path)
        print(f"✓ Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to saved model
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.metrics = model_data.get('metrics', {})
            print(f"✓ Model loaded from {model_path}")
            return self.model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

def train_and_save_model(data_path, model_path, algorithm='random_forest'):
    """
    Convenience function to train and save a model in one call
    
    Args:
        data_path (str): Path to preprocessed data
        model_path (str): Path where model should be saved
        algorithm (str): Algorithm to use
    """
    trainer = ModelTrainer(data_path)
    trainer.load_data()
    trainer.split_data()
    trainer.train(algorithm=algorithm)
    metrics = trainer.evaluate()
    trainer.save_model(model_path)
    
    return trainer, metrics

if __name__ == "__main__":
    # Example usage
    print("Model Training Module")
    print("-" * 50)
    
    data_path = "data/processed/cleaned_data.csv"
    model_path = "models/trained_model.pkl"
    
    try:
        trainer, metrics = train_and_save_model(data_path, model_path)
        
        print("\nModel Performance:")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
