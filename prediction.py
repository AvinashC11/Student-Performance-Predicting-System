"""
Prediction Module
Handles making predictions using trained models
"""

import pandas as pd
import numpy as np
import joblib

class PerformancePredictor:
    """Class to handle performance predictions"""
    
    def __init__(self, model_path):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.metrics = None
        
    def load_model(self):
        """Load the trained model from disk"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', None)
            self.metrics = model_data.get('metrics', {})
            print("✓ Model loaded successfully")
            return self.model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self, student_data):
        """
        Make prediction for a single student
        
        Args:
            student_data (dict): Dictionary containing student features
                Expected keys: study_hours, attendance, previous_grade, 
                              extracurricular, sleep_hours, family_support
        
        Returns:
            float: Predicted final grade
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate input
        required_features = ['study_hours', 'attendance', 'previous_grade', 
                           'extracurricular', 'sleep_hours', 'family_support']
        
        missing_features = [f for f in required_features if f not in student_data]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Create DataFrame with correct feature order
        if self.feature_names:
            input_df = pd.DataFrame([student_data])[self.feature_names]
        else:
            input_df = pd.DataFrame([student_data])
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        # Ensure prediction is within valid range (0-100)
        prediction = np.clip(prediction, 0, 100)
        
        return prediction
    
    def predict_batch(self, csv_path):
        """
        Make predictions for multiple students from CSV file
        
        Args:
            csv_path (str): Path to CSV file containing student data
        
        Returns:
            pandas.DataFrame: Original data with predictions added
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            
            # Make predictions
            predictions = self.model.predict(df)
            
            # Add predictions to dataframe
            df['predicted_grade'] = np.clip(predictions, 0, 100)
            
            print(f"✓ Predictions made for {len(df)} students")
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        except Exception as e:
            raise Exception(f"Error making batch predictions: {str(e)}")
    
    def predict_with_confidence(self, student_data, n_estimators=None):
        """
        Make prediction with confidence interval (for ensemble models)
        
        Args:
            student_data (dict): Student features
            n_estimators (int): Number of trees to use (for RandomForest)
        
        Returns:
            tuple: (prediction, confidence_interval)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # This only works for ensemble models like RandomForest
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Confidence intervals only available for ensemble models")
        
        # Create input DataFrame
        if self.feature_names:
            input_df = pd.DataFrame([student_data])[self.feature_names]
        else:
            input_df = pd.DataFrame([student_data])
        
        # Get predictions from individual estimators
        predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(input_df)[0]
            predictions.append(pred)
        
        # Calculate mean and confidence interval
        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        confidence_interval = (mean_prediction - 1.96 * std_prediction, 
                              mean_prediction + 1.96 * std_prediction)
        
        # Clip to valid range
        mean_prediction = np.clip(mean_prediction, 0, 100)
        confidence_interval = (np.clip(confidence_interval[0], 0, 100),
                              np.clip(confidence_interval[1], 0, 100))
        
        return mean_prediction, confidence_interval
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        info = {
            'model_type': type(self.model).__name__,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        return info

def predict_student_performance(model_path, student_data):
    """
    Convenience function to make a single prediction
    
    Args:
        model_path (str): Path to trained model
        student_data (dict): Student features
    
    Returns:
        float: Predicted grade
    """
    predictor = PerformancePredictor(model_path)
    predictor.load_model()
    prediction = predictor.predict(student_data)
    return prediction

if __name__ == "__main__":
    # Example usage
    print("Performance Prediction Module")
    print("-" * 50)
    
    model_path = "models/trained_model.pkl"
    
    # Example student data
    sample_student = {
        'study_hours': 15,
        'attendance': 85,
        'previous_grade': 75,
        'extracurricular': 5,
        'sleep_hours': 7,
        'family_support': 1
    }
    
    try:
        prediction = predict_student_performance(model_path, sample_student)
        print(f"\nPredicted Grade: {prediction:.2f}")
        
        if prediction >= 85:
            print("Performance Level: Excellent")
        elif prediction >= 70:
            print("Performance Level: Good")
        elif prediction >= 50:
            print("Performance Level: Average")
        else:
            print("Performance Level: At Risk")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
