"""
Data Preprocessing Module
Handles data loading, cleaning, outlier removal, and feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Class to handle all data preprocessing operations"""
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to the raw data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def handle_missing_values(self):
        """Handle missing values using mean imputation for numerical columns"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check for missing values
        missing_count = self.df.isnull().sum().sum()
        
        if missing_count > 0:
            print(f"  Found {missing_count} missing values")
            
            # Separate numerical and categorical columns
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            
            # Impute numerical columns with mean
            self.df[numerical_cols] = self.imputer.fit_transform(self.df[numerical_cols])
            
            print("  ✓ Missing values handled using mean imputation")
        else:
            print("  ✓ No missing values found")
        
        return self.df
    
    def remove_outliers(self, threshold=3):
        """
        Remove outliers using Z-score method
        
        Args:
            threshold (float): Z-score threshold (default: 3)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_size = len(self.df)
        
        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Calculate Z-scores
        z_scores = np.abs((self.df[numerical_cols] - self.df[numerical_cols].mean()) / self.df[numerical_cols].std())
        
        # Remove rows where any column has z-score > threshold
        self.df = self.df[(z_scores < threshold).all(axis=1)]
        
        removed_count = initial_size - len(self.df)
        print(f"  ✓ Removed {removed_count} outliers (Z-score threshold: {threshold})")
        
        return self.df
    
    def scale_features(self, exclude_columns=None):
        """
        Scale features using StandardScaler
        
        Args:
            exclude_columns (list): Columns to exclude from scaling (e.g., target variable)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if exclude_columns is None:
            exclude_columns = ['final_grade']  # Don't scale target variable
        
        # Get columns to scale
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_columns]
        
        # Scale the features
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])
        
        print(f"  ✓ Scaled {len(cols_to_scale)} features")
        
        return self.df
    
    def save_processed_data(self, output_path):
        """
        Save processed data to CSV
        
        Args:
            output_path (str): Path where processed data should be saved
        """
        if self.df is None:
            raise ValueError("No data to save. Process data first.")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"  ✓ Processed data saved to {output_path}")
    
    def show_statistics(self):
        """Display basic statistics about the processed data"""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        print("\n--- Data Statistics ---")
        print(f"Total records: {len(self.df)}")
        print(f"Total features: {len(self.df.columns)}")
        print("\nNumerical features summary:")
        print(self.df.describe())
        
        # Check data types
        print("\nData types:")
        print(self.df.dtypes)

# Utility function for standalone use
def preprocess_student_data(input_path, output_path):
    """
    Convenience function to preprocess data in one call
    
    Args:
        input_path (str): Path to raw data
        output_path (str): Path to save processed data
    """
    preprocessor = DataPreprocessor(input_path)
    preprocessor.load_data()
    preprocessor.handle_missing_values()
    preprocessor.remove_outliers()
    preprocessor.scale_features()
    preprocessor.save_processed_data(output_path)
    return preprocessor.df

if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("-" * 50)
    
    # This allows testing the module independently
    data_path = "data/raw/student_data.csv"
    output_path = "data/processed/cleaned_data.csv"
    
    try:
        df = preprocess_student_data(data_path, output_path)
        print("\n✓ Preprocessing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
