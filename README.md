# Student Performance Prediction System

## Overview
This project implements a machine learning-based system to predict student academic performance based on various factors including study hours, attendance patterns, previous grades, and participation in extracurricular activities. The goal is to help educators identify at-risk students early and provide targeted interventions.

## Features
- **Data Preprocessing Module**: Handles missing values, outlier detection, and feature scaling
- **ML Model Training**: Supports multiple algorithms (Random Forest, Decision Tree, Linear Regression)
- **Performance Prediction**: Predicts final grades based on input features
- **Visualization Dashboard**: Interactive charts showing feature importance and model performance
- **Model Evaluation**: Comprehensive metrics including accuracy, RMSE, and R² score

## Technologies Used
- **Language**: Python 3.8+
- **ML Libraries**: scikit-learn, numpy, pandas
- **Visualization**: matplotlib, seaborn
- **Data Handling**: pandas, openpyxl
- **Model Persistence**: joblib/pickle

## Project Structure
```
student-performance-prediction/
├── data/
│   ├── raw/
│   │   └── student_data.csv
│   └── processed/
│       └── cleaned_data.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── prediction.py
│   └── visualization.py
├── models/
│   └── trained_model.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   └── test_model.py
├── requirements.txt
├── README.md
├── PROBLEM_STATEMENT.md
└── main.py
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/student-performance-prediction.git
   cd student-performance-prediction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## Usage

### Training the Model
```python
python src/model_training.py --algorithm random_forest --test_size 0.2
```

### Making Predictions
```python
python src/prediction.py --input data/new_student.csv
```

### Generating Visualizations
```python
python src/visualization.py --type feature_importance
```

## Dataset
The project uses a dataset with the following features:
- **study_hours**: Weekly study hours (0-40)
- **attendance**: Attendance percentage (0-100)
- **previous_grade**: Previous semester grade (0-100)
- **extracurricular**: Participation hours per week (0-20)
- **sleep_hours**: Average sleep hours per day (4-10)
- **family_support**: Binary (0 or 1)
- **final_grade**: Target variable (0-100)

*Dataset source: Custom collected data / Public education dataset*

## Model Performance
Current best model: **Random Forest Regressor**
- **R² Score**: 0.87
- **RMSE**: 5.42
- **MAE**: 4.18

## Screenshots
*Uploaded with extension .png*

## Testing
Run unit tests:
```bash
python -m pytest tests/
```

## Challenges Faced
- Handling imbalanced data and outliers
- Selecting optimal features without overfitting
- Balancing model complexity with interpretability

## Future Enhancements
- Web interface using Flask/Django
- Real-time prediction API
- Integration with school management systems
- Support for multi-class classification (grade categories)
- Deep learning models for improved accuracy

## Contributors
AVINASH.C - 25BCY10032 VIT Bhopal

## License
This project is developed for academic purposes as part of the VITyarthi Build Your Own Project initiative.

## References
- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- Student Performance Dataset: UCI ML Repository
