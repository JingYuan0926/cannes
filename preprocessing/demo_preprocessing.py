#!/usr/bin/env python3
"""
Demo script for AI Data Preprocessing Agent

This script demonstrates how to use the preprocessing service programmatically
to analyze and preprocess datasets.
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os
import time

# Configuration
SERVICE_URL = "http://localhost:3031"
SAMPLE_DATA_FILE = "sample_preprocessing_data.csv"

def create_sample_data():
    """Create a sample dataset with various data quality issues for demonstration"""
    print("Creating sample dataset with various data types and quality issues...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        # Numerical features with different scales and outliers
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'height': np.random.normal(170, 10, n_samples),
        'weight': np.random.normal(70, 15, n_samples),
        
        # Categorical features
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        
        # Text features
        'description': [f"This is sample text description {i} with various content" for i in range(n_samples)],
        'comments': [f"Comment {i}: Some feedback about the product or service" for i in range(n_samples)],
        
        # Datetime features
        'registration_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
        'last_activity': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(n_samples)],
        
        # Binary features
        'is_active': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
        'has_subscription': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues
    # Add missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'education'] = np.nan
    
    # Add outliers
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[outlier_indices, 'income'] = df.loc[outlier_indices, 'income'] * 10
    
    # Add duplicates
    duplicate_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
    
    # Convert datetime to string for CSV compatibility
    df['registration_date'] = df['registration_date'].dt.strftime('%Y-%m-%d')
    df['last_activity'] = df['last_activity'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV
    df.to_csv(SAMPLE_DATA_FILE, index=False)
    print(f"Sample dataset created: {SAMPLE_DATA_FILE}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def check_service_health():
    """Check if the preprocessing service is running"""
    try:
        response = requests.get(f"{SERVICE_URL}/")
        if response.status_code == 200:
            print("✓ Preprocessing service is running")
            return True
        else:
            print(f"✗ Service returned status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("✗ Cannot connect to preprocessing service")
        print(f"  Make sure the service is running on {SERVICE_URL}")
        return False

def analyze_dataset():
    """Analyze the sample dataset using the preprocessing service"""
    print("\n" + "="*60)
    print("ANALYZING DATASET")
    print("="*60)
    
    if not os.path.exists(SAMPLE_DATA_FILE):
        print(f"Sample data file not found: {SAMPLE_DATA_FILE}")
        return None
    
    try:
        with open(SAMPLE_DATA_FILE, 'rb') as f:
            response = requests.post(f"{SERVICE_URL}/analyze", files={'file': f})
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['analysis']
            
            print("Dataset Analysis Results:")
            print(f"Shape: {analysis['basic_info']['shape']}")
            print(f"Columns: {len(analysis['basic_info']['columns'])}")
            print(f"Missing values: {sum(analysis['basic_info']['missing_values'].values())}")
            print(f"Duplicate rows: {analysis['basic_info']['duplicate_rows']}")
            
            print("\nNumerical Columns Analysis:")
            for col, info in analysis['numerical_analysis'].items():
                print(f"  {col}: mean={info['statistics']['mean']:.2f}, "
                      f"std={info['statistics']['std']:.2f}, "
                      f"outliers={info['distribution']['outlier_count']}")
            
            print("\nCategorical Columns Analysis:")
            for col, info in analysis['categorical_analysis'].items():
                print(f"  {col}: unique={info['statistics']['unique_count']}, "
                      f"recommended_encoding={info['encoding']['recommended_encoding']}")
            
            print("\nPreprocessing Recommendations:")
            recs = analysis['preprocessing_recommendations']
            print(f"Priority actions: {recs['priority_actions']}")
            print(f"Optional actions: {recs['optional_actions']}")
            print(f"Feature engineering: {recs['feature_engineering_actions']}")
            
            return analysis
            
        else:
            print(f"Error analyzing dataset: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

def preprocess_dataset():
    """Preprocess the dataset using AI recommendations"""
    print("\n" + "="*60)
    print("PREPROCESSING DATASET WITH AI RECOMMENDATIONS")
    print("="*60)
    
    if not os.path.exists(SAMPLE_DATA_FILE):
        print(f"Sample data file not found: {SAMPLE_DATA_FILE}")
        return None
    
    try:
        with open(SAMPLE_DATA_FILE, 'rb') as f:
            response = requests.post(f"{SERVICE_URL}/preprocess", files={'file': f})
        
        if response.status_code == 200:
            result = response.json()
            
            print("Preprocessing Results:")
            print(f"Original shape: {result['original_shape']}")
            print(f"Processed shape: {result['processed_shape']}")
            
            print("\nAI Recommendations Applied:")
            recommendations = result['recommendations']
            if 'preprocessing_pipeline' in recommendations:
                for i, step in enumerate(recommendations['preprocessing_pipeline'], 1):
                    print(f"  {i}. {step['technique']}")
                    print(f"     Parameters: {step['parameters']}")
                    print(f"     Reason: {step['reason']}")
                    print(f"     Priority: {step['priority']}")
            
            print("\nExecution Log:")
            for i, log_entry in enumerate(result['execution_log'], 1):
                status = "✓" if log_entry['status'] == 'success' else "✗"
                print(f"  {status} {log_entry['step']}: {log_entry['status']}")
                if log_entry['status'] == 'success':
                    print(f"     Shape after: {log_entry['shape_after']}")
                else:
                    print(f"     Error: {log_entry.get('error', 'Unknown error')}")
            
            return result
            
        else:
            print(f"Error preprocessing dataset: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None

def custom_preprocessing():
    """Demonstrate custom preprocessing pipeline"""
    print("\n" + "="*60)
    print("CUSTOM PREPROCESSING PIPELINE")
    print("="*60)
    
    # Define custom preprocessing pipeline
    custom_config = {
        "preprocessing_pipeline": [
            {
                "technique": "standardize_numerical_features",
                "parameters": {"method": "robust"},
                "columns": ["age", "income", "score"],
                "priority": 8,
                "reason": "Apply robust scaling to numerical features"
            },
            {
                "technique": "normalize_categorical_features",
                "parameters": {"method": "one_hot_encoding"},
                "columns": ["category", "city"],
                "priority": 7,
                "reason": "One-hot encode categorical variables"
            },
            {
                "technique": "engineer_datetime_features",
                "parameters": {"extract": ["year", "month", "weekday"]},
                "columns": ["registration_date", "last_activity"],
                "priority": 6,
                "reason": "Extract temporal features from datetime columns"
            },
            {
                "technique": "handle_text_features",
                "parameters": {"method": "tfidf", "max_features": 50},
                "columns": ["description"],
                "priority": 5,
                "reason": "Convert text to numerical features using TF-IDF"
            }
        ]
    }
    
    print("Custom Pipeline Configuration:")
    for i, step in enumerate(custom_config['preprocessing_pipeline'], 1):
        print(f"  {i}. {step['technique']}")
        print(f"     Method: {step['parameters']}")
        print(f"     Columns: {step['columns']}")
        print(f"     Priority: {step['priority']}")
    
    if not os.path.exists(SAMPLE_DATA_FILE):
        print(f"Sample data file not found: {SAMPLE_DATA_FILE}")
        return None
    
    try:
        with open(SAMPLE_DATA_FILE, 'rb') as f:
            response = requests.post(
                f"{SERVICE_URL}/custom-preprocess", 
                files={'file': f},
                data={'config': json.dumps(custom_config)}
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\nCustom Preprocessing Results:")
            print(f"Original shape: {result['original_shape']}")
            print(f"Processed shape: {result['processed_shape']}")
            
            print("\nExecution Log:")
            for i, log_entry in enumerate(result['execution_log'], 1):
                status = "✓" if log_entry['status'] == 'success' else "✗"
                print(f"  {status} {log_entry['step']}: {log_entry['status']}")
                if log_entry['status'] == 'success':
                    print(f"     Shape after: {log_entry['shape_after']}")
                else:
                    print(f"     Error: {log_entry.get('error', 'Unknown error')}")
            
            return result
            
        else:
            print(f"Error in custom preprocessing: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error during custom preprocessing: {str(e)}")
        return None

def get_available_techniques():
    """Get and display all available preprocessing techniques"""
    print("\n" + "="*60)
    print("AVAILABLE PREPROCESSING TECHNIQUES")
    print("="*60)
    
    try:
        response = requests.get(f"{SERVICE_URL}/techniques")
        
        if response.status_code == 200:
            techniques = response.json()['techniques']
            
            print(f"Total techniques available: {len(techniques)}")
            print()
            
            for name, details in techniques.items():
                print(f"• {name}")
                print(f"  Description: {details['description']}")
                print(f"  Methods: {', '.join(details['methods'])}")
                print(f"  Parameters: {list(details['parameters'].keys())}")
                print()
            
            return techniques
            
        else:
            print(f"Error getting techniques: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error getting techniques: {str(e)}")
        return None

def download_preprocessed_data():
    """Download the preprocessed dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING PREPROCESSED DATA")
    print("="*60)
    
    try:
        response = requests.get(f"{SERVICE_URL}/download/preprocessed")
        
        if response.status_code == 200:
            output_file = "downloaded_preprocessed_data.csv"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Preprocessed data downloaded: {output_file}")
            
            # Show some info about the downloaded file
            df = pd.read_csv(output_file)
            print(f"Downloaded dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            return output_file
            
        else:
            print(f"Error downloading data: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None

def main():
    """Main demonstration function"""
    print("="*60)
    print("AI DATA PREPROCESSING AGENT - DEMONSTRATION")
    print("="*60)
    
    # Check service health
    if not check_service_health():
        print("\nPlease start the preprocessing service first:")
        print("  cd preprocessing")
        print("  python app.py")
        return
    
    # Create sample data
    create_sample_data()
    
    # Get available techniques
    get_available_techniques()
    
    # Analyze dataset
    analysis = analyze_dataset()
    
    # AI-powered preprocessing
    if analysis:
        preprocess_result = preprocess_dataset()
        
        if preprocess_result:
            # Download preprocessed data
            download_preprocessed_data()
    
    # Demonstrate custom preprocessing
    custom_preprocessing()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED")
    print("="*60)
    print("\nFiles created:")
    print(f"  • {SAMPLE_DATA_FILE} - Original sample dataset")
    if os.path.exists("downloaded_preprocessed_data.csv"):
        print("  • downloaded_preprocessed_data.csv - AI-preprocessed dataset")
    if os.path.exists("data/custom_preprocessed.csv"):
        print("  • data/custom_preprocessed.csv - Custom-preprocessed dataset")
    
    print("\nNext steps:")
    print("  • Modify the custom preprocessing pipeline to suit your needs")
    print("  • Try uploading your own CSV files to the service")
    print("  • Integrate the service into your ML pipeline")
    print("  • Explore different preprocessing techniques and parameters")

if __name__ == "__main__":
    main() 