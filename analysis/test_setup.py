#!/usr/bin/env python3
"""
Test Script for AI Data Analysis Pipeline
This script verifies the basic setup and functionality of the analysis service.
"""

import os
import sys
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test Flask and web framework imports
        import flask
        from flask import Flask, request, jsonify
        print("✓ Flask imports successful")
        
        # Test data analysis imports
        import pandas as pd
        import numpy as np
        from scipy import stats
        print("✓ Data analysis imports successful")
        
        # Test visualization imports
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        print("✓ Visualization imports successful")
        
        # Test OpenAI import
        import openai
        print("✓ OpenAI import successful")
        
        # Test utility modules
        from utils.data_analyzer import DataAnalyzer
        from utils.visualization_engine import VisualizationEngine
        from utils.filter_engine import FilterEngine
        from utils.business_intelligence import BusinessIntelligenceEngine
        print("✓ Custom utility modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_utility_classes():
    """Test that utility classes can be instantiated."""
    print("\nTesting utility classes...")
    
    try:
        # Import classes first
        from utils.data_analyzer import DataAnalyzer
        from utils.visualization_engine import VisualizationEngine
        from utils.filter_engine import FilterEngine
        from utils.business_intelligence import BusinessIntelligenceEngine
        
        # Test DataAnalyzer
        analyzer = DataAnalyzer()
        print("✓ DataAnalyzer instantiated successfully")
        
        # Test VisualizationEngine
        viz_engine = VisualizationEngine()
        print("✓ VisualizationEngine instantiated successfully")
        
        # Test FilterEngine
        filter_engine = FilterEngine()
        print("✓ FilterEngine instantiated successfully")
        
        # Test BusinessIntelligenceEngine
        bi_engine = BusinessIntelligenceEngine()
        print("✓ BusinessIntelligenceEngine instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error instantiating classes: {e}")
        return False

def test_sample_data_analysis():
    """Test basic data analysis functionality with sample data."""
    print("\nTesting sample data analysis...")
    
    try:
        # Create sample dataset
        sample_data = {
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'sales': np.random.normal(1000, 200, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'customer_count': np.random.randint(50, 200, 100)
        }
        
        df = pd.DataFrame(sample_data)
        print("✓ Sample dataset created successfully")
        
        # Test DataAnalyzer
        from utils.data_analyzer import DataAnalyzer
        analyzer = DataAnalyzer()
        
        analysis_results = analyzer.comprehensive_analysis(df)
        print("✓ Data analysis completed successfully")
        print(f"  - Analysis contains {len(analysis_results)} sections")
        
        # Test basic statistics
        if 'basic_statistics' in analysis_results:
            stats = analysis_results['basic_statistics']
            print(f"  - Basic statistics: {len(stats)} columns analyzed")
        
        # Test data quality
        if 'data_quality' in analysis_results:
            quality = analysis_results['data_quality']
            print(f"  - Data quality assessed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data analysis: {e}")
        return False

def test_visualization_engine():
    """Test basic visualization functionality."""
    print("\nTesting visualization engine...")
    
    try:
        # Create sample data
        df = pd.DataFrame({
            'x': range(10),
            'y': np.random.randint(1, 100, 10),
            'category': ['A'] * 5 + ['B'] * 5
        })
        
        from utils.visualization_engine import VisualizationEngine
        viz_engine = VisualizationEngine()
        
        # Test bar chart creation
        config = {
            'chart_type': 'bar',
            'x_axis': 'x',
            'y_axis': 'y',
            'title': 'Test Bar Chart'
        }
        
        result = viz_engine.create_visualization(df, config)
        print("✓ Bar chart created successfully")
        
        if 'chart_html' in result:
            print("  - Chart HTML generated")
        if 'insights' in result:
            print(f"  - Generated insights")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        return False

def test_filter_engine():
    """Test filtering functionality."""
    print("\nTesting filter engine...")
    
    try:
        # Create sample data
        df = pd.DataFrame({
            'value': range(100),
            'category': ['A'] * 50 + ['B'] * 50,
            'score': np.random.normal(50, 15, 100)
        })
        
        from utils.filter_engine import FilterEngine
        filter_engine = FilterEngine()
        
        # Test basic filter
        filters = [{
            'column': 'value',
            'type': 'greater_than',
            'value': 50
        }]
        
        filtered_df = filter_engine.apply_filters(df, filters)
        print("✓ Filter applied successfully")
        print(f"  - Original rows: {len(df)}")
        print(f"  - Filtered rows: {len(filtered_df)}")
        
        # Test available filters
        available_filters = filter_engine.get_available_filters(df)
        print(f"  - Available filters: {len(available_filters)} columns")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in filtering: {e}")
        return False

def test_business_intelligence():
    """Test business intelligence functionality."""
    print("\nTesting business intelligence engine...")
    
    try:
        # Create sample business data
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=50, freq='D'),
            'revenue': np.random.normal(10000, 2000, 50),
            'customers': np.random.randint(100, 500, 50),
            'product': np.random.choice(['Product A', 'Product B', 'Product C'], 50),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 50)
        })
        
        from utils.business_intelligence import BusinessIntelligenceEngine
        bi_engine = BusinessIntelligenceEngine()
        
        # Test basic insights generation
        insights = bi_engine.generate_insights(df, "What are the revenue trends?")
        print("✓ Business intelligence insights generated successfully")
        
        if 'executive_summary' in insights:
            print("  - Executive summary generated")
        if 'key_findings' in insights:
            print("  - Key findings identified")
        if 'strategic_recommendations' in insights:
            print("  - Strategic recommendations provided")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in business intelligence: {e}")
        return False

def test_environment_setup():
    """Test environment configuration."""
    print("\nTesting environment setup...")
    
    try:
        # Check for .env.example file
        env_example_path = Path('.env.example')
        if env_example_path.exists():
            print("✓ .env.example file found")
        else:
            print("✗ .env.example file not found")
            return False
        
        # Check for requirements.txt
        requirements_path = Path('requirements.txt')
        if requirements_path.exists():
            print("✓ requirements.txt file found")
            
            # Count dependencies
            with open(requirements_path, 'r') as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                print(f"  - {len(deps)} dependencies listed")
        else:
            print("✗ requirements.txt file not found")
            return False
        
        # Check for README.md
        readme_path = Path('README.md')
        if readme_path.exists():
            print("✓ README.md file found")
        else:
            print("✗ README.md file not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking environment: {e}")
        return False

def test_flask_app_creation():
    """Test that the Flask app can be created."""
    print("\nTesting Flask app creation...")
    
    try:
        # Set minimal environment variables
        os.environ['OPENAI_API_KEY'] = 'test-key'
        os.environ['FLASK_ENV'] = 'testing'
        
        # Import and create app
        from app import app
        
        # Test app creation
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/')
            print("✓ Flask app created successfully")
            print(f"  - Health endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                print("  - Health endpoint responding correctly")
            
        return True
        
    except Exception as e:
        print(f"✗ Error creating Flask app: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI Data Analysis Pipeline - Setup Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_utility_classes,
        test_sample_data_analysis,
        test_visualization_engine,
        test_filter_engine,
        test_business_intelligence,
        test_environment_setup,
        test_flask_app_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The setup is working correctly.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your OpenAI API key")
        print("2. Run 'python app.py' to start the service")
        print("3. Visit http://localhost:3032 to check the health endpoint")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("- Missing dependencies: run 'pip install -r requirements.txt'")
        print("- Missing OpenAI API key: add OPENAI_API_KEY to .env file")
        print("- Import errors: check Python version (3.8+ required)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 