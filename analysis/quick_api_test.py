#!/usr/bin/env python3
"""
Quick API Test for AI-Powered Data Analysis System
"""

import requests
import json
import pandas as pd
import numpy as np
from io import StringIO

def create_test_data():
    """Create test data as CSV string"""
    np.random.seed(42)
    
    data = {
        'date': pd.date_range('2023-01-01', periods=50),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], 50),
        'sales_amount': np.random.normal(1000, 200, 50),
        'quantity_sold': np.random.poisson(5, 50),
        'customer_age': np.random.randint(18, 80, 50),
        'marketing_spend': np.random.uniform(50, 500, 50),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 50),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 50)
    }
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def test_api():
    """Test the Flask API"""
    base_url = "http://localhost:3040"
    
    print("🧪 Testing AI-Powered Data Analysis API")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("   Make sure the Flask server is running: python app.py")
        return False
    
    # Test 2: Service info
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Service info retrieved")
            service_info = response.json()
            print(f"   Service: {service_info['service']}")
            print(f"   Version: {service_info['version']}")
        else:
            print("❌ Service info failed")
    except Exception as e:
        print(f"❌ Service info failed: {e}")
    
    # Test 3: Analysis with JSON data
    try:
        print("\n📊 Testing analysis with JSON data...")
        test_data = create_test_data()
        df = pd.read_csv(StringIO(test_data))
        
        payload = {
            "data": df.to_dict('records'),
            "goal": "Analyze sales patterns and optimize marketing strategy"
        }
        
        response = requests.post(
            f"{base_url}/analyze",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis completed successfully")
            print(f"   Analysis ID: {result['analysis_id']}")
            print(f"   Total analyses: {result['summary']['total_analyses']}")
            print(f"   Total graphs: {result['summary']['total_graphs']}")
            print(f"   Analytics types: {result['summary']['analytics_types']}")
            
            # Test 4: Retrieve results
            analysis_id = result['analysis_id']
            response = requests.get(f"{base_url}/results/{analysis_id}")
            if response.status_code == 200:
                print("✅ Results retrieval successful")
            else:
                print("❌ Results retrieval failed")
                
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False
    
    print("\n🎉 All API tests passed!")
    return True

if __name__ == "__main__":
    test_api() 