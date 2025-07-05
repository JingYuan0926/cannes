#!/usr/bin/env python3
"""
Test script for AI-Powered Data Analysis System

This script creates sample data and tests the analysis system.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from io import StringIO

def create_sample_data():
    """Create sample sales data for testing"""
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
    data = {
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
        'sales_amount': np.random.normal(100, 30, n_samples),
        'quantity_sold': np.random.poisson(5, n_samples),
        'customer_age': np.random.normal(35, 12, n_samples),
        'marketing_spend': np.random.normal(20, 8, n_samples),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    }
    
    # Add some correlations
    data['sales_amount'] = data['sales_amount'] + data['marketing_spend'] * 0.5
    data['quantity_sold'] = data['quantity_sold'] + (data['sales_amount'] / 50).astype(int)
    
    # Ensure positive values
    data['sales_amount'] = np.abs(data['sales_amount'])
    data['customer_age'] = np.abs(data['customer_age'])
    data['marketing_spend'] = np.abs(data['marketing_spend'])
    
    df = pd.DataFrame(data)
    return df

def test_analysis_system(base_url="http://localhost:3040"):
    """Test the AI analysis system"""
    
    print("🧪 Testing AI-Powered Data Analysis System")
    print("=" * 50)
    
    # 1. Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # 2. Test service status
    print("\n2. Testing service status...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Service status check passed")
            print(f"   Service: {response.json().get('service', 'Unknown')}")
        else:
            print(f"❌ Service status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Service status failed: {e}")
    
    # 3. Create sample data
    print("\n3. Creating sample data...")
    df = create_sample_data()
    print(f"✅ Created sample dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Save to CSV for testing
    csv_file = "test_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"   Saved to {csv_file}")
    
    # 4. Test analysis with file upload
    print("\n4. Testing analysis with file upload...")
    try:
        with open(csv_file, 'rb') as f:
            files = {'file': f}
            data = {'goal': 'Analyze sales performance and identify key factors affecting revenue'}
            
            response = requests.post(f"{base_url}/analyze", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis completed successfully")
            print(f"   Analysis ID: {result.get('analysis_id', 'Unknown')}")
            print(f"   Recommended Category: {result.get('ai_strategy', {}).get('recommended_category', 'Unknown')}")
            print(f"   Recommended Algorithm: {result.get('ai_strategy', {}).get('recommended_algorithm', 'Unknown')}")
            
            # Print some insights
            ai_insights = result.get('ai_insights', {})
            insights = ai_insights.get('insights', [])
            if insights:
                print(f"   Key Insights:")
                for i, insight in enumerate(insights[:3], 1):
                    print(f"     {i}. {insight}")
            
            analysis_id = result.get('analysis_id')
            
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    
    # 5. Test retrieving results
    if analysis_id:
        print("\n5. Testing result retrieval...")
        try:
            response = requests.get(f"{base_url}/results/{analysis_id}")
            if response.status_code == 200:
                print("✅ Result retrieval successful")
            else:
                print(f"❌ Result retrieval failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Result retrieval failed: {e}")
    
    # 6. Test listing analyses
    print("\n6. Testing analysis listing...")
    try:
        response = requests.get(f"{base_url}/list-analyses")
        if response.status_code == 200:
            analyses = response.json()
            print(f"✅ Analysis listing successful")
            print(f"   Total analyses: {len(analyses.get('analyses', []))}")
        else:
            print(f"❌ Analysis listing failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Analysis listing failed: {e}")
    
    # 7. Test JSON data analysis
    print("\n7. Testing JSON data analysis...")
    try:
        # Use a smaller subset for JSON testing
        json_data = df.head(100).to_dict('records')
        
        payload = {
            'data': json_data,
            'goal': 'Identify customer segments and purchasing patterns'
        }
        
        response = requests.post(f"{base_url}/analyze", 
                               json=payload,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print("✅ JSON data analysis completed successfully")
            print(f"   Analysis ID: {result.get('analysis_id', 'Unknown')}")
        else:
            print(f"❌ JSON data analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ JSON data analysis failed: {e}")
    
    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"\n🧹 Cleaned up {csv_file}")
    
    print("\n" + "=" * 50)
    print("🎉 Test completed! Check the results above.")
    print("   If you see mostly ✅, the system is working correctly.")
    print("   If you see ❌, check the error messages and logs.")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Check if custom URL provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3040"
    
    print(f"Testing AI Analysis System at: {base_url}")
    print("Make sure the server is running before running this test!")
    print("Start the server with: python app.py")
    print()
    
    # Wait a moment for user to read
    time.sleep(2)
    
    # Run tests
    test_analysis_system(base_url) 