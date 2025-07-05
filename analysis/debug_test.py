#!/usr/bin/env python3
"""
Debug test to identify JSON serialization issues
"""

import pandas as pd
import numpy as np
import json
from utils.analysis_orchestrator import AnalysisOrchestrator

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    
    data = {
        'date': pd.date_range('2023-01-01', periods=100),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 100),
        'sales_amount': np.random.normal(1000, 200, 100),
        'quantity_sold': np.random.poisson(5, 100),
        'customer_age': np.random.randint(18, 80, 100),
        'marketing_spend': np.random.uniform(50, 500, 100),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    }
    
    # Ensure positive values
    data['sales_amount'] = np.abs(data['sales_amount'])
    data['marketing_spend'] = np.abs(data['marketing_spend'])
    
    # Add some correlations
    data['sales_amount'] = data['sales_amount'] + 0.3 * data['marketing_spend']
    data['quantity_sold'] = data['quantity_sold'] + (data['sales_amount'] / 200).astype(int)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("🔍 Debug Test: JSON Serialization")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    goal = "Understand sales patterns and optimize marketing spend"
    
    print(f"Dataset shape: {df.shape}")
    print(f"Goal: {goal}")
    print()
    
    # Initialize orchestrator
    orchestrator = AnalysisOrchestrator()
    
    try:
        # Perform analysis
        print("Performing analysis...")
        results = orchestrator.perform_comprehensive_analysis(df, goal)
        
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
        else:
            print("✅ Analysis completed successfully")
            
            # Try to serialize to JSON
            print("Testing JSON serialization...")
            try:
                json_str = json.dumps(results, cls=NumpyEncoder, indent=2)
                print("✅ JSON serialization successful")
                print(f"JSON length: {len(json_str)} characters")
            except Exception as e:
                print(f"❌ JSON serialization failed: {e}")
                
                # Try to identify the problematic object
                print("\nDebugging serialization issue...")
                for key, value in results.items():
                    try:
                        json.dumps({key: value}, cls=NumpyEncoder)
                        print(f"✅ {key}: OK")
                    except Exception as sub_e:
                        print(f"❌ {key}: {sub_e}")
                        
                        # If it's a list/dict, check individual items
                        if isinstance(value, (list, dict)):
                            print(f"   Checking contents of {key}...")
                            if isinstance(value, list):
                                for i, item in enumerate(value[:5]):  # Check first 5 items
                                    try:
                                        json.dumps(item, cls=NumpyEncoder)
                                        print(f"   ✅ Item {i}: OK")
                                    except Exception as item_e:
                                        print(f"   ❌ Item {i}: {item_e}")
                                        print(f"   Type: {type(item)}")
                                        if hasattr(item, '__dict__'):
                                            print(f"   Attributes: {list(item.__dict__.keys())}")
                            elif isinstance(value, dict):
                                for sub_key, sub_value in list(value.items())[:5]:  # Check first 5 items
                                    try:
                                        json.dumps({sub_key: sub_value}, cls=NumpyEncoder)
                                        print(f"   ✅ {sub_key}: OK")
                                    except Exception as sub_item_e:
                                        print(f"   ❌ {sub_key}: {sub_item_e}")
                                        print(f"   Type: {type(sub_value)}")
                
    except Exception as e:
        print(f"❌ Analysis failed with exception: {e}")
        import traceback
        traceback.print_exc() 