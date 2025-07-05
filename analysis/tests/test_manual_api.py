 #!/usr/bin/env python3
"""
Manual API Testing Script
This script demonstrates how to test the AI Data Analysis Pipeline API endpoints manually.
Run this script while your Flask app is running to test all endpoints.
"""

import requests
import json
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime


class APITester:
    """Class to test API endpoints manually"""
    
    def __init__(self, base_url="http://localhost:3032"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        print("🔍 Testing Health Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def create_sample_csv(self):
        """Create a sample CSV file for testing"""
        # Create sample business data
        data = {
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.random.randint(1000, 5000, 100),
            'profit': np.random.randint(100, 500, 100),
            'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Food'], 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'customer_count': np.random.randint(50, 200, 100),
            'marketing_spend': np.random.randint(100, 1000, 100)
        }
        
        df = pd.DataFrame(data)
        
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name, df
    
    def test_analyze_endpoint_csv(self):
        """Test the analyze endpoint with CSV upload"""
        print("\n🔍 Testing Analyze Endpoint (CSV Upload)...")
        
        csv_file, df = self.create_sample_csv()
        
        try:
            with open(csv_file, 'rb') as f:
                files = {'file': ('sample_data.csv', f, 'text/csv')}
                response = self.session.post(f"{self.base_url}/analyze", files=files)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Analysis completed successfully!")
                print(f"Dataset Info: {result['analysis']['basic_statistics']['dataset_info']}")
                print(f"Data Quality: Missing values = {result['analysis']['data_quality']['completeness']['total_missing_values']}")
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        finally:
            os.unlink(csv_file)
    
    def test_analyze_endpoint_json(self):
        """Test the analyze endpoint with JSON data"""
        print("\n🔍 Testing Analyze Endpoint (JSON Data)...")
        
        # Create sample JSON data
        data = {
            'sales': [100, 150, 200, 175, 300, 250, 180, 220, 190, 160],
            'profit': [20, 30, 40, 35, 60, 50, 36, 44, 38, 32],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'region': ['North', 'South', 'North', 'East', 'West', 'North', 'South', 'East', 'West', 'North']
        }
        
        json_payload = {
            'data': [dict(zip(data.keys(), values)) for values in zip(*data.values())],
            'columns': list(data.keys())
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=json_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Analysis completed successfully!")
                print(f"Dataset Info: {result['analysis']['basic_statistics']['dataset_info']}")
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_visualize_endpoint(self):
        """Test the visualize endpoint"""
        print("\n🔍 Testing Visualize Endpoint...")
        
        # First upload some data
        csv_file, df = self.create_sample_csv()
        
        try:
            # Upload data first
            with open(csv_file, 'rb') as f:
                files = {'file': ('sample_data.csv', f, 'text/csv')}
                upload_response = self.session.post(f"{self.base_url}/analyze", files=files)
            
            if upload_response.status_code != 200:
                print("❌ Failed to upload data for visualization test")
                return False
            
            # Now test visualization
            viz_config = {
                'chart_type': 'bar',
                'x_axis': 'category',
                'y_axis': 'sales',
                'title': 'Sales by Category',
                'color_scheme': 'viridis'
            }
            
            response = self.session.post(
                f"{self.base_url}/visualize",
                json=viz_config,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Visualization created successfully!")
                print(f"Chart Type: {result.get('chart_type', 'Unknown')}")
                print(f"Insights: {len(result.get('insights', []))} insights generated")
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        finally:
            os.unlink(csv_file)
    
    def test_filter_endpoint(self):
        """Test the filter endpoint"""
        print("\n🔍 Testing Filter Endpoint...")
        
        # First upload some data
        csv_file, df = self.create_sample_csv()
        
        try:
            # Upload data first
            with open(csv_file, 'rb') as f:
                files = {'file': ('sample_data.csv', f, 'text/csv')}
                upload_response = self.session.post(f"{self.base_url}/analyze", files=files)
            
            if upload_response.status_code != 200:
                print("❌ Failed to upload data for filter test")
                return False
            
            # Test filtering
            filters = [
                {
                    'column': 'sales',
                    'type': 'greater_than',
                    'value': 2000
                },
                {
                    'column': 'category',
                    'type': 'equals',
                    'value': 'Electronics'
                }
            ]
            
            response = self.session.post(
                f"{self.base_url}/filter",
                json={'filters': filters},
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Filtering completed successfully!")
                print(f"Original rows: {result.get('original_count', 'Unknown')}")
                print(f"Filtered rows: {result.get('filtered_count', 'Unknown')}")
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        finally:
            os.unlink(csv_file)
    
    def test_insights_endpoint(self):
        """Test the business insights endpoint"""
        print("\n🔍 Testing Business Insights Endpoint...")
        
        # First upload some data
        csv_file, df = self.create_sample_csv()
        
        try:
            # Upload data first
            with open(csv_file, 'rb') as f:
                files = {'file': ('sample_data.csv', f, 'text/csv')}
                upload_response = self.session.post(f"{self.base_url}/analyze", files=files)
            
            if upload_response.status_code != 200:
                print("❌ Failed to upload data for insights test")
                return False
            
            # Test insights generation
            insights_request = {
                'prompt': 'What are the key trends in sales and profit data? Are there any seasonal patterns?',
                'focus_areas': ['trends', 'performance', 'opportunities']
            }
            
            response = self.session.post(
                f"{self.base_url}/insights",
                json=insights_request,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Business insights generated successfully!")
                print(f"Executive Summary: {result.get('executive_summary', 'Not available')[:100]}...")
                print(f"Key Findings: {len(result.get('key_findings', []))} findings")
                print(f"Recommendations: {len(result.get('strategic_recommendations', []))} recommendations")
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        finally:
            os.unlink(csv_file)
    
    def test_export_endpoint(self):
        """Test the export endpoint"""
        print("\n🔍 Testing Export Endpoint...")
        
        # First upload some data
        csv_file, df = self.create_sample_csv()
        
        try:
            # Upload data first
            with open(csv_file, 'rb') as f:
                files = {'file': ('sample_data.csv', f, 'text/csv')}
                upload_response = self.session.post(f"{self.base_url}/analyze", files=files)
            
            if upload_response.status_code != 200:
                print("❌ Failed to upload data for export test")
                return False
            
            # Test export
            export_config = {
                'format': 'csv',
                'include_analysis': True,
                'include_visualizations': False
            }
            
            response = self.session.post(
                f"{self.base_url}/export",
                json=export_config,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Export completed successfully!")
                print(f"Content Type: {response.headers.get('content-type', 'Unknown')}")
                print(f"Response Size: {len(response.content)} bytes")
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        finally:
            os.unlink(csv_file)
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting API Tests...")
        print(f"Testing API at: {self.base_url}")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Analyze (CSV)", self.test_analyze_endpoint_csv),
            ("Analyze (JSON)", self.test_analyze_endpoint_json),
            ("Visualize", self.test_visualize_endpoint),
            ("Filter", self.test_filter_endpoint),
            ("Business Insights", self.test_insights_endpoint),
            ("Export", self.test_export_endpoint)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        print("=" * 50)
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
        
        if passed == len(results):
            print("🎉 All tests passed! Your API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed == len(results)


def main():
    """Main function to run the API tests"""
    print("AI Data Analysis Pipeline - API Testing Tool")
    print("=" * 50)
    
    # You can change the base URL here if your API is running on a different port
    base_url = "http://localhost:3032"
    
    print(f"Make sure your Flask app is running at: {base_url}")
    input("Press Enter to start testing...")
    
    tester = APITester(base_url)
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("Your AI Data Analysis Pipeline API is ready for production!")
    else:
        print("\n⚠️  Some tests failed. Please check your API implementation.")
    
    return success


if __name__ == "__main__":
    main()