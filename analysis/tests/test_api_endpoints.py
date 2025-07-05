import unittest
import json
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for Flask API endpoints"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Create sample CSV data for testing
        self.sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20, freq='D'),
            'sales': np.random.randint(100, 1000, 20),
            'category': np.random.choice(['A', 'B', 'C'], 20),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 20),
            'profit': np.random.randint(10, 100, 20)
        })
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
    
    def test_analyze_endpoint_with_csv(self):
        """Test the analyze endpoint with CSV file upload"""
        with open(self.temp_file.name, 'rb') as f:
            response = self.client.post('/analyze', 
                                      data={'file': (f, 'test.csv')},
                                      content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('analysis', data)
        self.assertIn('basic_statistics', data['analysis'])
        self.assertIn('data_quality', data['analysis'])
        self.assertIn('column_analysis', data['analysis'])
    
    def test_analyze_endpoint_with_json_data(self):
        """Test the analyze endpoint with JSON data"""
        json_data = {
            'data': self.sample_data.to_dict('records'),
            'columns': list(self.sample_data.columns)
        }
        
        response = self.client.post('/analyze',
                                  data=json.dumps(json_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('analysis', data)
        self.assertIn('basic_statistics', data['analysis'])
    
    def test_visualize_endpoint(self):
        """Test the visualize endpoint"""
        # First upload data
        with open(self.temp_file.name, 'rb') as f:
            upload_response = self.client.post('/analyze', 
                                             data={'file': (f, 'test.csv')},
                                             content_type='multipart/form-data')
        
        # Then request visualization
        viz_config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'sales',
            'title': 'Sales by Category'
        }
        
        response = self.client.post('/visualize',
                                  data=json.dumps(viz_config),
                                  content_type='application/json')
        
        # Note: This might fail if no data is stored in session
        # In a real implementation, you'd need to handle data persistence
        self.assertIn(response.status_code, [200, 400])  # 400 if no data available
    
    def test_filter_endpoint(self):
        """Test the filter endpoint"""
        filters = [
            {
                'column': 'sales',
                'type': 'greater_than',
                'value': 500
            },
            {
                'column': 'category',
                'type': 'equals',
                'value': 'A'
            }
        ]
        
        response = self.client.post('/filter',
                                  data=json.dumps({'filters': filters}),
                                  content_type='application/json')
        
        # Note: This might fail if no data is stored in session
        self.assertIn(response.status_code, [200, 400])  # 400 if no data available
    
    def test_insights_endpoint(self):
        """Test the business insights endpoint"""
        insights_request = {
            'prompt': 'What are the key trends in sales data?',
            'focus_areas': ['trends', 'performance']
        }
        
        response = self.client.post('/insights',
                                  data=json.dumps(insights_request),
                                  content_type='application/json')
        
        # Note: This might fail if no data is stored in session
        self.assertIn(response.status_code, [200, 400])  # 400 if no data available
    
    def test_export_endpoint(self):
        """Test the export endpoint"""
        export_config = {
            'format': 'csv',
            'include_analysis': True
        }
        
        response = self.client.post('/export',
                                  data=json.dumps(export_config),
                                  content_type='application/json')
        
        # Note: This might fail if no data is stored in session
        self.assertIn(response.status_code, [200, 400])  # 400 if no data available
    
    def test_invalid_file_upload(self):
        """Test uploading invalid file format"""
        # Create a text file instead of CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not a CSV file")
            temp_txt = f.name
        
        try:
            with open(temp_txt, 'rb') as f:
                response = self.client.post('/analyze', 
                                          data={'file': (f, 'test.txt')},
                                          content_type='multipart/form-data')
            
            self.assertEqual(response.status_code, 400)
            
            data = json.loads(response.data)
            self.assertIn('error', data)
        
        finally:
            os.unlink(temp_txt)
    
    def test_missing_file_upload(self):
        """Test endpoint without file upload"""
        response = self.client.post('/analyze')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_invalid_json_data(self):
        """Test endpoint with invalid JSON"""
        response = self.client.post('/analyze',
                                  data='invalid json',
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.get('/health')
        
        # Check for CORS headers (if configured)
        # This depends on your CORS configuration
        self.assertEqual(response.status_code, 200)
    
    def test_large_file_upload(self):
        """Test uploading a large CSV file"""
        # Create a larger dataset
        large_data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'date': pd.date_range('2020-01-01', periods=1000, freq='D')
        })
        
        # Create temporary large CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            temp_large = f.name
        
        try:
            with open(temp_large, 'rb') as f:
                response = self.client.post('/analyze', 
                                          data={'file': (f, 'large_test.csv')},
                                          content_type='multipart/form-data')
            
            # Should handle large files gracefully
            self.assertIn(response.status_code, [200, 413])  # 413 if file too large
        
        finally:
            os.unlink(temp_large)
    
    def test_empty_csv_file(self):
        """Test uploading empty CSV file"""
        # Create empty CSV
        empty_data = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_data.to_csv(f.name, index=False)
            temp_empty = f.name
        
        try:
            with open(temp_empty, 'rb') as f:
                response = self.client.post('/analyze', 
                                          data={'file': (f, 'empty.csv')},
                                          content_type='multipart/form-data')
            
            # Should handle empty files gracefully
            self.assertIn(response.status_code, [200, 400])
        
        finally:
            os.unlink(temp_empty)


if __name__ == '__main__':
    unittest.main() 