import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.visualization_engine import VisualizationEngine

class TestVisualizationEngine(unittest.TestCase):
    """Test cases for VisualizationEngine class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.viz_engine = VisualizationEngine()
        
        # Create sample datasets for testing
        self.sample_data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
            'value': [10, 20, 15, 25, 12, 18, 17, 22],
            'date': pd.date_range('2023-01-01', periods=8, freq='D'),
            'region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West']
        })
        
        self.numerical_data = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50),
            'z': np.random.randn(50)
        })
        
        self.time_series_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30, freq='D'),
            'sales': np.random.randint(100, 1000, 30),
            'profit': np.random.randint(10, 100, 30)
        })
    
    def test_create_bar_chart(self):
        """Test bar chart creation"""
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Value by Category'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('chart_type', result)
        self.assertEqual(result['chart_type'], 'bar')
        self.assertIn('config', result)
    
    def test_create_line_chart(self):
        """Test line chart creation"""
        config = {
            'chart_type': 'line',
            'x_axis': 'date',
            'y_axis': 'sales',
            'title': 'Sales Over Time'
        }
        
        result = self.viz_engine.create_visualization(self.time_series_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('chart_type', result)
        self.assertEqual(result['chart_type'], 'line')
    
    def test_create_scatter_plot(self):
        """Test scatter plot creation"""
        config = {
            'chart_type': 'scatter',
            'x_axis': 'x',
            'y_axis': 'y',
            'title': 'X vs Y Scatter Plot'
        }
        
        result = self.viz_engine.create_visualization(self.numerical_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('chart_type', result)
        self.assertEqual(result['chart_type'], 'scatter')
    
    def test_create_pie_chart(self):
        """Test pie chart creation"""
        config = {
            'chart_type': 'pie',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Category Distribution'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('chart_type', result)
        self.assertEqual(result['chart_type'], 'pie')
    
    def test_create_histogram(self):
        """Test histogram creation"""
        config = {
            'chart_type': 'histogram',
            'x_axis': 'y',
            'title': 'Distribution of Y Values'
        }
        
        result = self.viz_engine.create_visualization(self.numerical_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('chart_type', result)
        self.assertEqual(result['chart_type'], 'histogram')
    
    def test_create_box_plot(self):
        """Test box plot creation"""
        config = {
            'chart_type': 'box',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Value Distribution by Category'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('chart_type', result)
        self.assertEqual(result['chart_type'], 'box')
    
    def test_create_heatmap(self):
        """Test heatmap creation"""
        config = {
            'chart_type': 'heatmap',
            'title': 'Correlation Heatmap'
        }
        
        result = self.viz_engine.create_visualization(self.numerical_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('chart_type', result)
        self.assertEqual(result['chart_type'], 'heatmap')
    
    def test_auto_chart_selection(self):
        """Test automatic chart type selection"""
        config = {
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Auto Selected Chart'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('chart_type', result)
        # Should auto-select bar chart for categorical vs numerical
        self.assertEqual(result['chart_type'], 'bar')
    
    def test_generate_insights(self):
        """Test chart insights generation"""
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Value by Category'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('insights', result)
        self.assertIsInstance(result['insights'], list)
    
    def test_create_dashboard(self):
        """Test dashboard creation with multiple charts"""
        charts = [
            {
                'chart_type': 'bar',
                'x_axis': 'category',
                'y_axis': 'value',
                'title': 'Value by Category'
            },
            {
                'chart_type': 'line',
                'x_axis': 'date',
                'y_axis': 'value',
                'title': 'Value Over Time'
            }
        ]
        
        dashboard = self.viz_engine.create_dashboard(self.sample_data, charts)
        
        self.assertIsNotNone(dashboard)
        self.assertIn('charts', dashboard)
        self.assertEqual(len(dashboard['charts']), 2)
    
    def test_invalid_chart_type(self):
        """Test handling of invalid chart type"""
        config = {
            'chart_type': 'invalid_type',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Invalid Chart'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        # Should handle gracefully or return None
        self.assertIsNone(result)
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns"""
        config = {
            'chart_type': 'bar',
            'x_axis': 'nonexistent_column',
            'y_axis': 'value',
            'title': 'Missing Column Chart'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        # Should handle gracefully or return None
        self.assertIsNone(result)
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Empty Data Chart'
        }
        
        result = self.viz_engine.create_visualization(empty_df, config)
        
        # Should handle gracefully or return None
        self.assertIsNone(result)
    
    def test_data_validation(self):
        """Test data validation for chart creation"""
        # Test with invalid data types
        invalid_data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': ['not_a_number', 'also_not_a_number', 'still_not_a_number']
        })
        
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Invalid Data Chart'
        }
        
        result = self.viz_engine.create_visualization(invalid_data, config)
        
        # Should handle gracefully
        self.assertIsNone(result)
    
    def test_chart_styling(self):
        """Test chart styling options"""
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Styled Chart',
            'style': {
                'color_scheme': 'viridis',
                'width': 800,
                'height': 600
            }
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsNotNone(result)
        self.assertIn('style', result)
    
    def test_export_formats(self):
        """Test different export formats"""
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'value',
            'title': 'Export Test Chart'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        # Test different export formats
        formats = ['png', 'svg', 'html', 'json']
        for fmt in formats:
            export_result = self.viz_engine.export_chart(result, fmt)
            self.assertIsNotNone(export_result)

if __name__ == '__main__':
    unittest.main() 