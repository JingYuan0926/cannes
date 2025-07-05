import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization_engine import VisualizationEngine


class TestVisualizationEngine(unittest.TestCase):
    """Test cases for VisualizationEngine class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.viz_engine = VisualizationEngine()
        
        # Create sample datasets for testing
        self.sample_data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
            'values': [10, 15, 20, 12, 18, 25, 14, 16, 22, 11],
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South']
        })
        
        self.numerical_data = pd.DataFrame({
            'x': range(20),
            'y': np.random.randn(20) * 10 + 50,
            'z': np.random.randn(20) * 5 + 25
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
            'y_axis': 'values',
            'title': 'Test Bar Chart'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        self.assertIn('insights', result)
        self.assertIn('chart_type', result)
        self.assertEqual(result['chart_type'], 'bar')
    
    def test_create_line_chart(self):
        """Test line chart creation"""
        config = {
            'chart_type': 'line',
            'x_axis': 'date',
            'y_axis': 'sales',
            'title': 'Sales Over Time'
        }
        
        result = self.viz_engine.create_visualization(self.time_series_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        self.assertIn('insights', result)
        self.assertEqual(result['chart_type'], 'line')
    
    def test_create_scatter_plot(self):
        """Test scatter plot creation"""
        config = {
            'chart_type': 'scatter',
            'x_axis': 'x',
            'y_axis': 'y',
            'title': 'Scatter Plot Test'
        }
        
        result = self.viz_engine.create_visualization(self.numerical_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        self.assertIn('insights', result)
        self.assertEqual(result['chart_type'], 'scatter')
    
    def test_create_pie_chart(self):
        """Test pie chart creation"""
        config = {
            'chart_type': 'pie',
            'values': 'values',
            'labels': 'category',
            'title': 'Category Distribution'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        self.assertIn('insights', result)
        self.assertEqual(result['chart_type'], 'pie')
    
    def test_create_histogram(self):
        """Test histogram creation"""
        config = {
            'chart_type': 'histogram',
            'x_axis': 'y',
            'title': 'Distribution of Y Values'
        }
        
        result = self.viz_engine.create_visualization(self.numerical_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        self.assertIn('insights', result)
        self.assertEqual(result['chart_type'], 'histogram')
    
    def test_create_box_plot(self):
        """Test box plot creation"""
        config = {
            'chart_type': 'box',
            'y_axis': 'values',
            'x_axis': 'category',
            'title': 'Values by Category'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        self.assertIn('insights', result)
        self.assertEqual(result['chart_type'], 'box')
    
    def test_create_heatmap(self):
        """Test heatmap creation"""
        config = {
            'chart_type': 'heatmap',
            'title': 'Correlation Heatmap'
        }
        
        result = self.viz_engine.create_visualization(self.numerical_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        self.assertIn('insights', result)
        self.assertEqual(result['chart_type'], 'heatmap')
    
    def test_auto_chart_selection(self):
        """Test automatic chart type selection"""
        # Test with different data types
        result = self.viz_engine.auto_select_chart_type(self.sample_data, 'category', 'values')
        self.assertIn(result, ['bar', 'column'])
        
        result = self.viz_engine.auto_select_chart_type(self.time_series_data, 'date', 'sales')
        self.assertEqual(result, 'line')
        
        result = self.viz_engine.auto_select_chart_type(self.numerical_data, 'x', 'y')
        self.assertEqual(result, 'scatter')
    
    def test_chart_insights_generation(self):
        """Test chart insights generation"""
        # Test bar chart insights
        insights = self.viz_engine._generate_chart_insights(self.sample_data, 'bar', 'category', 'values')
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        # Test line chart insights
        insights = self.viz_engine._generate_chart_insights(self.time_series_data, 'line', 'date', 'sales')
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
    
    def test_create_dashboard(self):
        """Test dashboard creation"""
        charts = [
            {
                'chart_type': 'bar',
                'x_axis': 'category',
                'y_axis': 'values',
                'title': 'Values by Category'
            },
            {
                'chart_type': 'line',
                'x_axis': 'date',
                'y_axis': 'values',
                'title': 'Values Over Time'
            }
        ]
        
        result = self.viz_engine.create_dashboard(self.sample_data, charts)
        
        self.assertIsInstance(result, dict)
        self.assertIn('dashboard_html', result)
        self.assertIn('charts', result)
        self.assertEqual(len(result['charts']), 2)
    
    def test_invalid_chart_type(self):
        """Test handling of invalid chart type"""
        config = {
            'chart_type': 'invalid_type',
            'x_axis': 'category',
            'y_axis': 'values'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        # Should handle gracefully and return error or default chart
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns"""
        config = {
            'chart_type': 'bar',
            'x_axis': 'nonexistent_column',
            'y_axis': 'values'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        # Should handle gracefully and return error
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'values'
        }
        
        result = self.viz_engine.create_visualization(empty_df, config)
        
        # Should handle gracefully and return error
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
    
    def test_chart_styling(self):
        """Test chart styling options"""
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'values',
            'title': 'Styled Chart',
            'color_scheme': 'viridis',
            'width': 800,
            'height': 600
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        # Check if styling options are applied (this depends on implementation)
    
    def test_multiple_series_chart(self):
        """Test chart with multiple data series"""
        config = {
            'chart_type': 'line',
            'x_axis': 'date',
            'y_axis': ['sales', 'profit'],
            'title': 'Sales and Profit Over Time'
        }
        
        result = self.viz_engine.create_visualization(self.time_series_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        self.assertIn('insights', result)
    
    def test_chart_with_grouping(self):
        """Test chart with grouping/color coding"""
        config = {
            'chart_type': 'scatter',
            'x_axis': 'values',
            'y_axis': 'values',
            'color_by': 'category',
            'title': 'Grouped Scatter Plot'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
    
    def test_chart_data_validation(self):
        """Test data validation for chart creation"""
        # Test with non-numeric data for numeric chart
        invalid_data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'text_values': ['low', 'medium', 'high']
        })
        
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'text_values'
        }
        
        result = self.viz_engine.create_visualization(invalid_data, config)
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)
        # May return error or attempt to convert/handle the data
    
    def test_chart_export_formats(self):
        """Test different chart export formats"""
        config = {
            'chart_type': 'bar',
            'x_axis': 'category',
            'y_axis': 'values',
            'title': 'Export Test',
            'export_format': 'png'
        }
        
        result = self.viz_engine.create_visualization(self.sample_data, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chart_html', result)
        # Check if export format is handled (depends on implementation)


if __name__ == '__main__':
    unittest.main() 