"""
Enhanced Image Analyzer Utility for EDA Analysis System
Analyzes generated plot images and provides detailed, data-driven explanations for each visualization type.

This utility reads the actual JSON plot data to provide specific insights with real numbers,
percentages, and statistical interpretations rather than generic descriptions.

Supported chart types:
- Histograms: Distribution analysis with actual bin counts and percentages
- Bar charts: Comparative analysis with specific values and rankings
- Scatter plots: Correlation analysis with actual correlation coefficients
- Heatmaps: Correlation matrix interpretation with specific values
- Box plots: Statistical summary with quartiles, outliers, and ranges
- Pie charts: Proportional analysis with exact percentages

Key features:
- Reads JSON plot data for accurate interpretations
- Provides specific statistical insights
- Calculates percentages and comparative metrics
- Identifies patterns, trends, and outliers
- Generates comprehensive HTML reports

Created: 2025-07-05
"""

import os
import json
import logging
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """
    Enhanced analyzer that reads JSON plot data to provide specific, data-driven insights
    """
    
    def __init__(self, images_dir: str = "images", plots_dir: str = "plots"):
        self.images_dir = images_dir
        self.plots_dir = plots_dir
        
    def _load_json_data(self, image_filename: str) -> Optional[Dict]:
        """Load corresponding JSON data for an image"""
        try:
            # Convert image filename to JSON filename
            json_filename = image_filename.replace('.png', '.json')
            json_path = os.path.join(self.plots_dir, json_filename)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    plot_file_data = json.load(f)
                
                # Extract the chart_json string and parse it
                chart_json_str = plot_file_data.get('chart_json', '')
                if chart_json_str:
                    try:
                        # Parse the chart_json string to get the actual plot data
                        chart_data = json.loads(chart_json_str)
                        return chart_data
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing chart_json for {image_filename}: {e}")
                        return None
                else:
                    logger.warning(f"No chart_json found in {json_path}")
                    return None
            else:
                logger.warning(f"JSON file not found: {json_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading JSON data for {image_filename}: {e}")
            return None
    
    def _analyze_histogram(self, data: Dict, filename: str) -> str:
        """Analyze histogram data and provide specific insights"""
        try:
            plot_data = data.get('data', [{}])[0]
            x_values = plot_data.get('x', [])
            y_values = plot_data.get('y', [])
            
            if not x_values or not y_values:
                return "Histogram data not available for detailed analysis."
            
            total_count = sum(y_values)
            max_count = max(y_values)
            max_bin_idx = y_values.index(max_count)
            
            # Calculate percentages
            percentages = [(count/total_count)*100 for count in y_values]
            max_percentage = max(percentages)
            
            # Find range information
            if isinstance(x_values[0], (int, float)):
                min_val = min(x_values)
                max_val = max(x_values)
                range_val = max_val - min_val
            else:
                min_val = max_val = range_val = "N/A"
            
            analysis = f"""
            📊 **Distribution Analysis:**
            • **Total observations:** {total_count:,}
            • **Data range:** {min_val} to {max_val} (span: {range_val})
            • **Most frequent bin:** Contains {max_count:,} observations ({max_percentage:.1f}% of data)
            • **Distribution shape:** {'Right-skewed' if max_bin_idx < len(y_values)//2 else 'Left-skewed' if max_bin_idx > len(y_values)//2 else 'Roughly symmetric'}
            
            **Key Insights:**
            • {max_percentage:.1f}% of data falls in the peak range
            • {'Concentration in lower values suggests most data points are below average' if max_bin_idx < len(y_values)//2 else 'Even distribution across the range' if abs(max_bin_idx - len(y_values)//2) < 2 else 'Concentration in higher values indicates above-average clustering'}
            """
            
            return analysis.strip()
        except Exception as e:
            return f"Error analyzing histogram: {e}"
    
    def _analyze_bar_chart(self, data: Dict, filename: str) -> str:
        """Analyze bar chart data and provide specific insights"""
        try:
            plot_data = data.get('data', [{}])[0]
            x_values = plot_data.get('x', [])
            y_values = plot_data.get('y', [])
            
            if not x_values or not y_values:
                return "Bar chart data not available for detailed analysis."
            
            # Convert to numeric if possible
            try:
                y_numeric = [float(y) for y in y_values]
            except:
                y_numeric = y_values
            
            total = sum(y_numeric) if all(isinstance(y, (int, float)) for y in y_numeric) else len(y_values)
            
            # Find top performers
            sorted_pairs = sorted(zip(x_values, y_numeric), key=lambda x: x[1], reverse=True)
            top_category = sorted_pairs[0]
            bottom_category = sorted_pairs[-1]
            
            # Calculate percentages
            if isinstance(y_numeric[0], (int, float)):
                top_percentage = (top_category[1] / total) * 100
                bottom_percentage = (bottom_category[1] / total) * 100
                avg_value = total / len(y_numeric)
                
                analysis = f"""
                📊 **Comparative Analysis:**
                • **Top performer:** {top_category[0]} with {top_category[1]:,.0f} ({top_percentage:.1f}% of total)
                • **Lowest performer:** {bottom_category[0]} with {bottom_category[1]:,.0f} ({bottom_percentage:.1f}% of total)
                • **Average value:** {avg_value:,.0f}
                • **Performance gap:** {top_category[1] - bottom_category[1]:,.0f} ({((top_category[1] - bottom_category[1])/bottom_category[1]*100):.0f}% difference)
                
                **Key Insights:**
                • {top_category[0]} outperforms the average by {((top_category[1] - avg_value)/avg_value*100):+.0f}%
                • Top 3 categories: {', '.join([f"{pair[0]} ({pair[1]:,.0f})" for pair in sorted_pairs[:3]])}
                • {'High variation in performance across categories' if (top_category[1] - bottom_category[1]) > avg_value else 'Relatively consistent performance across categories'}
                """
            else:
                analysis = f"""
                📊 **Category Analysis:**
                • **Categories analyzed:** {len(x_values)}
                • **Top category:** {top_category[0]}
                • **Full ranking:** {', '.join([pair[0] for pair in sorted_pairs])}
                """
            
            return analysis.strip()
        except Exception as e:
            return f"Error analyzing bar chart: {e}"
    
    def _analyze_scatter_plot(self, data: Dict, filename: str) -> str:
        """Analyze scatter plot data and provide correlation insights"""
        try:
            plot_data = data.get('data', [{}])[0]
            x_values = plot_data.get('x', [])
            y_values = plot_data.get('y', [])
            
            if not x_values or not y_values or len(x_values) != len(y_values):
                return "Scatter plot data not available for detailed analysis."
            
            # Calculate correlation
            try:
                correlation = np.corrcoef(x_values, y_values)[0, 1]
                correlation_strength = abs(correlation)
                
                if correlation_strength > 0.8:
                    strength_desc = "very strong"
                elif correlation_strength > 0.6:
                    strength_desc = "strong"
                elif correlation_strength > 0.4:
                    strength_desc = "moderate"
                elif correlation_strength > 0.2:
                    strength_desc = "weak"
                else:
                    strength_desc = "very weak"
                
                direction = "positive" if correlation > 0 else "negative"
                
                # Calculate ranges and outliers
                x_range = max(x_values) - min(x_values)
                y_range = max(y_values) - min(y_values)
                
                analysis = f"""
                📊 **Correlation Analysis:**
                • **Correlation coefficient:** {correlation:.3f}
                • **Relationship strength:** {strength_desc.title()} {direction} correlation
                • **Data points:** {len(x_values):,} observations
                • **X-axis range:** {min(x_values):.1f} to {max(x_values):.1f}
                • **Y-axis range:** {min(y_values):.1f} to {max(y_values):.1f}
                
                **Key Insights:**
                • {correlation*100:.1f}% of variation in Y is explained by X
                • {'As X increases, Y tends to increase significantly' if correlation > 0.6 else 'As X increases, Y tends to decrease significantly' if correlation < -0.6 else 'X and Y show limited linear relationship'}
                • {'Strong predictive relationship - X is a good predictor of Y' if abs(correlation) > 0.7 else 'Moderate relationship - other factors may influence Y' if abs(correlation) > 0.4 else 'Weak relationship - X is not a strong predictor of Y'}
                """
                
                return analysis.strip()
            except Exception as e:
                return f"Error calculating correlation: {e}"
        except Exception as e:
            return f"Error analyzing scatter plot: {e}"
    
    def _analyze_heatmap(self, data: Dict, filename: str) -> str:
        """Analyze heatmap correlation matrix"""
        try:
            plot_data = data.get('data', [{}])[0]
            z_values = plot_data.get('z', [])
            x_labels = plot_data.get('x', [])
            y_labels = plot_data.get('y', [])
            
            if not z_values:
                return "Heatmap data not available for detailed analysis."
            
            # Flatten correlation matrix to find strongest correlations
            correlations = []
            for i, row in enumerate(z_values):
                for j, val in enumerate(row):
                    if i != j:  # Exclude diagonal (self-correlation)
                        correlations.append((abs(val), val, y_labels[i] if i < len(y_labels) else f"Var{i}", 
                                          x_labels[j] if j < len(x_labels) else f"Var{j}"))
            
            # Sort by absolute correlation strength
            correlations.sort(reverse=True)
            
            strongest = correlations[0] if correlations else None
            
            if strongest:
                analysis = f"""
                📊 **Correlation Matrix Analysis:**
                • **Variables analyzed:** {len(x_labels)} variables
                • **Strongest correlation:** {strongest[2]} ↔ {strongest[3]} (r = {strongest[1]:.3f})
                • **Correlation strength:** {'Very Strong' if strongest[0] > 0.8 else 'Strong' if strongest[0] > 0.6 else 'Moderate' if strongest[0] > 0.4 else 'Weak'}
                
                **Top 3 Correlations:**
                """
                
                for i, (abs_corr, corr, var1, var2) in enumerate(correlations[:3]):
                    direction = "positively" if corr > 0 else "negatively"
                    analysis += f"\n• {var1} & {var2}: {direction} correlated (r = {corr:.3f})"
                
                # Count strong correlations
                strong_count = sum(1 for corr in correlations if corr[0] > 0.6)
                analysis += f"\n\n**Key Insights:**\n• {strong_count} pairs show strong correlations (|r| > 0.6)"
                analysis += f"\n• {'High multicollinearity detected - variables are highly interdependent' if strong_count > len(x_labels)//2 else 'Moderate variable independence' if strong_count > 2 else 'Variables are largely independent'}"
                
                return analysis.strip()
            else:
                return "Unable to analyze correlation patterns in heatmap."
        except Exception as e:
            return f"Error analyzing heatmap: {e}"
    
    def _analyze_box_plot(self, data: Dict, filename: str) -> str:
        """Analyze box plot statistical summary"""
        try:
            plot_data = data.get('data', [{}])[0]
            y_values = plot_data.get('y', [])
            
            if not y_values:
                return "Box plot data not available for detailed analysis."
            
            # Calculate quartiles and statistics
            q1 = np.percentile(y_values, 25)
            q2 = np.percentile(y_values, 50)  # Median
            q3 = np.percentile(y_values, 75)
            iqr = q3 - q1
            
            # Identify outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [y for y in y_values if y < lower_bound or y > upper_bound]
            
            mean_val = np.mean(y_values)
            std_val = np.std(y_values)
            
            analysis = f"""
            📊 **Statistical Distribution Analysis:**
            • **Sample size:** {len(y_values):,} observations
            • **Median (Q2):** {q2:.2f}
            • **Mean:** {mean_val:.2f}
            • **First Quartile (Q1):** {q1:.2f}
            • **Third Quartile (Q3):** {q3:.2f}
            • **Interquartile Range (IQR):** {iqr:.2f}
            • **Standard Deviation:** {std_val:.2f}
            
            **Distribution Insights:**
            • **Outliers detected:** {len(outliers)} ({(len(outliers)/len(y_values)*100):.1f}% of data)
            • **Skewness:** {'Right-skewed (mean > median)' if mean_val > q2 else 'Left-skewed (mean < median)' if mean_val < q2 else 'Roughly symmetric'}
            • **Spread:** {'High variability' if std_val > iqr else 'Moderate variability' if std_val > iqr/2 else 'Low variability'}
            • **Middle 50% range:** {q1:.2f} to {q3:.2f}
            """
            
            if outliers:
                analysis += f"\n• **Extreme values:** {min(outliers):.2f} (lowest) to {max(outliers):.2f} (highest)"
            
            return analysis.strip()
        except Exception as e:
            return f"Error analyzing box plot: {e}"
    
    def _analyze_pie_chart(self, data: Dict, filename: str) -> str:
        """Analyze pie chart proportional data"""
        try:
            plot_data = data.get('data', [{}])[0]
            labels = plot_data.get('labels', [])
            values = plot_data.get('values', [])
            
            if not labels or not values:
                return "Pie chart data not available for detailed analysis."
            
            total = sum(values)
            percentages = [(val/total)*100 for val in values]
            
            # Sort by percentage
            sorted_data = sorted(zip(labels, values, percentages), key=lambda x: x[2], reverse=True)
            
            dominant_segment = sorted_data[0]
            smallest_segment = sorted_data[-1]
            
            analysis = f"""
            📊 **Proportional Analysis:**
            • **Total segments:** {len(labels)}
            • **Total value:** {total:,.0f}
            • **Dominant segment:** {dominant_segment[0]} ({dominant_segment[2]:.1f}%)
            • **Smallest segment:** {smallest_segment[0]} ({smallest_segment[2]:.1f}%)
            
            **Segment Breakdown:**
            """
            
            for label, value, percentage in sorted_data:
                analysis += f"\n• {label}: {value:,.0f} ({percentage:.1f}%)"
            
            # Calculate concentration
            top_3_percentage = sum([item[2] for item in sorted_data[:3]])
            
            analysis += f"""
            
            **Key Insights:**
            • Top 3 segments represent {top_3_percentage:.1f}% of total
            • {'Highly concentrated - dominated by few segments' if top_3_percentage > 80 else 'Moderately concentrated' if top_3_percentage > 60 else 'Well distributed across segments'}
            • Largest segment is {dominant_segment[2]/smallest_segment[2]:.1f}x larger than smallest
            """
            
            return analysis.strip()
        except Exception as e:
            return f"Error analyzing pie chart: {e}"
    
    def _analyze_line_chart(self, data: Dict, filename: str) -> str:
        """Analyze line chart trend data"""
        try:
            plot_data = data.get('data', [{}])[0]
            x_values = plot_data.get('x', [])
            y_values = plot_data.get('y', [])
            
            if not x_values or not y_values:
                return "Line chart data not available for detailed analysis."
            
            # Calculate trend metrics
            if len(y_values) < 2:
                return "Insufficient data points for trend analysis."
            
            # Calculate overall trend
            first_value = y_values[0]
            last_value = y_values[-1]
            total_change = last_value - first_value
            percent_change = (total_change / first_value) * 100 if first_value != 0 else 0
            
            # Find peaks and valleys
            max_value = max(y_values)
            min_value = min(y_values)
            max_idx = y_values.index(max_value)
            min_idx = y_values.index(min_value)
            
            # Calculate volatility (standard deviation)
            mean_value = np.mean(y_values)
            std_value = np.std(y_values)
            volatility = (std_value / mean_value) * 100 if mean_value != 0 else 0
            
            analysis = f"""
            📈 **Trend Analysis:**
            • **Data points:** {len(y_values):,} observations
            • **Overall trend:** {'Upward' if total_change > 0 else 'Downward' if total_change < 0 else 'Flat'} ({percent_change:+.1f}%)
            • **Starting value:** {first_value:,.2f}
            • **Ending value:** {last_value:,.2f}
            • **Peak value:** {max_value:,.2f} (position {max_idx + 1})
            • **Valley value:** {min_value:,.2f} (position {min_idx + 1})
            • **Average value:** {mean_value:,.2f}
            • **Volatility:** {volatility:.1f}%
            
            **Key Insights:**
            • {'Strong upward momentum' if percent_change > 20 else 'Moderate growth' if percent_change > 5 else 'Slight decline' if percent_change < -5 else 'Strong decline' if percent_change < -20 else 'Stable trend'}
            • {'High volatility - significant fluctuations' if volatility > 30 else 'Moderate volatility' if volatility > 15 else 'Low volatility - stable pattern'}
            • Range span: {max_value - min_value:,.2f} ({((max_value - min_value) / mean_value * 100):.1f}% of average)
            """
            
            return analysis.strip()
        except Exception as e:
            return f"Error analyzing line chart: {e}"
    
    def _analyze_area_chart(self, data: Dict, filename: str) -> str:
        """Analyze area chart data (similar to line chart but with emphasis on volume)"""
        try:
            plot_data = data.get('data', [{}])[0]
            x_values = plot_data.get('x', [])
            y_values = plot_data.get('y', [])
            
            if not x_values or not y_values:
                return "Area chart data not available for detailed analysis."
            
            # Calculate area under curve (approximation using trapezoidal rule)
            if len(y_values) < 2:
                return "Insufficient data points for area analysis."
            
            # Calculate cumulative metrics
            total_area = sum(y_values)  # Simplified area calculation
            avg_height = np.mean(y_values)
            
            # Find growth periods
            increasing_periods = 0
            decreasing_periods = 0
            for i in range(1, len(y_values)):
                if y_values[i] > y_values[i-1]:
                    increasing_periods += 1
                elif y_values[i] < y_values[i-1]:
                    decreasing_periods += 1
            
            # Calculate quartiles for area distribution
            q1 = np.percentile(y_values, 25)
            q2 = np.percentile(y_values, 50)
            q3 = np.percentile(y_values, 75)
            
            analysis = f"""
            📊 **Area Analysis:**
            • **Total area:** {total_area:,.0f} units
            • **Average height:** {avg_height:,.2f}
            • **Data points:** {len(y_values):,} observations
            • **Peak value:** {max(y_values):,.2f}
            • **Minimum value:** {min(y_values):,.2f}
            • **Median height:** {q2:.2f}
            
            **Growth Pattern:**
            • **Increasing periods:** {increasing_periods} ({(increasing_periods/(len(y_values)-1)*100):.1f}% of time)
            • **Decreasing periods:** {decreasing_periods} ({(decreasing_periods/(len(y_values)-1)*100):.1f}% of time)
            • **Stable periods:** {len(y_values)-1-increasing_periods-decreasing_periods}
            
            **Key Insights:**
            • {'Predominantly growing area' if increasing_periods > decreasing_periods else 'Predominantly declining area' if decreasing_periods > increasing_periods else 'Mixed growth pattern'}
            • Area distribution: 25% below {q1:.2f}, 50% below {q2:.2f}, 75% below {q3:.2f}
            • {'High concentration in upper range' if q3 > 1.5 * avg_height else 'Evenly distributed values' if q1 > 0.5 * avg_height else 'Concentration in lower range'}
            """
            
            return analysis.strip()
        except Exception as e:
            return f"Error analyzing area chart: {e}"
    
    def detect_plot_type(self, filename: str) -> str:
        """Detect plot type from filename"""
        filename_lower = filename.lower()
        
        if 'histogram' in filename_lower or 'distribution' in filename_lower:
            return 'histogram'
        elif 'bar' in filename_lower:
            return 'bar'
        elif 'scatter' in filename_lower:
            return 'scatter'
        elif 'heatmap' in filename_lower or 'correlation' in filename_lower:
            return 'heatmap'
        elif 'box' in filename_lower:
            return 'box'
        elif 'pie' in filename_lower:
            return 'pie'
        elif 'line' in filename_lower:
            return 'line'
        elif 'area' in filename_lower:
            return 'area'
        else:
            return 'unknown'
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get basic image information"""
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'size_mb': os.path.getsize(image_path) / (1024 * 1024)
                }
        except Exception as e:
            logger.error(f"Error getting image info for {image_path}: {e}")
            return {}
    
    def analyze_single_image(self, image_filename: str) -> Dict[str, Any]:
        """Analyze a single image with data-driven insights"""
        image_path = os.path.join(self.images_dir, image_filename)
        
        if not os.path.exists(image_path):
            return {
                'filename': image_filename,
                'error': 'Image file not found',
                'analysis': 'Unable to analyze - file not found'
            }
        
        # Get image info
        image_info = self.get_image_info(image_path)
        
        # Detect plot type
        plot_type = self.detect_plot_type(image_filename)
        
        # Load JSON data
        json_data = self._load_json_data(image_filename)
        
        # Generate analysis based on actual data
        if json_data:
            if plot_type == 'histogram':
                analysis = self._analyze_histogram(json_data, image_filename)
            elif plot_type == 'bar':
                analysis = self._analyze_bar_chart(json_data, image_filename)
            elif plot_type == 'scatter':
                analysis = self._analyze_scatter_plot(json_data, image_filename)
            elif plot_type == 'heatmap':
                analysis = self._analyze_heatmap(json_data, image_filename)
            elif plot_type == 'box':
                analysis = self._analyze_box_plot(json_data, image_filename)
            elif plot_type == 'pie':
                analysis = self._analyze_pie_chart(json_data, image_filename)
            elif plot_type == 'line':
                analysis = self._analyze_line_chart(json_data, image_filename)
            elif plot_type == 'area':
                analysis = self._analyze_area_chart(json_data, image_filename)
            else:
                analysis = "Unknown plot type - unable to provide specific analysis"
        else:
            analysis = f"JSON data not available for {image_filename}. Cannot provide data-driven insights."
        
        return {
            'filename': image_filename,
            'plot_type': plot_type,
            'image_info': image_info,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_all_images(self, images_dir: str = None) -> Dict[str, Any]:
        """Analyze all images in the directory"""
        if images_dir:
            self.images_dir = images_dir
        
        if not os.path.exists(self.images_dir):
            return {
                'error': f'Images directory not found: {self.images_dir}',
                'analyses': []
            }
        
        # Get all PNG files
        image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith('.png')]
        
        if not image_files:
            return {
                'message': f'No PNG images found in {self.images_dir}',
                'analyses': []
            }
        
        # Analyze each image
        analyses = []
        for image_file in sorted(image_files):
            analysis = self.analyze_single_image(image_file)
            analyses.append(analysis)
        
        # Generate summary
        total_images = len(analyses)
        successful_analyses = len([a for a in analyses if 'error' not in a])
        
        return {
            'summary': {
                'total_images': total_images,
                'successful_analyses': successful_analyses,
                'images_directory': self.images_dir,
                'plots_directory': self.plots_dir,
                'timestamp': datetime.now().isoformat()
            },
            'analyses': analyses
        }
    
    def create_analysis_report(self, analyses_result: Dict[str, Any], output_path: str = None) -> str:
        """Create HTML report with all analyses"""
        if not output_path:
            output_path = os.path.join(self.images_dir, 'enhanced_analysis_report.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced EDA Image Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .summary {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .analysis-card {{
                    background: white;
                    margin-bottom: 30px;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .analysis-header {{
                    background: #4a90e2;
                    color: white;
                    padding: 20px;
                    font-size: 1.2em;
                    font-weight: bold;
                }}
                .analysis-content {{
                    padding: 20px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .analysis-text {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                    white-space: pre-line;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                }}
                .plot-type {{
                    display: inline-block;
                    background: #28a745;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 0.8em;
                    margin: 10px 0;
                }}
                .error {{
                    background: #dc3545;
                    color: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 0.8em;
                    text-align: right;
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Enhanced EDA Image Analysis Report</h1>
                <p>Data-Driven Insights from Generated Visualizations</p>
            </div>
        """
        
        # Add summary
        summary = analyses_result.get('summary', {})
        html_content += f"""
            <div class="summary">
                <h2>📊 Analysis Summary</h2>
                <p><strong>Total Images:</strong> {summary.get('total_images', 0)}</p>
                <p><strong>Successful Analyses:</strong> {summary.get('successful_analyses', 0)}</p>
                <p><strong>Images Directory:</strong> {summary.get('images_directory', 'N/A')}</p>
                <p><strong>Plots Directory:</strong> {summary.get('plots_directory', 'N/A')}</p>
                <p><strong>Generated:</strong> {summary.get('timestamp', 'N/A')}</p>
            </div>
        """
        
        # Add individual analyses
        for analysis in analyses_result.get('analyses', []):
            filename = analysis.get('filename', 'Unknown')
            plot_type = analysis.get('plot_type', 'unknown')
            analysis_text = analysis.get('analysis', 'No analysis available')
            
            html_content += f"""
            <div class="analysis-card">
                <div class="analysis-header">
                    📈 {filename}
                    <span class="plot-type">{plot_type.upper()}</span>
                </div>
                <div class="analysis-content">
                    <div class="image-container">
                        <img src="{filename}" alt="{filename}">
                    </div>
                    <div class="analysis-text">{analysis_text}</div>
                    <div class="timestamp">
                        Analysis generated: {analysis.get('timestamp', 'N/A')}
                    </div>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_path
        except Exception as e:
            logger.error(f"Error creating HTML report: {e}")
            return None

# CLI usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_analyzer.py <images_directory> [plots_directory]")
        sys.exit(1)
    
    images_dir = sys.argv[1]
    plots_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    
    analyzer = ImageAnalyzer(images_dir, plots_dir)
    result = analyzer.analyze_all_images()
    
    if result.get('analyses'):
        report_path = analyzer.create_analysis_report(result)
        print(f"Analysis complete! Report saved to: {report_path}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Analyzed {result['summary']['total_images']} images")
        print(f"- {result['summary']['successful_analyses']} successful analyses")
    else:
        print("No images found or analysis failed.") 