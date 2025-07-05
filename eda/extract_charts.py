#!/usr/bin/env python3
"""
Extract Charts from Analysis Results

This script extracts all generated charts from the analysis results
and saves them as individual HTML files that can be opened in a browser.
"""

import json
import os
from datetime import datetime

def extract_charts_from_json(json_file='analysis_results.json'):
    """
    Extract all charts from analysis results JSON file
    
    Args:
        json_file (str): Path to the analysis results JSON file
    """
    try:
        # Load the analysis results
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create charts directory if it doesn't exist
        charts_dir = 'charts'
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        # Extract visualizations
        visualizations = data.get('analysis', {}).get('visualizations', [])
        
        if not visualizations:
            print("No visualizations found in the analysis results.")
            return
        
        print(f"Found {len(visualizations)} visualizations")
        
        # Extract each chart
        for i, viz in enumerate(visualizations):
            chart_html = viz.get('chart_html', '')
            chart_title = viz.get('config', {}).get('title', f'Chart_{i+1}')
            chart_type = viz.get('chart_type', 'unknown')
            
            if chart_html:
                # Clean filename
                filename = f"{i+1:02d}_{chart_title.replace(' ', '_').replace('/', '_')}.html"
                filepath = os.path.join(charts_dir, filename)
                
                # Save HTML file
                with open(filepath, 'w') as f:
                    f.write(chart_html)
                
                print(f"✅ Saved: {filename} ({chart_type} chart)")
            else:
                print(f"❌ No HTML data for chart {i+1}: {chart_title}")
        
        # Create an index HTML file
        create_index_html(visualizations, charts_dir)
        
        print(f"\n🎉 All charts extracted to '{charts_dir}' directory")
        print(f"📖 Open '{charts_dir}/index.html' to view all charts")
        
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please run the analysis first.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

def create_index_html(visualizations, charts_dir):
    """
    Create an index HTML file with links to all charts
    
    Args:
        visualizations (list): List of visualization objects
        charts_dir (str): Directory where charts are saved
    """
    index_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Analysis Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .chart-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .chart-card:hover {{
            transform: translateY(-5px);
        }}
        .chart-title {{
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .chart-description {{
            color: #7f8c8d;
            margin-bottom: 15px;
            font-size: 0.9em;
        }}
        .chart-type {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-bottom: 15px;
        }}
        .chart-link {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            transition: opacity 0.3s ease;
        }}
        .chart-link:hover {{
            opacity: 0.8;
        }}
        .insights {{
            background: #e8f5e8;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.85em;
        }}
        .insights ul {{
            margin: 5px 0;
            padding-left: 20px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Sales Analysis Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="charts-grid">
"""
    
    for i, viz in enumerate(visualizations):
        chart_title = viz.get('config', {}).get('title', f'Chart {i+1}')
        chart_type = viz.get('chart_type', 'unknown')
        chart_description = viz.get('config', {}).get('description', 'No description available')
        insights = viz.get('insights', [])
        filename = f"{i+1:02d}_{chart_title.replace(' ', '_').replace('/', '_')}.html"
        
        insights_html = ""
        if insights:
            insights_html = f"""
            <div class="insights">
                <strong>Key Insights:</strong>
                <ul>
                    {"".join([f"<li>{insight}</li>" for insight in insights[:3]])}
                </ul>
            </div>
            """
        
        index_html += f"""
            <div class="chart-card">
                <div class="chart-title">{chart_title}</div>
                <div class="chart-type">{chart_type.upper()}</div>
                <div class="chart-description">{chart_description}</div>
                <a href="{filename}" target="_blank" class="chart-link">View Chart →</a>
                {insights_html}
            </div>
        """
    
    index_html += f"""
        </div>
        
        <div class="footer">
            <p>Generated by AI-Powered EDA Analysis System</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save index file
    with open(os.path.join(charts_dir, 'index.html'), 'w') as f:
        f.write(index_html)

if __name__ == "__main__":
    extract_charts_from_json() 