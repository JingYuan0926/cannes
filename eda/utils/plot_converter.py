"""
Plot Converter Utility

This module converts JSON plot data to PNG images using Plotly and Kaleido.
It can process individual plot files or batch convert all plots in a directory.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlotConverter:
    """
    Convert JSON plot data to PNG images
    """
    
    def __init__(self, output_dir: str = 'images'):
        """
        Initialize the plot converter
        
        Args:
            output_dir (str): Directory to save PNG images
        """
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # Set up plotly for static image export
        try:
            # Try to use kaleido for better image generation
            pio.kaleido.scope.mathjax = None
            logger.info("Kaleido engine initialized for image export")
        except Exception as e:
            logger.warning(f"Kaleido initialization failed: {str(e)}")
            logger.info("Falling back to default image export")
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def convert_json_to_png(self, json_file_path: str, 
                           width: int = 1200, height: int = 800,
                           scale: float = 2.0) -> Optional[str]:
        """
        Convert a single JSON plot file to PNG
        
        Args:
            json_file_path (str): Path to the JSON plot file
            width (int): Image width in pixels
            height (int): Image height in pixels
            scale (float): Scale factor for image resolution
            
        Returns:
            str: Path to the generated PNG file, or None if failed
        """
        try:
            # Load JSON plot data
            with open(json_file_path, 'r') as f:
                plot_data = json.load(f)
            
            # Extract plot information
            chart_json = plot_data.get('chart_json', '')
            chart_type = plot_data.get('chart_type', 'unknown')
            title = plot_data.get('title', 'Chart')
            
            if not chart_json:
                logger.error(f"No chart JSON data found in {json_file_path}")
                return None
            
            # Parse the chart JSON
            chart_data = json.loads(chart_json)
            
            # Create plotly figure
            fig = go.Figure(data=chart_data.get('data', []), 
                          layout=chart_data.get('layout', {}))
            
            # Update layout for better PNG export
            fig.update_layout(
                width=width,
                height=height,
                font=dict(size=12),
                title=dict(font=dict(size=16)),
                showlegend=True if chart_type in ['pie', 'scatter'] else False,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            # Generate output filename
            base_name = Path(json_file_path).stem
            png_filename = f"{base_name}.png"
            png_path = os.path.join(self.output_dir, png_filename)
            
            # Convert to PNG
            fig.write_image(png_path, width=width, height=height, scale=scale)
            
            logger.info(f"✅ Converted {json_file_path} -> {png_path}")
            return png_path
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {json_file_path}: {str(e)}")
            return None
    
    def convert_all_plots(self, plots_dir: str = 'plots', 
                         width: int = 1200, height: int = 800,
                         scale: float = 2.0) -> List[str]:
        """
        Convert all JSON plot files in a directory to PNG
        
        Args:
            plots_dir (str): Directory containing JSON plot files
            width (int): Image width in pixels
            height (int): Image height in pixels
            scale (float): Scale factor for image resolution
            
        Returns:
            List[str]: List of paths to generated PNG files
        """
        if not os.path.exists(plots_dir):
            logger.error(f"Plots directory {plots_dir} does not exist")
            return []
        
        # Find all JSON files
        json_files = [f for f in os.listdir(plots_dir) if f.endswith('.json')]
        
        if not json_files:
            logger.warning(f"No JSON plot files found in {plots_dir}")
            return []
        
        logger.info(f"Found {len(json_files)} JSON plot files to convert")
        
        # Convert each file
        converted_files = []
        for json_file in sorted(json_files):
            json_path = os.path.join(plots_dir, json_file)
            png_path = self.convert_json_to_png(json_path, width, height, scale)
            if png_path:
                converted_files.append(png_path)
        
        logger.info(f"🎉 Successfully converted {len(converted_files)} plots to PNG")
        return converted_files
    
    def create_summary_report(self, converted_files: List[str]) -> str:
        """
        Create a summary report of converted images
        
        Args:
            converted_files (List[str]): List of converted PNG file paths
            
        Returns:
            str: Path to the summary HTML report
        """
        if not converted_files:
            return ""
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plot Conversion Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
        }}
        .header p {{
            color: #7f8c8d;
            margin: 10px 0 0 0;
        }}
        .images-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .image-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            background: #fafafa;
        }}
        .image-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .image-title {{
            font-weight: bold;
            margin: 10px 0 5px 0;
            color: #2c3e50;
        }}
        .image-path {{
            font-size: 0.9em;
            color: #7f8c8d;
            font-family: monospace;
        }}
        .stats {{
            background: #e8f5e8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Plot Conversion Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="stats">
            <strong>Total Images Generated: {len(converted_files)}</strong>
        </div>
        
        <div class="images-grid">
"""
        
        for i, png_path in enumerate(converted_files):
            filename = os.path.basename(png_path)
            title = filename.replace('.png', '').replace('_', ' ').title()
            
            html_content += f"""
            <div class="image-card">
                <div class="image-title">{title}</div>
                <img src="{filename}" alt="{title}">
                <div class="image-path">{filename}</div>
            </div>
            """
        
        html_content += """
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'conversion_report.html')
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📄 Summary report saved to {report_path}")
        return report_path
    
    def get_image_info(self, png_path: str) -> Dict:
        """
        Get information about a generated PNG image
        
        Args:
            png_path (str): Path to the PNG file
            
        Returns:
            Dict: Image information
        """
        try:
            from PIL import Image
            
            with Image.open(png_path) as img:
                return {
                    'filename': os.path.basename(png_path),
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': os.path.getsize(png_path)
                }
        except Exception as e:
            logger.error(f"Error getting image info for {png_path}: {str(e)}")
            return {}

def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSON plots to PNG images')
    parser.add_argument('--plots-dir', default='plots', help='Directory containing JSON plot files')
    parser.add_argument('--output-dir', default='images', help='Output directory for PNG files')
    parser.add_argument('--width', type=int, default=1200, help='Image width in pixels')
    parser.add_argument('--height', type=int, default=800, help='Image height in pixels')
    parser.add_argument('--scale', type=float, default=2.0, help='Scale factor for resolution')
    parser.add_argument('--single-file', help='Convert a single JSON file')
    
    args = parser.parse_args()
    
    # Create converter
    converter = PlotConverter(output_dir=args.output_dir)
    
    if args.single_file:
        # Convert single file
        png_path = converter.convert_json_to_png(
            args.single_file, args.width, args.height, args.scale
        )
        if png_path:
            print(f"✅ Converted to: {png_path}")
        else:
            print("❌ Conversion failed")
    else:
        # Convert all files
        converted_files = converter.convert_all_plots(
            args.plots_dir, args.width, args.height, args.scale
        )
        
        if converted_files:
            # Create summary report
            report_path = converter.create_summary_report(converted_files)
            print(f"📄 Summary report: {report_path}")
        else:
            print("❌ No files were converted")

if __name__ == "__main__":
    main() 