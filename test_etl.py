import requests
import json
import os
from pathlib import Path
from datetime import datetime

def test_etl_with_csv():
    # Path to your CSV file
    csv_file_path = "us_house_Sales_data.csv"  # Adjust path as needed
    
    # ETL service URL
    url = "http://localhost:3030/analyze"
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"etl_test_results_{timestamp}.txt"
    
    try:
        # Open and send the CSV file
        with open(csv_file_path, 'rb') as file:
            files = {'file': file}
            data = {'goal': 'Clean and analyze housing data'}
            
            print("Sending CSV to ETL service...")
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Write to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("="*50 + "\n")
                    f.write("ETL PROCESS RESULTS\n")
                    f.write("="*50 + "\n")
                    f.write(f"Test run at: {datetime.now()}\n")
                    f.write(f"CSV file: {csv_file_path}\n")
                    f.write("="*50 + "\n\n")
                    
                    # Write full JSON response
                    f.write("FULL JSON RESPONSE:\n")
                    f.write("-" * 30 + "\n")
                    f.write(json.dumps(result, indent=2))
                    f.write("\n\n")
                    
                    # Write summary
                    f.write("="*50 + "\n")
                    f.write("SUMMARY\n")
                    f.write("="*50 + "\n")
                    f.write(f"Status: {result['status']}\n")
                    f.write(f"Original shape: {result['original_shape']}\n")
                    f.write(f"Cleaned shape: {result['cleaned_shape']}\n")
                    f.write(f"Number of records: {len(result['processed_data'])}\n")
                    f.write(f"Number of cleaning steps: {len(result['execution_log'])}\n")
                    
                    # Write first few records as sample
                    f.write("\nSAMPLE PROCESSED DATA (first 3 records):\n")
                    f.write("-" * 40 + "\n")
                    for i, record in enumerate(result['processed_data'][:3]):
                        f.write(f"Record {i+1}:\n")
                        f.write(json.dumps(record, indent=2))
                        f.write("\n\n")
                    
                    # Write analysis summary
                    f.write("ANALYSIS SUMMARY:\n")
                    f.write("-" * 20 + "\n")
                    if 'analysis' in result:
                        f.write(f"Basic info: {result['analysis'].get('basic_info', {})}\n")
                        f.write(f"Missing data: {result['analysis'].get('missing_data', {})}\n")
                        f.write(f"Duplicates: {result['analysis'].get('duplicates', {})}\n")
                    
                    # Write recommendations
                    f.write("\nRECOMMENDATIONS:\n")
                    f.write("-" * 15 + "\n")
                    if 'recommendations' in result:
                        f.write(json.dumps(result['recommendations'], indent=2))
                    
                    # Write execution log
                    f.write("\n\nEXECUTION LOG:\n")
                    f.write("-" * 15 + "\n")
                    if result['execution_log']:
                        f.write(json.dumps(result['execution_log'], indent=2))
                    else:
                        f.write("No cleaning steps were executed (data was already clean)\n")
                
                print(f"\n✅ Test completed successfully!")
                print(f"📄 Full results saved to: {output_file}")
                print(f"📊 Summary:")
                print(f"   - Status: {result['status']}")
                print(f"   - Original shape: {result['original_shape']}")
                print(f"   - Cleaned shape: {result['cleaned_shape']}")
                print(f"   - Records processed: {len(result['processed_data'])}")
                print(f"   - Cleaning steps: {len(result['execution_log'])}")
                
            else:
                error_msg = f"Error: {response.status_code}\n{response.text}"
                print(error_msg)
                
                # Save error to file too
                with open(f"etl_error_{timestamp}.txt", 'w') as f:
                    f.write(error_msg)
                
    except FileNotFoundError:
        error_msg = f"CSV file not found: {csv_file_path}"
        print(error_msg)
        with open(f"etl_error_{timestamp}.txt", 'w') as f:
            f.write(error_msg)
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to ETL service. Make sure it's running on localhost:3030"
        print(error_msg)
        with open(f"etl_error_{timestamp}.txt", 'w') as f:
            f.write(error_msg)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        with open(f"etl_error_{timestamp}.txt", 'w') as f:
            f.write(error_msg)

if __name__ == "__main__":
    test_etl_with_csv()