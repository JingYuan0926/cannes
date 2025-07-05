#!/usr/bin/env python3
"""
Test Runner for AI Data Analysis Pipeline
This script runs all tests with various options and generates reports.
"""

import unittest
import sys
import os
import argparse
from datetime import datetime
import subprocess


def run_unit_tests(verbose=False):
    """Run unit tests"""
    print("🧪 Running Unit Tests...")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Unit Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests"""
    print("\n🔗 Running Integration Tests...")
    print("=" * 50)
    
    try:
        # Run the setup test
        result = subprocess.run([sys.executable, 'test_setup.py'], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running integration tests: {e}")
        return False


def run_api_tests():
    """Run API tests"""
    print("\n🌐 Running API Tests...")
    print("=" * 50)
    print("Note: Make sure your Flask app is running before running API tests!")
    
    try:
        # Check if user wants to run API tests
        response = input("Is your Flask app running? (y/n): ").lower()
        if response != 'y':
            print("⏭️  Skipping API tests. Start your Flask app with 'python app.py' first.")
            return True
        
        # Run API tests
        api_test_path = os.path.join(os.path.dirname(__file__), 'tests', 'test_manual_api.py')
        result = subprocess.run([sys.executable, api_test_path], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running API tests: {e}")
        return False


def run_performance_tests():
    """Run performance tests"""
    print("\n⚡ Running Performance Tests...")
    print("=" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        import time
        from utils.data_analyzer import DataAnalyzer
        from utils.visualization_engine import VisualizationEngine
        
        # Create test datasets of different sizes
        sizes = [100, 1000, 10000]
        results = []
        
        for size in sizes:
            print(f"Testing with {size} rows...")
            
            # Create test data
            data = pd.DataFrame({
                'id': range(size),
                'value': np.random.randn(size),
                'category': np.random.choice(['A', 'B', 'C', 'D'], size),
                'date': pd.date_range('2020-01-01', periods=size, freq='D')
            })
            
            # Test data analysis performance
            analyzer = DataAnalyzer()
            start_time = time.time()
            analysis = analyzer.comprehensive_analysis(data)
            analysis_time = time.time() - start_time
            
            # Test visualization performance
            viz_engine = VisualizationEngine()
            config = {'chart_type': 'bar', 'x_axis': 'category', 'y_axis': 'value'}
            start_time = time.time()
            viz_result = viz_engine.create_visualization(data, config)
            viz_time = time.time() - start_time
            
            results.append({
                'size': size,
                'analysis_time': analysis_time,
                'viz_time': viz_time
            })
            
            print(f"  Analysis: {analysis_time:.2f}s")
            print(f"  Visualization: {viz_time:.2f}s")
        
        # Print performance summary
        print("\n📈 Performance Summary:")
        print("Size\tAnalysis\tVisualization")
        print("-" * 35)
        for result in results:
            print(f"{result['size']}\t{result['analysis_time']:.2f}s\t\t{result['viz_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running performance tests: {e}")
        return False


def generate_test_report(results):
    """Generate a test report"""
    print("\n📋 Generating Test Report...")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
AI Data Analysis Pipeline - Test Report
Generated: {timestamp}

Test Results:
- Unit Tests: {'✅ PASSED' if results.get('unit', False) else '❌ FAILED'}
- Integration Tests: {'✅ PASSED' if results.get('integration', False) else '❌ FAILED'}
- API Tests: {'✅ PASSED' if results.get('api', False) else '❌ FAILED'}
- Performance Tests: {'✅ PASSED' if results.get('performance', False) else '❌ FAILED'}

Overall Status: {'✅ ALL TESTS PASSED' if all(results.values()) else '❌ SOME TESTS FAILED'}
"""
    
    print(report)
    
    # Save report to file
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run tests for AI Data Analysis Pipeline')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--api', action='store_true', help='Run API tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--report', action='store_true', help='Generate test report')
    
    args = parser.parse_args()
    
    print("🚀 AI Data Analysis Pipeline Test Runner")
    print("=" * 50)
    
    results = {}
    
    # If no specific test type is specified, run all tests
    if not any([args.unit, args.integration, args.api, args.performance]):
        print("Running all tests...")
        results['unit'] = run_unit_tests(args.verbose)
        results['integration'] = run_integration_tests()
        results['api'] = run_api_tests()
        results['performance'] = run_performance_tests()
    else:
        # Run specific tests
        if args.unit:
            results['unit'] = run_unit_tests(args.verbose)
        if args.integration:
            results['integration'] = run_integration_tests()
        if args.api:
            results['api'] = run_api_tests()
        if args.performance:
            results['performance'] = run_performance_tests()
    
    # Generate report if requested
    if args.report or len(results) > 1:
        generate_test_report(results)
    
    # Print final summary
    print("\n🎯 Final Summary:")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    if passed == total:
        print("🎉 All tests passed! Your AI Data Analysis Pipeline is ready!")
        return 0
    else:
        print(f"⚠️  {total - passed} out of {total} test suites failed.")
        print("Please check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 