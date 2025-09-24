#!/usr/bin/env python3
"""
Demo script to show execution in qlib environment
"""
import sys
import os
from datetime import datetime

def main():
    print("="*50)
    print("Demo Script Execution")
    print("="*50)
    print(f"Execution time: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script arguments: {sys.argv}")
    
    # Check environment
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
    
    # Try to import qlib
    try:
        import qlib
        print(f"✅ Qlib is available!")
        print(f"   Version: {qlib.__version__}")
        print(f"   Location: {qlib.__file__}")
    except ImportError as e:
        print(f"❌ Qlib import failed: {e}")
    
    # Try to import other common packages
    packages_to_test = ['pandas', 'numpy', 'flask']
    print("\nPackage availability:")
    for pkg in packages_to_test:
        try:
            __import__(pkg)
            print(f"✅ {pkg}: Available")
        except ImportError:
            print(f"❌ {pkg}: Not available")
    
    return "Script completed successfully!"

if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
    print("="*50)