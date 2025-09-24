#!/usr/bin/env python3
"""
Simple test to verify qlib environment is working
Run this with: python test_qlib_env.py
"""

import sys
import os

def test_environment():
    print("🔍 Testing qlib Environment")
    print("=" * 50)
    
    # Basic Python info
    print(f"✅ Python Version: {sys.version}")
    print(f"✅ Python Executable: {sys.executable}")
    print(f"✅ Current Directory: {os.getcwd()}")
    
    # Environment info
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not set')
    print(f"✅ Conda Environment: {conda_env}")
    
    # Test package imports
    packages = {
        'sys': 'Built-in',
        'os': 'Built-in', 
        'pandas': None,
        'numpy': None,
        'flask': None,
        'jupyter': None,
        'IPython': None
    }
    
    print("\n📦 Package Availability:")
    print("-" * 30)
    
    for package, expected_version in packages.items():
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                version = module.__version__
            elif expected_version:
                version = expected_version
            else:
                version = "Available"
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not available")
    
    # Test Jupyter kernel
    print(f"\n🔬 Testing Jupyter Integration:")
    print("-" * 30)
    
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            print("✅ Running in IPython/Jupyter environment")
        else:
            print("ℹ️  Running in standard Python (not Jupyter)")
    except:
        print("ℹ️  IPython not available or not in interactive mode")
    
    print(f"\n🎯 Environment Summary:")
    print("-" * 30)
    if conda_env == 'qlib':
        print("✅ Correct qlib environment detected!")
        print("✅ Ready for Jupyter notebook execution")
        return True
    else:
        print("❌ Not in qlib environment")
        print(f"   Current: {conda_env}")
        print("   Expected: qlib")
        return False

if __name__ == "__main__":
    success = test_environment()
    print(f"\n🏁 Test Result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)