#!/usr/bin/env python3
"""
Emotion-Aware Voice Pipeline - Main Runner
Unified entry point for the application
"""
import sys
import os
import argparse
import subprocess

def run_streamlit_app():
    """Run the main Streamlit application"""
    print("🚀 Starting Emotion-Aware Voice Pipeline...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False
    return True

def run_simple_app():
    """Run the simple console application"""
    print("🚀 Starting Simple Voice Pipeline...")
    try:
        subprocess.run([sys.executable, "simple_app.py"], check=True)
    except Exception as e:
        print(f"❌ Error running simple app: {e}")
        return False
    return True

def run_tests():
    """Run all tests"""
    print("🧪 Running all tests...")
    test_dir = "tests"
    if not os.path.exists(test_dir):
        print(f"❌ Tests directory '{test_dir}' not found")
        return False
    
    # Find all test files
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    
    if not test_files:
        print("❌ No test files found")
        return False
    
    passed = 0
    total = len(test_files)
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        try:
            result = subprocess.run([sys.executable, os.path.join(test_dir, test_file)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {test_file} PASSED")
                passed += 1
            else:
                print(f"❌ {test_file} FAILED")
                if result.stderr:
                    print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"❌ {test_file} ERROR: {e}")
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    return passed == total

def setup_environment():
    """Setup and validate environment"""
    print("🔧 Setting up environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if virtual environment is active
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected. Consider using one.")
    
    # Check required packages
    required_packages = ['streamlit', 'torch', 'transformers', 'librosa', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ Environment setup complete")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Emotion-Aware Voice Pipeline")
    parser.add_argument('mode', nargs='?', default='app', 
                       choices=['app', 'simple', 'test', 'setup'],
                       help='Run mode (default: app)')
    parser.add_argument('--no-setup', action='store_true',
                       help='Skip environment setup check')
    
    args = parser.parse_args()
    
    print("🎤 Emotion-Aware Voice Pipeline")
    print("=" * 40)
    
    # Setup environment unless skipped
    if not args.no_setup:
        if not setup_environment():
            print("\n❌ Environment setup failed")
            sys.exit(1)
    
    # Run selected mode
    success = False
    if args.mode == 'app':
        success = run_streamlit_app()
    elif args.mode == 'simple':
        success = run_simple_app()
    elif args.mode == 'test':
        success = run_tests()
    elif args.mode == 'setup':
        success = True  # Already ran setup
        print("\n✅ Setup complete!")
    
    if success:
        print("\n🎉 Operation completed successfully!")
    else:
        print("\n❌ Operation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()