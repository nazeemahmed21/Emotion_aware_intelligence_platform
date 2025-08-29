#!/usr/bin/env python3
"""
Emotion-Aware Voice Intelligence Platform Launcher
Professional launcher script for the enterprise application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import streamlit
        import numpy
        import pandas
        import plotly
        import soundfile
        import aiohttp
        from dotenv import load_dotenv
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_environment():
    """Check if environment is properly configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('HUME_API_KEY')
    if not api_key:
        print("❌ HUME_API_KEY not found in environment variables")
        print("Please add HUME_API_KEY=your_key_here to your .env file")
        return False
    
    print(f"✅ HUME_API_KEY configured: {api_key[:10]}...")
    return True

def main():
    """Main launcher function"""
    print("🧠 Emotion-Aware Voice Intelligence Platform")
    print("=" * 50)
    
    # Check requirements
    print("📦 Checking dependencies...")
    if not check_requirements():
        print("\n💡 Install missing dependencies:")
        print("pip install -r requirements_production.txt")
        return 1
    
    print("✅ All dependencies installed")
    
    # Check environment
    print("\n🔧 Checking environment configuration...")
    if not check_environment():
        return 1
    
    print("✅ Environment configured correctly")
    
    # Launch application
    print("\n🚀 Launching application...")
    print("Opening in your default browser...")
    print("\n📍 Application will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "emotion_aware_voice_analyzer.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)